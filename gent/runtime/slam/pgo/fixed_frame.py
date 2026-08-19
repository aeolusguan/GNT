from __future__ import annotations

import torch
from lietorch import SE3, SO3

from .common import (
    DEFAULT_LM_MAX_ATTEMPTS,
    EPS,
    LM_DAMPING_DECREASE,
    LM_DAMPING_INCREASE,
    LM_DAMPING_MAX,
    LM_DAMPING_MIN,
    Sim3PGOResult,
    _edge_sqrt_information,
    _scale_sqrt_information,
    _so3_hat,
    _so3_right_jacobian_inverse,
)


def _residuals(
    source_pose: torch.Tensor,
    source_log_scale: torch.Tensor,
    target_poses: torch.Tensor,
    target_log_scales: torch.Tensor,
    relative_poses: torch.Tensor,
    relative_log_scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source_poses = source_pose[None].expand(target_poses.shape[0], -1).contiguous()
    predicted = (SE3(target_poses) * SE3(source_poses).inv()).data
    source_scale = torch.exp(source_log_scale).clamp_min(EPS)
    predicted_translation = predicted[:, :3] / source_scale
    rotation_residual = (
        SO3(relative_poses[:, 3:7]).inv() * SO3(predicted[:, 3:7])
    ).log()
    pose_residual = torch.cat(
        (predicted_translation - relative_poses[:, :3], rotation_residual),
        dim=-1,
    )
    scale_residual = target_log_scales - source_log_scale - relative_log_scales
    return pose_residual, scale_residual, predicted_translation


def _linear_system(
    source_pose: torch.Tensor,
    source_log_scale: torch.Tensor,
    target_poses: torch.Tensor,
    target_log_scales: torch.Tensor,
    relative_poses: torch.Tensor,
    relative_log_scales: torch.Tensor,
    edge_confidence: torch.Tensor,
    robust: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pose_residual, scale_residual, predicted_translation = _residuals(
        source_pose,
        source_log_scale,
        target_poses,
        target_log_scales,
        relative_poses,
        relative_log_scales,
    )
    pose_sqrt_information = _edge_sqrt_information(edge_confidence)
    scale_sqrt_information = _scale_sqrt_information(edge_confidence)
    row_weight = pose_sqrt_information * robust

    source_rotation = SO3(source_pose[3:7]).matrix()[:3, :3]
    target_rotation = SO3(target_poses[:, 3:7]).matrix()[:, :3, :3]
    source_translation_local = source_rotation.transpose(0, 1) @ source_pose[:3]
    inverse_source_scale = torch.exp(source_log_scale).clamp_min(EPS).reciprocal()

    translation_block = target_rotation * inverse_source_scale
    translation_rotation_block = target_rotation @ _so3_hat(
        source_translation_local[None].expand(target_poses.shape[0], -1)
    )
    rotation_block = _so3_right_jacobian_inverse(pose_residual[:, 3:6]) @ source_rotation

    jacobian = torch.zeros(
        target_poses.shape[0],
        6,
        7,
        device=source_pose.device,
        dtype=source_pose.dtype,
    )
    jacobian[:, :3, :3] = -translation_block
    jacobian[:, :3, 3:6] = -translation_rotation_block * inverse_source_scale
    jacobian[:, 3:6, 3:6] = -rotation_block
    jacobian[:, :3, 6] = -predicted_translation
    jacobian = jacobian * row_weight[:, :, None]

    weighted_pose_residual = pose_residual * row_weight
    scale_jacobian = torch.zeros(
        target_poses.shape[0],
        7,
        device=source_pose.device,
        dtype=source_pose.dtype,
    )
    scale_jacobian[:, 6] = -scale_sqrt_information
    weighted_scale_residual = scale_residual * scale_sqrt_information

    flat_jacobian = torch.cat((jacobian.reshape(-1, 7), scale_jacobian), dim=0).double()
    flat_residual = torch.cat(
        (weighted_pose_residual.reshape(-1), weighted_scale_residual),
        dim=0,
    ).double()
    return flat_jacobian.transpose(0, 1) @ flat_jacobian, flat_jacobian.transpose(0, 1) @ flat_residual


def _apply_step(
    source_pose: torch.Tensor,
    source_log_scale: torch.Tensor,
    step: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    step = step.to(device=source_pose.device, dtype=source_pose.dtype)
    pose_step = SE3(
        torch.cat((step[:3], SO3.exp(step[None, 3:6]).data[0]), dim=0)
    )
    return (SE3(source_pose) * pose_step).data, source_log_scale + step[6]


def optimize_fixed_frame_pose_scale(
    source_pose: torch.Tensor,
    source_log_scale: torch.Tensor,
    target_poses: torch.Tensor,
    target_log_scales: torch.Tensor,
    relative_poses: torch.Tensor,
    relative_log_scales: torch.Tensor,
    edge_confidence: torch.Tensor,
    *,
    n_iters: int = 12,
    damping: float = 1e-3,
    lm_max_attempts: int = DEFAULT_LM_MAX_ATTEMPTS,
    huber_delta: float = 1.0,
) -> Sim3PGOResult:
    """Optimize one temporary frame while all target keyframes stay fixed."""
    n_edges = target_poses.shape[0]
    if n_edges == 0:
        raise ValueError("fixed-frame PGO requires at least one target keyframe")
    if target_poses.shape != (n_edges, 7):
        raise ValueError("target_poses must have shape (n_edges, 7)")
    if target_log_scales.shape != (n_edges,):
        raise ValueError("target_log_scales must have shape (n_edges,)")
    if relative_poses.shape != (n_edges, 7):
        raise ValueError("relative_poses must have shape (n_edges, 7)")
    if relative_log_scales.shape != (n_edges,):
        raise ValueError("relative_log_scales must have shape (n_edges,)")
    if edge_confidence.shape != (n_edges, 3):
        raise ValueError("edge_confidence must have shape (n_edges, 3)")
    if n_iters < 1:
        raise ValueError("n_iters must be positive")
    if damping <= 0:
        raise ValueError("damping must be positive")
    if lm_max_attempts < 1:
        raise ValueError("lm_max_attempts must be positive")
    if huber_delta <= 0:
        raise ValueError("huber_delta must be positive")

    source_pose = source_pose.float().clone()
    source_log_scale = source_log_scale.float().reshape(()).clone()
    target_poses = target_poses.to(device=source_pose.device, dtype=source_pose.dtype)
    target_log_scales = target_log_scales.to(device=source_pose.device, dtype=source_pose.dtype)
    relative_poses = relative_poses.to(device=source_pose.device, dtype=source_pose.dtype)
    relative_log_scales = relative_log_scales.to(device=source_pose.device, dtype=source_pose.dtype)
    edge_confidence = edge_confidence.to(device=source_pose.device, dtype=source_pose.dtype)

    pose_sqrt_information = _edge_sqrt_information(edge_confidence)
    scale_sqrt_information = _scale_sqrt_information(edge_confidence)
    identity = torch.eye(7, device=source_pose.device, dtype=torch.float64)
    lm = float(damping)
    accepted = 0
    initial_cost = None

    for _ in range(int(n_iters)):
        pose_residual, scale_residual, _ = _residuals(
            source_pose,
            source_log_scale,
            target_poses,
            target_log_scales,
            relative_poses,
            relative_log_scales,
        )
        weighted_pose = pose_residual * pose_sqrt_information
        edge_norm = weighted_pose.norm(dim=-1).clamp_min(EPS)
        robust = torch.where(
            edge_norm <= huber_delta,
            torch.ones_like(edge_norm),
            huber_delta / edge_norm,
        ).sqrt()[:, None]
        weighted_pose = weighted_pose * robust
        weighted_scale = scale_residual * scale_sqrt_information
        current_cost = 0.5 * (
            weighted_pose.square().sum() + weighted_scale.square().sum()
        )
        if initial_cost is None:
            initial_cost = float(current_cost.cpu())

        hessian, gradient = _linear_system(
            source_pose,
            source_log_scale,
            target_poses,
            target_log_scales,
            relative_poses,
            relative_log_scales,
            edge_confidence,
            robust,
        )

        improved = False
        step_norm = 0.0
        for _ in range(int(lm_max_attempts)):
            chol, info = torch.linalg.cholesky_ex(hessian + lm * identity)
            if bool((info != 0).any()):
                lm = min(lm * LM_DAMPING_INCREASE, LM_DAMPING_MAX)
                continue
            step = -torch.cholesky_solve(gradient[:, None], chol).squeeze(-1)
            candidate_pose, candidate_log_scale = _apply_step(
                source_pose,
                source_log_scale,
                step,
            )
            candidate_pose_residual, candidate_scale_residual, _ = _residuals(
                candidate_pose,
                candidate_log_scale,
                target_poses,
                target_log_scales,
                relative_poses,
                relative_log_scales,
            )
            candidate_cost = 0.5 * (
                (candidate_pose_residual * pose_sqrt_information * robust).square().sum()
                + (candidate_scale_residual * scale_sqrt_information).square().sum()
            )
            step_norm = float(step.norm().cpu())
            if candidate_cost <= current_cost:
                source_pose = candidate_pose
                source_log_scale = candidate_log_scale
                lm = max(lm * LM_DAMPING_DECREASE, LM_DAMPING_MIN)
                accepted += 1
                improved = True
                break
            lm = min(lm * LM_DAMPING_INCREASE, LM_DAMPING_MAX)

        if not improved or step_norm < 1e-5:
            break

    final_pose_residual, final_scale_residual, _ = _residuals(
        source_pose,
        source_log_scale,
        target_poses,
        target_log_scales,
        relative_poses,
        relative_log_scales,
    )
    final_cost = 0.5 * (
        (final_pose_residual * pose_sqrt_information).square().sum()
        + (final_scale_residual * scale_sqrt_information).square().sum()
    )
    finite = torch.isfinite(final_pose_residual).all() & torch.isfinite(final_scale_residual).all()
    return Sim3PGOResult(
        poses=source_pose[None],
        log_scales=source_log_scale[None],
        info={
            "success": bool(finite.cpu()) and (accepted > 0 or float(final_cost.cpu()) <= initial_cost + 1e-6),
            "mode": "fixed_frame_se3_scale",
            "backend": "torch",
            "n_edges": int(n_edges),
            "n_iters": int(n_iters),
            "accepted_iters": int(accepted),
            "cost": float(final_cost.cpu()),
        },
    )
