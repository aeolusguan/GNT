from __future__ import annotations

import torch
from lietorch import SO3, SE3

from .common import (
    EPS,
    LM_DAMPING_DECREASE,
    LM_DAMPING_INCREASE,
    LM_DAMPING_MAX,
    LM_DAMPING_MIN,
    RelativeEdges,
    Sim3PGOResult,
    _apply_delta,
    _apply_rotation_delta,
    _apply_translation_scale_delta,
    _dct_scale_modes,
    _edge_sqrt_information,
    _mode_quadratic_cost,
    _moge_mode_prior_covariance,
    _regularize_mode_covariance,
    _relative_scale_residuals,
    _rotation_residuals,
    _rotation_sqrt_information,
    _scale_sqrt_information,
    _scaled_se3_residuals_cached_rotation,
    _translation_residuals,
    _translation_sqrt_information,
)


CUDA_EIGEN_BACKEND = "cuda_eigen"
def _pgo_info(
    *,
    success: bool,
    mode: str,
    n_edges: int,
    n_iters: int,
    accepted_iters: int,
    cost: float,
) -> dict:
    return {
        "success": bool(success),
        "mode": mode,
        "backend": CUDA_EIGEN_BACKEND,
        "n_edges": int(n_edges),
        "n_iters": int(n_iters),
        "accepted_iters": int(accepted_iters),
        "cost": float(cost),
    }


def _require_cuda_float32(poses: SE3) -> None:
    if poses.data.device.type != "cuda" or poses.data.dtype != torch.float32:
        raise RuntimeError("backend='cuda_eigen' requires CUDA float32 tensors")


def _append_relative_scale_row(
    source_block: torch.Tensor,
    target_block: torch.Tensor,
    edge_residual: torch.Tensor,
    log_s: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    rel_edges: RelativeEdges,
    scale_sqrt_info: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_edges, pose_res_dim, block_dim = source_block.shape
    source = source_block.new_zeros(n_edges, pose_res_dim + 1, block_dim)
    target = target_block.new_zeros(n_edges, pose_res_dim + 1, block_dim)
    residual = edge_residual.new_zeros(n_edges, pose_res_dim + 1)
    source[:, :pose_res_dim] = source_block
    target[:, :pose_res_dim] = target_block
    residual[:, :pose_res_dim] = edge_residual

    scale_col = block_dim - 1
    source[:, pose_res_dim, scale_col] = -scale_sqrt_info
    target[:, pose_res_dim, scale_col] = scale_sqrt_info
    residual[:, pose_res_dim] = _relative_scale_residuals(log_s, ii, jj, rel_edges) * scale_sqrt_info
    return source, target, residual


def _rotation_cuda_blocks(
    rotations: SO3,
    ii: torch.Tensor,
    jj: torch.Tensor,
    meas_rotations: SO3,
    sqrt_info: torch.Tensor,
    robust: torch.Tensor,
):
    from . import cuda_backend as pgo_cuda

    return pgo_cuda.build_rotation_blocks(
        rotations.data,
        meas_rotations.data,
        ii,
        jj,
        sqrt_info,
        robust,
    )


def _translation_scale_cuda_blocks(
    poses: SE3,
    log_s: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    rel_edges: RelativeEdges,
    sqrt_info: torch.Tensor,
    robust: torch.Tensor,
    scale_sqrt_info: torch.Tensor,
):
    from . import cuda_backend as pgo_cuda

    source, target, residual = pgo_cuda.build_translation_scale_blocks(
        poses.data,
        log_s,
        rel_edges.poses.data,
        ii,
        jj,
        sqrt_info,
        robust,
    )
    return _append_relative_scale_row(source, target, residual, log_s, ii, jj, rel_edges, scale_sqrt_info)


def _se3_scale_cuda_blocks(
    poses: SE3,
    log_s: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    rel_edges: RelativeEdges,
    sqrt_info: torch.Tensor,
    robust: torch.Tensor,
    scale_sqrt_info: torch.Tensor,
):
    from . import cuda_backend as pgo_cuda

    source, target, residual = pgo_cuda.build_se3_scale_blocks(
        poses.data,
        log_s,
        rel_edges.poses.data,
        ii,
        jj,
        sqrt_info,
        robust,
    )
    return _append_relative_scale_row(source, target, residual, log_s, ii, jj, rel_edges, scale_sqrt_info)


def _se3_scale_cuda_weighted_blocks(
    poses: SE3,
    log_s: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    rel_edges: RelativeEdges,
    sqrt_info: torch.Tensor,
    huber_delta: float,
    scale_sqrt_info: torch.Tensor,
    buffers: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
):
    from . import cuda_backend as pgo_cuda

    source, target, residual, robust, cost_summary = pgo_cuda.build_se3_scale_weighted_blocks(
        poses.data,
        log_s,
        rel_edges.poses.data,
        rel_edges.log_scales,
        ii,
        jj,
        sqrt_info,
        scale_sqrt_info,
        huber_delta,
        buffers,
    )
    scale_residual = residual[:, -1]
    scale_cost = 0.5 * (scale_residual * scale_residual).sum()
    return (
        source,
        target,
        residual,
        robust,
        cost_summary[0] + scale_cost,
        cost_summary[1] + scale_cost,
    )


def _optimize_rotation_pose_graph_cuda(
    poses: SE3,
    log_s: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    meas_rotations: SO3,
    edge_conf: torch.Tensor,
    anchor: int,
    n_iters: int,
    damping: float,
    lm_max_attempts: int,
    huber_delta: float,
) -> Sim3PGOResult:
    from . import cuda_backend as pgo_cuda

    _require_cuda_float32(poses)
    n_nodes = poses.data.shape[0]
    sqrt_info = _rotation_sqrt_information(edge_conf)
    rotations = SO3(poses.data[..., 3:7])
    if n_nodes <= 1:
        return Sim3PGOResult(
            poses=poses.data,
            log_scales=log_s,
            info=_pgo_info(
                success=True,
                mode="rotation_only",
                n_edges=int(ii.numel()),
                n_iters=0,
                accepted_iters=0,
                cost=0.0,
            ),
        )

    node_ids = torch.arange(n_nodes, device=poses.data.device)
    optimized_nodes = node_ids[node_ids != anchor]
    accepted = 0
    initial_cost = None
    lm = damping
    eigen_solver = pgo_cuda.RotationEigenSimplicialLLTSolver(ii, jj, n_nodes, anchor)
    unit_robust = torch.ones(ii.numel(), 1, device=poses.data.device, dtype=poses.data.dtype)

    for _ in range(n_iters):
        with torch.no_grad():
            source_block, target_block, edge_residual = _rotation_cuda_blocks(
                rotations,
                ii,
                jj,
                meas_rotations,
                sqrt_info,
                unit_robust,
            )
            edge_norm = edge_residual.norm(dim=-1).clamp_min(EPS)
            robust = torch.where(edge_norm <= huber_delta, torch.ones_like(edge_norm), huber_delta / edge_norm)
            robust = robust.sqrt().unsqueeze(-1)
            source_block = source_block * robust[:, :, None]
            target_block = target_block * robust[:, :, None]
            edge_residual = edge_residual * robust
            current_cost = 0.5 * (edge_residual * edge_residual).sum()
            if initial_cost is None:
                unrobust = edge_residual / robust.clamp_min(EPS)
                initial_cost = float((0.5 * (unrobust * unrobust).sum()).cpu())

        improved = False
        step_norm = 0.0
        for _ in range(lm_max_attempts):
            try:
                step_full = eigen_solver.solve(source_block, target_block, edge_residual, lm)
            except RuntimeError:
                lm = min(lm * LM_DAMPING_INCREASE, LM_DAMPING_MAX)
                continue
            delta = step_full[optimized_nodes]
            cand_rotations = _apply_rotation_delta(rotations, delta, anchor)
            cand_r = _rotation_residuals(cand_rotations, ii, jj, meas_rotations)
            cand_cost = 0.5 * ((cand_r * sqrt_info * robust) ** 2).sum()
            step_norm = float(step_full.norm().cpu())
            if cand_cost <= current_cost:
                rotations = cand_rotations
                lm = max(lm * LM_DAMPING_DECREASE, LM_DAMPING_MIN)
                accepted += 1
                improved = True
                break
            lm = min(lm * LM_DAMPING_INCREASE, LM_DAMPING_MAX)

        if not improved or step_norm < 1e-5:
            break

    pose_data = torch.cat((poses.data[:, :3], rotations.data), dim=-1)
    final_res = _rotation_residuals(rotations, ii, jj, meas_rotations) * sqrt_info
    final_cost = float((0.5 * (final_res * final_res).sum()).cpu())
    start_cost = final_cost if initial_cost is None else initial_cost
    success = bool(torch.isfinite(final_res).all().cpu()) and (
        n_iters == 0 or accepted > 0 or final_cost <= start_cost + 1e-6
    )
    return Sim3PGOResult(
        poses=pose_data,
        log_scales=log_s,
        info=_pgo_info(
            success=success,
            mode="rotation_only",
            n_edges=int(ii.numel()),
            n_iters=int(n_iters),
            accepted_iters=accepted,
            cost=final_cost,
        ),
    )


def _optimize_translation_scale_pose_graph_cuda(
    poses: SE3,
    log_s: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    rel_edges: RelativeEdges,
    edge_conf: torch.Tensor,
    anchor: int,
    n_iters: int,
    damping: float,
    lm_max_attempts: int,
    huber_delta: float,
) -> Sim3PGOResult:
    from . import cuda_backend as pgo_cuda

    _require_cuda_float32(poses)
    n_nodes = poses.data.shape[0]
    sqrt_info = _translation_sqrt_information(edge_conf)
    scale_sqrt_info = _scale_sqrt_information(edge_conf)

    node_ids = torch.arange(n_nodes, device=poses.data.device)
    optimized_nodes = node_ids[node_ids != anchor]
    accepted = 0
    initial_cost = None
    lm = damping
    eigen_solver = pgo_cuda.TranslationScaleEigenSimplicialLLTSolver(ii, jj, n_nodes, anchor)
    unit_robust = torch.ones(ii.numel(), 1, device=poses.data.device, dtype=poses.data.dtype)

    for _ in range(n_iters):
        with torch.no_grad():
            source_block, target_block, edge_residual = _translation_scale_cuda_blocks(
                poses,
                log_s,
                ii,
                jj,
                rel_edges,
                sqrt_info,
                unit_robust,
                scale_sqrt_info,
            )
            pose_residual = edge_residual[:, :3]
            edge_norm = pose_residual.norm(dim=-1).clamp_min(EPS)
            robust = torch.where(edge_norm <= huber_delta, torch.ones_like(edge_norm), huber_delta / edge_norm)
            robust = robust.sqrt().unsqueeze(-1)
            source_block[:, :3] = source_block[:, :3] * robust[:, :, None]
            target_block[:, :3] = target_block[:, :3] * robust[:, :, None]
            edge_residual[:, :3] = edge_residual[:, :3] * robust
            current_cost = 0.5 * (edge_residual * edge_residual).sum()
            if initial_cost is None:
                unrobust_pose = edge_residual[:, :3] / robust.clamp_min(EPS)
                scale_residual = edge_residual[:, 3]
                initial_cost = float((0.5 * ((unrobust_pose * unrobust_pose).sum() + (scale_residual * scale_residual).sum())).cpu())

        improved = False
        step_norm = 0.0
        for _ in range(lm_max_attempts):
            try:
                step_full = eigen_solver.solve(source_block, target_block, edge_residual, lm)
            except RuntimeError:
                lm = min(lm * LM_DAMPING_INCREASE, LM_DAMPING_MAX)
                continue
            translation_delta = step_full[optimized_nodes, :3]
            scale_delta = step_full[:, 3]
            cand_poses, cand_log_s = _apply_translation_scale_delta(
                poses,
                log_s,
                translation_delta,
                scale_delta,
                anchor,
            )
            cand_r = _translation_residuals(cand_poses, cand_log_s, ii, jj, rel_edges)
            cand_s = _relative_scale_residuals(cand_log_s, ii, jj, rel_edges) * scale_sqrt_info
            cand_cost = 0.5 * (((cand_r * sqrt_info * robust) ** 2).sum() + (cand_s * cand_s).sum())
            step_norm = float(step_full.norm().cpu())
            if cand_cost <= current_cost:
                poses, log_s = cand_poses, cand_log_s
                lm = max(lm * LM_DAMPING_DECREASE, LM_DAMPING_MIN)
                accepted += 1
                improved = True
                break
            lm = min(lm * LM_DAMPING_INCREASE, LM_DAMPING_MAX)

        if not improved or step_norm < 1e-5:
            break

    final_res = _translation_residuals(poses, log_s, ii, jj, rel_edges) * sqrt_info
    final_scale = _relative_scale_residuals(log_s, ii, jj, rel_edges) * scale_sqrt_info
    final_cost = float((0.5 * ((final_res * final_res).sum() + (final_scale * final_scale).sum())).cpu())
    start_cost = final_cost if initial_cost is None else initial_cost
    success = bool((torch.isfinite(final_res).all() & torch.isfinite(final_scale).all()).cpu()) and (
        n_iters == 0 or accepted > 0 or final_cost <= start_cost + 1e-6
    )
    return Sim3PGOResult(
        poses=poses.data,
        log_scales=log_s,
        info=_pgo_info(
            success=success,
            mode="translation_scale",
            n_edges=int(ii.numel()),
            n_iters=int(n_iters),
            accepted_iters=accepted,
            cost=final_cost,
        ),
    )


def _optimize_se3_scale_pose_graph_cuda(
    poses: SE3,
    log_s: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    rel_edges: RelativeEdges,
    edge_conf: torch.Tensor,
    anchor: int,
    n_iters: int,
    damping: float,
    lm_max_attempts: int,
    huber_delta: float,
    moge_log_s: torch.Tensor,
    moge_mode_nis: bool,
    moge_mode_count: int,
    moge_mode_nis_cutoff: float,
) -> Sim3PGOResult:
    from . import cuda_backend as pgo_cuda

    _require_cuda_float32(poses)
    n_nodes = poses.data.shape[0]
    sqrt_info = _edge_sqrt_information(edge_conf)
    scale_sqrt_info = _scale_sqrt_information(edge_conf)

    node_ids = torch.arange(n_nodes, device=poses.data.device)
    optimized_nodes = node_ids[node_ids != anchor]
    accepted = 0
    initial_cost = None
    lm = damping
    eigen_solver = pgo_cuda.Se3ScaleEigenSimplicialLDLTSolver(ii, jj, n_nodes, anchor)
    n_edges = int(ii.numel())
    weighted_buffers = (
        poses.data.new_empty(n_edges, 7, 7),
        poses.data.new_empty(n_edges, 7, 7),
        poses.data.new_empty(n_edges, 7),
        poses.data.new_empty(n_edges, 1),
        poses.data.new_empty(2),
    )
    meas_rotations_inv = SO3(rel_edges.poses.data[..., 3:7]).inv()
    mode_basis = _dct_scale_modes(n_nodes, moge_mode_count, log_s) if moge_mode_nis else None
    mode_rhs = (
        mode_basis.transpose(0, 1).to(device="cpu", dtype=torch.float64).contiguous()
        if mode_basis is not None
        else None
    )
    mode_identity = (
        torch.eye(mode_basis.shape[0], device=log_s.device, dtype=log_s.dtype)
        if mode_basis is not None
        else None
    )
    mode_residual = mode_basis @ (log_s - moge_log_s) if mode_basis is not None else None
    mode_prior_covariance = None
    mode_prior_precision = None
    mode_current_cost = None
    mode_nis_score = 0.0
    mode_alpha = 0.0

    for _ in range(n_iters):
        with torch.no_grad():
            source_block, target_block, edge_residual, robust, current_cost_value, unrobust_cost = (
                _se3_scale_cuda_weighted_blocks(
                    poses,
                    log_s,
                    ii,
                    jj,
                    rel_edges,
                    sqrt_info,
                    huber_delta,
                    scale_sqrt_info,
                    weighted_buffers,
                )
            )
            base_current_cost = current_cost_value

        improved = False
        step_norm = 0.0
        for _ in range(lm_max_attempts):
            try:
                if mode_basis is None:
                    step_full = eigen_solver.solve(
                        source_block,
                        target_block,
                        edge_residual,
                        lm,
                    )
                    mode_response = None
                else:
                    multi_rhs = eigen_solver.solve_multi_rhs(
                        source_block,
                        target_block,
                        edge_residual,
                        mode_rhs,
                        lm,
                    )
                    step_full = multi_rhs[:, :, 0]
                    mode_response = multi_rhs[:, :, 1:]
            except RuntimeError:
                lm = min(lm * LM_DAMPING_INCREASE, LM_DAMPING_MAX)
                continue
            current_cost = base_current_cost
            if mode_basis is not None:
                mode_covariance = _regularize_mode_covariance(
                    mode_basis @ mode_response[:, 6, :],
                    mode_identity,
                )
                innovation = mode_residual + mode_basis @ step_full[:, 6]
                if mode_prior_covariance is None:
                    mode_prior_covariance, mode_nis_score, mode_alpha = _moge_mode_prior_covariance(
                        mode_covariance,
                        innovation,
                        moge_mode_nis_cutoff,
                    )
                    mode_prior_precision = torch.linalg.inv(mode_prior_covariance)
                if mode_current_cost is None:
                    mode_current_cost = _mode_quadratic_cost(mode_residual, mode_prior_precision)
                middle = mode_prior_covariance + mode_covariance
                correction = torch.linalg.solve(middle, innovation)
                step_full = step_full - torch.einsum("nvk,k->nv", mode_response, correction)
                current_cost = current_cost + mode_current_cost

            if initial_cost is None:
                initial_cost = float(unrobust_cost + (current_cost - base_current_cost).cpu())
            pose_delta = step_full[optimized_nodes, :6]
            scale_delta = step_full[:, 6]
            cand_poses, cand_log_s = _apply_delta(
                poses,
                log_s,
                pose_delta,
                scale_delta,
                optimized_nodes,
            )
            cand_r = _scaled_se3_residuals_cached_rotation(
                cand_poses,
                cand_log_s,
                ii,
                jj,
                rel_edges,
                meas_rotations_inv,
            )
            cand_s = _relative_scale_residuals(cand_log_s, ii, jj, rel_edges) * scale_sqrt_info
            cand_cost = 0.5 * (
                ((cand_r * sqrt_info * robust) ** 2).sum()
                + (cand_s * cand_s).sum()
            )
            if mode_basis is not None:
                cand_moge = cand_log_s - moge_log_s
                cand_mode_residual = mode_basis @ cand_moge
                cand_mode_cost = _mode_quadratic_cost(
                    cand_mode_residual,
                    mode_prior_precision,
                )
                cand_cost = cand_cost + cand_mode_cost
            cand_cost_value, current_cost_value, step_norm = torch.stack(
                (cand_cost, current_cost, step_full.norm())
            ).tolist()
            if cand_cost_value <= current_cost_value:
                poses, log_s = cand_poses, cand_log_s
                if mode_basis is not None:
                    mode_residual = cand_mode_residual
                    mode_current_cost = cand_mode_cost
                lm = max(lm * LM_DAMPING_DECREASE, LM_DAMPING_MIN)
                accepted += 1
                improved = True
                break
            lm = min(lm * LM_DAMPING_INCREASE, LM_DAMPING_MAX)

        if not improved or step_norm < 1e-5:
            break

    final_res = _scaled_se3_residuals_cached_rotation(
        poses,
        log_s,
        ii,
        jj,
        rel_edges,
        meas_rotations_inv,
    ) * sqrt_info
    final_scale = _relative_scale_residuals(log_s, ii, jj, rel_edges) * scale_sqrt_info
    final_cost_tensor = 0.5 * (
        (final_res * final_res).sum()
        + (final_scale * final_scale).sum()
    )
    if mode_basis is not None and mode_prior_covariance is not None:
        final_cost_tensor = final_cost_tensor + mode_current_cost
    final_cost = float(final_cost_tensor.cpu())
    start_cost = final_cost if initial_cost is None else initial_cost
    success = bool((torch.isfinite(final_res).all() & torch.isfinite(final_scale).all()).cpu()) and (
        n_iters == 0 or accepted > 0 or final_cost <= start_cost + 1e-6
    )
    info = _pgo_info(
        success=success,
        mode="se3_scale",
        n_edges=int(ii.numel()),
        n_iters=int(n_iters),
        accepted_iters=accepted,
        cost=final_cost,
    )
    info.update(
        moge_mode_nis=bool(moge_mode_nis),
        moge_mode_count=0 if mode_basis is None else int(mode_basis.shape[0]),
        moge_mode_nis_score=float(mode_nis_score),
        moge_mode_alpha=float(mode_alpha),
    )
    return Sim3PGOResult(
        poses=poses.data,
        log_scales=log_s,
        info=info,
    )
