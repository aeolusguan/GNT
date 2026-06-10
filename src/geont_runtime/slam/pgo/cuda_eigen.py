from __future__ import annotations

import torch
from lietorch import SO3, SE3

from .common import (
    EPS,
    CUDA_EIGEN_BACKEND,
    CUDA_EIGEN_SOLVER_INFO,
    LM_DAMPING_DECREASE,
    LM_DAMPING_INCREASE,
    LM_DAMPING_MAX,
    LM_DAMPING_MIN,
    RelativeEdges,
    Sim3PGOResult,
    _apply_delta,
    _apply_rotation_delta,
    _apply_translation_scale_delta,
    _edge_sqrt_information,
    _rotation_residuals,
    _rotation_sqrt_information,
    _scale_prior_residuals,
    _scaled_se3_residuals,
    _translation_residuals,
    _translation_sqrt_information,
)


def _rotation_cuda_blocks(
    rotations: SO3,
    ii: torch.Tensor,
    jj: torch.Tensor,
    meas_rotations: SO3,
    sqrt_info: torch.Tensor,
    robust: torch.Tensor,
):
    """Build weighted rotation blocks for the CUDA fixed-layout solver.

    Returns source/target blocks (E, 3, 3) and weighted residuals (E, 3).
    The CUDA solver stores variables as (N, 3) and keeps the anchor row fixed.
    """
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
    prior_log_s: torch.Tensor,
    sqrt_info: torch.Tensor,
    robust: torch.Tensor,
    scale_prior_diag: float,
):
    """Build weighted translation+scale blocks for the CUDA fixed-layout solver.

    source_block/target_block are (E, 3, 4) with per-node variable order
    [dt(3), dlog_s]. The target scale column is zero because the source-
    normalized translation residual depends on the source node scale.
    """
    from . import cuda_backend as pgo_cuda

    return pgo_cuda.build_translation_scale_blocks(
        poses.data,
        log_s,
        rel_edges.poses.data,
        prior_log_s,
        ii,
        jj,
        sqrt_info,
        robust,
        scale_prior_diag,
    )


def _se3_scale_cuda_blocks(
    poses: SE3,
    log_s: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    rel_edges: RelativeEdges,
    prior_log_s: torch.Tensor,
    sqrt_info: torch.Tensor,
    robust: torch.Tensor,
    scale_prior_diag: float,
):
    """Build weighted SE3+scale blocks for the CUDA fixed-layout solver.

    source_block/target_block are (E, 6, 7) with per-node variable order
    [dt(3), dphi(3), dlog_s]. The target scale column is zero because the GNT
    source-normalized translation residual depends on the source node scale.
    """
    from . import cuda_backend as pgo_cuda

    return pgo_cuda.build_se3_scale_blocks(
        poses.data,
        log_s,
        rel_edges.poses.data,
        prior_log_s,
        ii,
        jj,
        sqrt_info,
        robust,
        scale_prior_diag,
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
    """Rotation-only PGO using CUDA block assembly and cached Eigen sparse solve."""
    from . import cuda_backend as pgo_cuda

    pose_data = poses.data
    device = pose_data.device
    dtype = pose_data.dtype
    n_nodes = pose_data.shape[0]
    sqrt_info = _rotation_sqrt_information(edge_conf)
    rotations = SO3(poses.data[..., 3:7])
    backend_name = CUDA_EIGEN_BACKEND
    solver_info = CUDA_EIGEN_SOLVER_INFO

    if device.type != "cuda" or dtype != torch.float32:
        raise RuntimeError(f"backend='{backend_name}' requires CUDA float32 tensors")

    if n_nodes <= 1:
        return Sim3PGOResult(
            poses=poses.data,
            log_scales=log_s,
            info={
                "success": True,
                "mode": "rotation_only",
                "backend": backend_name,
                **solver_info,
                "n_edges": int(ii.numel()),
                "solver_failures": 0,
            },
        )

    node_ids = torch.arange(n_nodes, device=device)
    optimized_nodes = node_ids[node_ids != anchor]
    initial_cost = None
    accepted = 0
    rejected = 0
    solver_failures = 0
    lm = damping
    eigen_solver = pgo_cuda.RotationEigenSimplicialLLTSolver(ii, jj, n_nodes, anchor)
    unit_robust = torch.ones(ii.numel(), 1, device=device, dtype=dtype)
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
            current_cost = 0.5 * (edge_residual ** 2).sum()
            if initial_cost is None:
                unrobust_residual = edge_residual / robust.clamp_min(EPS)
                initial_cost = float((0.5 * (unrobust_residual ** 2).sum()).cpu())

        improved = False
        step_norm = 0.0
        for _ in range(lm_max_attempts):
            try:
                step_full = eigen_solver.solve(source_block, target_block, edge_residual, lm)
            except RuntimeError:
                solver_failures += 1
                lm = min(lm * LM_DAMPING_INCREASE, LM_DAMPING_MAX)
                continue
            delta = step_full[optimized_nodes]
            cand_rotations = _apply_rotation_delta(rotations, delta, anchor)
            cand_r = _rotation_residuals(cand_rotations, ii, jj, meas_rotations)
            cand_weighted = cand_r * sqrt_info * robust
            cand_cost = 0.5 * (cand_weighted ** 2).sum()
            step_norm = float(step_full.norm().cpu())
            if cand_cost <= current_cost:
                rotations = cand_rotations
                lm = max(lm * LM_DAMPING_DECREASE, LM_DAMPING_MIN)
                accepted += 1
                improved = True
                break
            rejected += 1
            lm = min(lm * LM_DAMPING_INCREASE, LM_DAMPING_MAX)

        if not improved:
            break

        if step_norm < 1e-5:
            break

    poses = SE3(torch.cat((pose_data[:, :3], rotations.data), dim=-1))
    final_res = _rotation_residuals(rotations, ii, jj, meas_rotations) * sqrt_info
    edge_res = final_res.norm(dim=-1)
    final_cost = float((0.5 * (final_res ** 2).sum()).cpu())
    finite = bool(torch.isfinite(final_res).all().cpu())
    start_cost = final_cost if initial_cost is None else initial_cost
    info = {
        "success": finite and (n_iters == 0 or accepted > 0 or final_cost <= start_cost + 1e-6),
        "mode": "rotation_only",
        "backend": backend_name,
        **solver_info,
        "n_edges": int(ii.numel()),
        "n_iters": int(n_iters),
        "lm_max_attempts": int(lm_max_attempts),
        "accepted_iters": accepted,
        "rejected_attempts": rejected,
        "solver_failures": solver_failures,
        "initial_cost": start_cost,
        "cost": final_cost,
        "edge_residual_mean": float(edge_res.mean().cpu()),
        "edge_residual_max": float(edge_res.max().cpu()),
        "rotation_residual_mean": float(edge_res.mean().cpu()),
        "rotation_residual_max": float(edge_res.max().cpu()),
        "scale_prior_residual_mean": 0.0,
        "scale_prior_residual_max": 0.0,
    }
    return Sim3PGOResult(poses=poses.data, log_scales=log_s, info=info)


def _optimize_translation_scale_pose_graph_cuda(
    poses: SE3,
    log_s: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    rel_edges: RelativeEdges,
    edge_conf: torch.Tensor,
    prior_log_s: torch.Tensor,
    anchor: int,
    n_iters: int,
    damping: float,
    lm_max_attempts: int,
    huber_delta: float,
    scale_conf: float,
) -> Sim3PGOResult:
    """Translation+scale PGO using CUDA block assembly and cached Eigen sparse solve."""
    from . import cuda_backend as pgo_cuda

    device = poses.data.device
    dtype = poses.data.dtype
    n_nodes = poses.data.shape[0]
    sqrt_info = _translation_sqrt_information(edge_conf)
    scale_sqrt_info = torch.as_tensor(float(scale_conf), device=device, dtype=dtype).clamp_min(1e-6).sqrt()
    scale_prior_diag = float((scale_sqrt_info * scale_sqrt_info).cpu())
    backend_name = CUDA_EIGEN_BACKEND
    solver_info = CUDA_EIGEN_SOLVER_INFO

    if device.type != "cuda" or dtype != torch.float32:
        raise RuntimeError(f"backend='{backend_name}' requires CUDA float32 tensors")

    node_ids = torch.arange(n_nodes, device=device)
    optimized_nodes = node_ids[node_ids != anchor]
    initial_cost = None
    accepted = 0
    rejected = 0
    solver_failures = 0
    lm = damping
    eigen_solver = pgo_cuda.TranslationScaleEigenSimplicialLLTSolver(ii, jj, n_nodes, anchor)
    unit_robust = torch.ones(ii.numel(), 1, device=device, dtype=dtype)

    for _ in range(n_iters):
        with torch.no_grad():
            p0 = _scale_prior_residuals(log_s, prior_log_s) * scale_sqrt_info
            source_block, target_block, edge_residual, prior_gradient = _translation_scale_cuda_blocks(
                poses,
                log_s,
                ii,
                jj,
                rel_edges,
                prior_log_s,
                sqrt_info,
                unit_robust,
                scale_prior_diag,
            )
            edge_norm = edge_residual.norm(dim=-1).clamp_min(EPS)
            robust = torch.where(edge_norm <= huber_delta, torch.ones_like(edge_norm), huber_delta / edge_norm)
            robust = robust.sqrt().unsqueeze(-1)
            source_block = source_block * robust[:, :, None]
            target_block = target_block * robust[:, :, None]
            edge_residual = edge_residual * robust
            current_cost = 0.5 * ((edge_residual ** 2).sum() + (p0 ** 2).sum())
            if initial_cost is None:
                unrobust_residual = edge_residual / robust.clamp_min(EPS)
                initial_cost = float((0.5 * ((unrobust_residual ** 2).sum() + (p0 ** 2).sum())).cpu())

        improved = False
        step_norm = 0.0
        for _ in range(lm_max_attempts):
            try:
                step_full = eigen_solver.solve(
                    source_block,
                    target_block,
                    edge_residual,
                    prior_gradient,
                    lm,
                    scale_prior_diag,
                )
            except RuntimeError:
                solver_failures += 1
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
            cand_p = _scale_prior_residuals(cand_log_s, prior_log_s) * scale_sqrt_info
            cand_weighted = cand_r * sqrt_info * robust
            cand_cost = 0.5 * ((cand_weighted ** 2).sum() + (cand_p ** 2).sum())
            step_norm = float(step_full.norm().cpu())
            if cand_cost <= current_cost:
                poses, log_s = cand_poses, cand_log_s
                lm = max(lm * LM_DAMPING_DECREASE, LM_DAMPING_MIN)
                accepted += 1
                improved = True
                break
            rejected += 1
            lm = min(lm * LM_DAMPING_INCREASE, LM_DAMPING_MAX)

        if not improved:
            break

        if step_norm < 1e-5:
            break

    final_res = _translation_residuals(poses, log_s, ii, jj, rel_edges) * sqrt_info
    final_prior = _scale_prior_residuals(log_s, prior_log_s) * scale_sqrt_info
    edge_res = final_res.norm(dim=-1)
    final_cost = float((0.5 * ((final_res ** 2).sum() + (final_prior ** 2).sum())).cpu())
    finite = bool((torch.isfinite(final_res).all() & torch.isfinite(final_prior).all()).cpu())
    start_cost = final_cost if initial_cost is None else initial_cost
    info = {
        "success": finite and (n_iters == 0 or accepted > 0 or final_cost <= start_cost + 1e-6),
        "mode": "translation_scale",
        "backend": backend_name,
        **solver_info,
        "n_edges": int(ii.numel()),
        "n_iters": int(n_iters),
        "lm_max_attempts": int(lm_max_attempts),
        "accepted_iters": accepted,
        "rejected_attempts": rejected,
        "solver_failures": solver_failures,
        "initial_cost": start_cost,
        "cost": final_cost,
        "edge_residual_mean": float(edge_res.mean().cpu()),
        "edge_residual_max": float(edge_res.max().cpu()),
        "translation_residual_mean": float(edge_res.mean().cpu()),
        "translation_residual_max": float(edge_res.max().cpu()),
        "scale_prior_residual_mean": float(final_prior.abs().mean().cpu()),
        "scale_prior_residual_max": float(final_prior.abs().max().cpu()),
    }
    return Sim3PGOResult(poses=poses.data, log_scales=log_s, info=info)


def _optimize_se3_scale_pose_graph_cuda(
    poses: SE3,
    log_s: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    rel_edges: RelativeEdges,
    edge_conf: torch.Tensor,
    prior_log_s: torch.Tensor,
    anchor: int,
    n_iters: int,
    damping: float,
    lm_max_attempts: int,
    huber_delta: float,
    scale_conf: float,
) -> Sim3PGOResult:
    """SE3+scale PGO using CUDA block assembly and cached Eigen sparse solve."""
    from . import cuda_backend as pgo_cuda

    device = poses.data.device
    dtype = poses.data.dtype
    n_nodes = poses.data.shape[0]
    sqrt_info = _edge_sqrt_information(edge_conf)
    scale_sqrt_info = torch.as_tensor(float(scale_conf), device=device, dtype=dtype).clamp_min(1e-6).sqrt()
    scale_prior_diag = float((scale_sqrt_info * scale_sqrt_info).cpu())
    backend_name = CUDA_EIGEN_BACKEND
    solver_info = CUDA_EIGEN_SOLVER_INFO

    if device.type != "cuda" or dtype != torch.float32:
        raise RuntimeError(f"backend='{backend_name}' requires CUDA float32 tensors")

    if n_nodes <= 1:
        return Sim3PGOResult(
            poses=poses.data,
            log_scales=log_s,
            info={
                "success": True,
                "mode": "se3_scale",
                "backend": backend_name,
                **solver_info,
                "n_edges": int(ii.numel()),
                "solver_failures": 0,
            },
        )

    node_ids = torch.arange(n_nodes, device=device)
    optimized_nodes = node_ids[node_ids != anchor]
    initial_cost = None
    accepted = 0
    rejected = 0
    solver_failures = 0
    lm = damping
    eigen_solver = pgo_cuda.Se3ScaleEigenSimplicialLLTSolver(ii, jj, n_nodes, anchor)
    unit_robust = torch.ones(ii.numel(), 1, device=device, dtype=dtype)
    for _ in range(n_iters):
        with torch.no_grad():
            p0 = _scale_prior_residuals(log_s, prior_log_s) * scale_sqrt_info
            source_block, target_block, edge_residual, prior_gradient = _se3_scale_cuda_blocks(
                poses,
                log_s,
                ii,
                jj,
                rel_edges,
                prior_log_s,
                sqrt_info,
                unit_robust,
                scale_prior_diag,
            )
            edge_norm = edge_residual.norm(dim=-1).clamp_min(EPS)
            robust = torch.where(edge_norm <= huber_delta, torch.ones_like(edge_norm), huber_delta / edge_norm)
            robust = robust.sqrt().unsqueeze(-1)
            source_block = source_block * robust[:, :, None]
            target_block = target_block * robust[:, :, None]
            edge_residual = edge_residual * robust
            current_cost = 0.5 * ((edge_residual ** 2).sum() + (p0 ** 2).sum())
            if initial_cost is None:
                unrobust_residual = edge_residual / robust.clamp_min(EPS)
                initial_cost = float((0.5 * ((unrobust_residual ** 2).sum() + (p0 ** 2).sum())).cpu())

        improved = False
        step_norm = 0.0
        for _ in range(lm_max_attempts):
            try:
                step_full = eigen_solver.solve(
                    source_block,
                    target_block,
                    edge_residual,
                    prior_gradient,
                    lm,
                    scale_prior_diag,
                )
            except RuntimeError:
                solver_failures += 1
                lm = min(lm * LM_DAMPING_INCREASE, LM_DAMPING_MAX)
                continue
            pose_delta = step_full[optimized_nodes, :6]
            scale_delta = step_full[:, 6]
            cand_poses, cand_log_s = _apply_delta(poses, log_s, pose_delta, scale_delta, anchor)
            cand_r = _scaled_se3_residuals(cand_poses, cand_log_s, ii, jj, rel_edges)
            cand_p = _scale_prior_residuals(cand_log_s, prior_log_s) * scale_sqrt_info
            cand_weighted = cand_r * sqrt_info * robust
            cand_cost = 0.5 * ((cand_weighted ** 2).sum() + (cand_p ** 2).sum())
            step_norm = float(step_full.norm().cpu())
            if cand_cost <= current_cost:
                poses, log_s = cand_poses, cand_log_s
                lm = max(lm * LM_DAMPING_DECREASE, LM_DAMPING_MIN)
                accepted += 1
                improved = True
                break
            rejected += 1
            lm = min(lm * LM_DAMPING_INCREASE, LM_DAMPING_MAX)

        if not improved:
            break

        if step_norm < 1e-5:
            break

    final_res = _scaled_se3_residuals(poses, log_s, ii, jj, rel_edges) * sqrt_info
    final_prior = _scale_prior_residuals(log_s, prior_log_s) * scale_sqrt_info
    edge_res = final_res.norm(dim=-1)
    final_cost = float((0.5 * ((final_res ** 2).sum() + (final_prior ** 2).sum())).cpu())
    finite = bool((torch.isfinite(final_res).all() & torch.isfinite(final_prior).all()).cpu())
    start_cost = final_cost if initial_cost is None else initial_cost
    info = {
        "success": finite and (n_iters == 0 or accepted > 0 or final_cost <= start_cost + 1e-6),
        "mode": "se3_scale",
        "backend": backend_name,
        **solver_info,
        "n_edges": int(ii.numel()),
        "n_iters": int(n_iters),
        "lm_max_attempts": int(lm_max_attempts),
        "accepted_iters": accepted,
        "rejected_attempts": rejected,
        "solver_failures": solver_failures,
        "initial_cost": start_cost,
        "cost": final_cost,
        "edge_residual_mean": float(edge_res.mean().cpu()),
        "edge_residual_max": float(edge_res.max().cpu()),
        "scale_prior_residual_mean": float(final_prior.abs().mean().cpu()),
        "scale_prior_residual_max": float(final_prior.abs().max().cpu()),
    }
    return Sim3PGOResult(poses=poses.data, log_scales=log_s, info=info)
