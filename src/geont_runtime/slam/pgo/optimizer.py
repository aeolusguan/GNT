from __future__ import annotations

import time

import torch
from lietorch import SO3, SE3

from .common import (
    CUDA_EIGEN_BACKEND,
    CUDA_EIGEN_SOLVER_INFO,
    DEFAULT_LM_MAX_ATTEMPTS,
    PGO_MODES,
    TORCH_SOLVER_INFO,
    RelativeEdges,
    Sim3PGOResult,
    _apply_delta,
    _apply_rotation_delta,
    _apply_translation_scale_delta,
    _edge_sqrt_information,
    _initial_from_edges,
    _rotation_residuals,
    _rotation_sqrt_information,
    _scale_prior_residuals,
    _scaled_se3_residuals,
    _so3_hat,
    _so3_right_jacobian_inverse,
    _translation_residuals,
    _translation_sqrt_information,
)
from .cuda_eigen import (
    _optimize_rotation_pose_graph_cuda,
    _optimize_se3_scale_pose_graph_cuda,
    _optimize_translation_scale_pose_graph_cuda,
    _rotation_cuda_blocks,
    _se3_scale_cuda_blocks,
    _translation_scale_cuda_blocks,
)
from .torch_backend import (
    _optimize_rotation_pose_graph,
    _optimize_se3_scale_pose_graph,
    _optimize_translation_scale_pose_graph,
    _rotation_linear_system,
    _se3_scale_linear_system,
    _solve_lm_step,
    _translation_scale_linear_system,
)


def _optimize_staged_pose_graph(
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
    backend: str,
) -> Sim3PGOResult:
    """Variant A: rotation PGO, then fixed-rotation translation+scale PGO."""
    meas_rotations = SO3(rel_edges.poses.data[..., 3:7])
    if backend == CUDA_EIGEN_BACKEND:
        rotation_result = _optimize_rotation_pose_graph_cuda(
            poses,
            log_s,
            ii,
            jj,
            meas_rotations,
            edge_conf,
            anchor,
            n_iters,
            damping,
            lm_max_attempts,
            huber_delta,
        )
    else:
        rotation_result = _optimize_rotation_pose_graph(
            poses,
            log_s,
            ii,
            jj,
            meas_rotations,
            edge_conf,
            anchor,
            n_iters,
            damping,
            lm_max_attempts,
            huber_delta,
        )

    if backend == CUDA_EIGEN_BACKEND:
        translation_result = _optimize_translation_scale_pose_graph_cuda(
            SE3(rotation_result.poses),
            rotation_result.log_scales,
            ii,
            jj,
            rel_edges,
            edge_conf,
            prior_log_s,
            anchor,
            n_iters,
            damping,
            lm_max_attempts,
            huber_delta,
            scale_conf,
        )
    else:
        translation_result = _optimize_translation_scale_pose_graph(
            SE3(rotation_result.poses),
            rotation_result.log_scales,
            ii,
            jj,
            rel_edges,
            edge_conf,
            prior_log_s,
            anchor,
            n_iters,
            damping,
            lm_max_attempts,
            huber_delta,
            scale_conf,
        )

    final_poses = SE3(translation_result.poses)
    edge_sqrt_info = _edge_sqrt_information(edge_conf)
    scale_sqrt_info = torch.as_tensor(
        float(scale_conf),
        device=translation_result.log_scales.device,
        dtype=translation_result.log_scales.dtype,
    ).clamp_min(1e-6).sqrt()
    final_res = _scaled_se3_residuals(final_poses, translation_result.log_scales, ii, jj, rel_edges) * edge_sqrt_info
    final_prior = _scale_prior_residuals(translation_result.log_scales, prior_log_s) * scale_sqrt_info
    edge_res = final_res.norm(dim=-1)
    finite = bool((torch.isfinite(final_res).all() & torch.isfinite(final_prior).all()).cpu())

    info = {
        "success": finite and bool(rotation_result.info["success"]) and bool(translation_result.info["success"]),
        "mode": "staged",
        "backend": backend,
        "linear_solver": "staged_cholesky",
        "normal_equation_assembly": (
            f"rotation:{rotation_result.info['normal_equation_assembly']};"
            f"translation_scale:{translation_result.info['normal_equation_assembly']}"
        ),
        "linear_solver_impl": (
            f"rotation:{rotation_result.info['linear_solver_impl']};"
            f"translation_scale:{translation_result.info['linear_solver_impl']}"
        ),
        "stage_order": ["rotation_only", "translation_scale"],
        "stage_info": {
            "rotation_only": rotation_result.info,
            "translation_scale": translation_result.info,
        },
        "n_edges": int(ii.numel()),
        "n_iters": int(n_iters),
        "lm_max_attempts": int(lm_max_attempts),
        "accepted_iters": int(rotation_result.info["accepted_iters"]) + int(translation_result.info["accepted_iters"]),
        "rejected_attempts": int(rotation_result.info["rejected_attempts"])
        + int(translation_result.info["rejected_attempts"]),
        "solver_failures": int(rotation_result.info["solver_failures"]) + int(translation_result.info["solver_failures"]),
        "initial_cost": float(rotation_result.info["initial_cost"]) + float(translation_result.info["initial_cost"]),
        "cost": float((0.5 * ((final_res ** 2).sum() + (final_prior ** 2).sum())).cpu()),
        "edge_residual_mean": float(edge_res.mean().cpu()),
        "edge_residual_max": float(edge_res.max().cpu()),
        "rotation_residual_mean": float(rotation_result.info["rotation_residual_mean"]),
        "rotation_residual_max": float(rotation_result.info["rotation_residual_max"]),
        "translation_residual_mean": float(translation_result.info["translation_residual_mean"]),
        "translation_residual_max": float(translation_result.info["translation_residual_max"]),
        "scale_prior_residual_mean": float(final_prior.abs().mean().cpu()),
        "scale_prior_residual_max": float(final_prior.abs().max().cpu()),
    }
    return Sim3PGOResult(poses=translation_result.poses, log_scales=translation_result.log_scales, info=info)



def optimize_sim3_pose_graph(
    n_nodes: int,
    ii: torch.Tensor,
    jj: torch.Tensor,
    rel_poses: torch.Tensor,
    rel_scales: torch.Tensor,
    initial_log_scales: torch.Tensor,
    edge_conf: torch.Tensor,
    initial_poses: torch.Tensor | None = None,
    anchor: int = 0,
    n_iters: int = 12,
    damping: float = 1e-3,
    lm_max_attempts: int = DEFAULT_LM_MAX_ATTEMPTS,
    huber_delta: float = 1.0,
    scale_conf: float = 0.01,
    mode: str = "se3_scale",
    backend: str = CUDA_EIGEN_BACKEND,
) -> Sim3PGOResult:
    """Optimize a scaled SE(3) pose graph with the GNT edge convention.

    Input shapes:
        ii, jj: (E,)
        rel_poses: (E, 7), source-normalized relative SE(3) edges
        rel_scales: (E,), diagnostic source/reference scales
        initial_log_scales: (N,)
        edge_conf: (E, 2) = [translation_conf, rotation_conf]
        initial_poses: (N, 7), optional

    Finalized pose translations are interpreted in source-normalized depth
    units, matching the camera loss used in GNT training:

        rel_pose_ij ~= normalize_translation(T_j * inv(T_i), scale_i)
        rel_scale_ij carries a source/reference metric scale diagnostic

        scale_conf is a soft per-node pose-scale prior strength.
    Relative edge scales are not used as hard scale measurements in the PGO
    objective. The default solve is joint SE3+scale PGO. Staged rotation then
    fixed-rotation translation+scale PGO remains available as an ablation.
    backend="cuda_eigen" is the production CUDA
    path: it builds fixed-layout CUDA Jacobian blocks, copies them to CPU
    double precision, and solves the sparse normal system with cached Eigen
    SimplicialLLT. backend="torch" is retained as the dense reference path for
    CPU/unit tests.
    """
    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive")
    if not 0 <= anchor < n_nodes:
        raise ValueError(f"anchor must be in [0, {n_nodes - 1}]")
    if ii.shape != jj.shape:
        raise ValueError("ii and jj must have the same shape")
    if rel_poses.ndim != 2 or rel_poses.shape[-1] != 7:
        raise ValueError("rel_poses must have shape (n_edges, 7)")
    if rel_scales.ndim != 1:
        raise ValueError("rel_scales must have shape (n_edges,)")
    if rel_poses.shape[0] != ii.numel() or rel_scales.shape[0] != ii.numel():
        raise ValueError("edge tensors must agree on n_edges")
    if mode not in PGO_MODES:
        raise ValueError("mode must be 'rotation_only', 'staged', or 'se3_scale'")
    if backend not in {"torch", CUDA_EIGEN_BACKEND}:
        raise ValueError("backend must be 'cuda_eigen' or 'torch'")
    if lm_max_attempts < 1:
        raise ValueError("lm_max_attempts must be positive")
    device = rel_poses.device
    dtype = rel_poses.dtype
    if backend == CUDA_EIGEN_BACKEND and (device.type != "cuda" or dtype != torch.float32):
        raise RuntimeError(f"backend='{backend}' requires CUDA float32 tensors")
    start_time = time.perf_counter()
    log_s0 = initial_log_scales.to(device=device, dtype=dtype)
    edge_conf = edge_conf.to(device=device, dtype=dtype)
    if ii.numel() == 0:
        if initial_poses is None:
            poses = rel_poses.new_zeros((n_nodes, 7))
            poses[:, 6] = 1
        else:
            poses = initial_poses.to(device=device, dtype=dtype)
        result = Sim3PGOResult(
            poses=poses,
            log_scales=log_s0,
            info={
                "success": True,
                "mode": mode,
                "n_edges": 0,
                "backend": backend,
                **(CUDA_EIGEN_SOLVER_INFO if backend == CUDA_EIGEN_BACKEND else TORCH_SOLVER_INFO),
            },
            initial_poses=poses,
        )
        result.info["runtime_sec"] = time.perf_counter() - start_time
        return result

    ii = ii.to(device=device, dtype=torch.long)
    jj = jj.to(device=device, dtype=torch.long)
    if ii.min() < 0 or jj.min() < 0 or ii.max() >= n_nodes or jj.max() >= n_nodes:
        raise ValueError("edge indices must be in [0, n_nodes)")
    rel_edges = RelativeEdges(
        poses=SE3(rel_poses),
        scales=rel_scales.to(device=device, dtype=dtype),
    )
    if initial_poses is None:
        poses, _ = _initial_from_edges(
            n_nodes,
            ii,
            jj,
            rel_edges,
            anchor,
            log_s0,
            edge_conf,
        )
        log_s = log_s0
    else:
        initial_poses = initial_poses.to(device=device, dtype=dtype)
        poses = SE3(initial_poses)
        log_s = log_s0
    prior_log_s = log_s0
    initial_pose_data = poses.data

    if mode == "rotation_only":
        meas_rotations = SO3(rel_edges.poses.data[..., 3:7])
        if backend == CUDA_EIGEN_BACKEND:
            result = _optimize_rotation_pose_graph_cuda(
                poses,
                log_s,
                ii,
                jj,
                meas_rotations,
                edge_conf,
                anchor,
                n_iters,
                damping,
                lm_max_attempts,
                huber_delta,
            )
        else:
            result = _optimize_rotation_pose_graph(
                poses,
                log_s,
                ii,
                jj,
                meas_rotations,
                edge_conf,
                anchor,
                n_iters,
                damping,
                lm_max_attempts,
                huber_delta,
            )
        result.info["runtime_sec"] = time.perf_counter() - start_time
        result.initial_poses = initial_pose_data
        return result

    if mode == "staged":
        result = _optimize_staged_pose_graph(
            poses,
            log_s,
            ii,
            jj,
            rel_edges,
            edge_conf,
            prior_log_s,
            anchor,
            n_iters,
            damping,
            lm_max_attempts,
            huber_delta,
            scale_conf,
            backend,
        )
    elif backend == CUDA_EIGEN_BACKEND:
        result = _optimize_se3_scale_pose_graph_cuda(
            poses,
            log_s,
            ii,
            jj,
            rel_edges,
            edge_conf,
            prior_log_s,
            anchor,
            n_iters,
            damping,
            lm_max_attempts,
            huber_delta,
            scale_conf,
        )
    else:
        result = _optimize_se3_scale_pose_graph(
            poses,
            log_s,
            ii,
            jj,
            rel_edges,
            edge_conf,
            prior_log_s,
            anchor,
            n_iters,
            damping,
            lm_max_attempts,
            huber_delta,
            scale_conf,
        )
    result.info["runtime_sec"] = time.perf_counter() - start_time
    result.initial_poses = initial_pose_data
    return result
