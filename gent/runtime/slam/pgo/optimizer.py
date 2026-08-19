from __future__ import annotations

import time

import torch
from lietorch import SO3, SE3

from .common import (
    DEFAULT_LM_MAX_ATTEMPTS,
    PGO_BACKENDS,
    PGO_MODES,
    RelativeEdges,
    Sim3PGOResult,
    _edge_sqrt_information,
    _initial_from_edges,
    _relative_scale_residuals,
    _scale_sqrt_information,
    _scaled_se3_residuals,
)
from .torch_backend import (
    _optimize_rotation_pose_graph,
    _optimize_se3_scale_pose_graph,
    _optimize_translation_scale_pose_graph,
)
from .cuda_eigen import (
    _optimize_rotation_pose_graph_cuda,
    _optimize_se3_scale_pose_graph_cuda,
    _optimize_translation_scale_pose_graph_cuda,
)


def _optimize_staged_pose_graph(
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
    backend: str,
) -> Sim3PGOResult:
    """Rotation PGO followed by fixed-rotation translation+relative-scale PGO."""
    rotation_fn = _optimize_rotation_pose_graph_cuda if backend == "cuda_eigen" else _optimize_rotation_pose_graph
    translation_fn = (
        _optimize_translation_scale_pose_graph_cuda
        if backend == "cuda_eigen"
        else _optimize_translation_scale_pose_graph
    )
    rotation_result = rotation_fn(
        poses,
        log_s,
        ii,
        jj,
        SO3(rel_edges.poses.data[..., 3:7]),
        edge_conf,
        anchor,
        n_iters,
        damping,
        lm_max_attempts,
        huber_delta,
    )
    translation_result = translation_fn(
        SE3(rotation_result.poses),
        rotation_result.log_scales,
        ii,
        jj,
        rel_edges,
        edge_conf,
        anchor,
        n_iters,
        damping,
        lm_max_attempts,
        huber_delta,
    )

    final_poses = SE3(translation_result.poses)
    final_pose_res = _scaled_se3_residuals(final_poses, translation_result.log_scales, ii, jj, rel_edges)
    final_pose_res = final_pose_res * _edge_sqrt_information(edge_conf)
    final_scale_res = _relative_scale_residuals(translation_result.log_scales, ii, jj, rel_edges)
    final_scale_res = final_scale_res * _scale_sqrt_information(edge_conf)
    finite = bool((torch.isfinite(final_pose_res).all() & torch.isfinite(final_scale_res).all()).cpu())

    info = {
        "success": finite and bool(rotation_result.info["success"]) and bool(translation_result.info["success"]),
        "mode": "staged",
        "backend": backend,
        "n_edges": int(ii.numel()),
        "n_iters": int(n_iters),
        "accepted_iters": int(rotation_result.info["accepted_iters"]) + int(translation_result.info["accepted_iters"]),
        "cost": float((0.5 * ((final_pose_res ** 2).sum() + (final_scale_res ** 2).sum())).cpu()),
    }
    return Sim3PGOResult(poses=translation_result.poses, log_scales=translation_result.log_scales, info=info)


def optimize_sim3_pose_graph(
    n_nodes: int,
    ii: torch.Tensor,
    jj: torch.Tensor,
    rel_poses: torch.Tensor,
    rel_log_scales: torch.Tensor,
    initial_log_scales: torch.Tensor,
    moge_log_scales: torch.Tensor,
    edge_conf: torch.Tensor,
    initial_poses: torch.Tensor | None = None,
    anchor: int = 0,
    n_iters: int = 12,
    damping: float = 1e-3,
    lm_max_attempts: int = DEFAULT_LM_MAX_ATTEMPTS,
    huber_delta: float = 1.0,
    mode: str = "se3_scale",
    backend: str = "cuda_eigen",
    moge_mode_nis: bool = False,
    moge_mode_count: int = 8,
    moge_mode_nis_cutoff: float = 0.01,
) -> Sim3PGOResult:
    """Optimize GeNT pose/scale graph edges.

    Edges contain source-scale-normalized relative SE(3) and a relative
    log-scale measurement:

        translation(T_j * inv(T_i)) / scale_i ~= rel_pose_ij[:3]
        log_s[j] - log_s[i] ~= rel_log_scale_ij

    The anchor pose and anchor log-scale are fixed.
    moge_mode_nis adds mean-zero low-frequency MoGe factors whose fixed
    covariance comes from the initial joint pose-scale Hessian.
    """
    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive")
    if not 0 <= anchor < n_nodes:
        raise ValueError(f"anchor must be in [0, {n_nodes - 1}]")
    if ii.shape != jj.shape:
        raise ValueError("ii and jj must have the same shape")
    if rel_poses.ndim != 2 or rel_poses.shape[-1] != 7:
        raise ValueError("rel_poses must have shape (n_edges, 7)")
    if rel_log_scales.ndim != 1:
        raise ValueError("rel_log_scales must have shape (n_edges,)")
    if rel_poses.shape[0] != ii.numel() or rel_log_scales.shape[0] != ii.numel():
        raise ValueError("edge tensors must agree on n_edges")
    if moge_log_scales.shape != initial_log_scales.shape:
        raise ValueError("moge_log_scales must have one value per node")
    if edge_conf.ndim != 2 or edge_conf.shape != (ii.numel(), 3):
        raise ValueError("edge_conf must have shape (n_edges, 3)")
    if mode not in PGO_MODES:
        raise ValueError("mode must be 'rotation_only', 'staged', or 'se3_scale'")
    if backend not in PGO_BACKENDS:
        raise ValueError("backend must be 'torch' or 'cuda_eigen'")
    if lm_max_attempts < 1:
        raise ValueError("lm_max_attempts must be positive")
    if moge_mode_nis and mode != "se3_scale":
        raise ValueError("MoGe mode factors require mode='se3_scale'")
    if moge_mode_count < 2:
        raise ValueError("moge_mode_count must be at least 2")
    if moge_mode_nis_cutoff <= 0.0:
        raise ValueError("moge_mode_nis_cutoff must be positive")

    start_time = time.perf_counter()
    device = rel_poses.device
    dtype = rel_poses.dtype
    log_s0 = initial_log_scales.to(device=device, dtype=dtype)
    moge_log_s = moge_log_scales.to(device=device, dtype=dtype)
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
                "backend": backend,
                "n_edges": 0,
                "n_iters": 0,
                "accepted_iters": 0,
                "cost": 0.0,
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
        log_scales=rel_log_scales.to(device=device, dtype=dtype),
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
        poses = SE3(initial_poses.to(device=device, dtype=dtype))
        log_s = log_s0
    initial_pose_data = poses.data

    if mode == "rotation_only":
        fn = _optimize_rotation_pose_graph_cuda if backend == "cuda_eigen" else _optimize_rotation_pose_graph
        result = fn(
            poses,
            log_s,
            ii,
            jj,
            SO3(rel_edges.poses.data[..., 3:7]),
            edge_conf,
            anchor,
            n_iters,
            damping,
            lm_max_attempts,
            huber_delta,
        )
    elif mode == "staged":
        result = _optimize_staged_pose_graph(
            poses,
            log_s,
            ii,
            jj,
            rel_edges,
            edge_conf,
            anchor,
            n_iters,
            damping,
            lm_max_attempts,
            huber_delta,
            backend,
        )
    else:
        fn = _optimize_se3_scale_pose_graph_cuda if backend == "cuda_eigen" else _optimize_se3_scale_pose_graph
        result = fn(
            poses,
            log_s,
            ii,
            jj,
            rel_edges,
            edge_conf,
            anchor,
            n_iters,
            damping,
            lm_max_attempts,
            huber_delta,
            moge_log_s,
            bool(moge_mode_nis),
            int(moge_mode_count),
            float(moge_mode_nis_cutoff),
        )

    result.info["runtime_sec"] = time.perf_counter() - start_time
    result.info["backend"] = backend
    result.initial_poses = initial_pose_data
    return result
