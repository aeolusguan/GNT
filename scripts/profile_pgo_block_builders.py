#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from lietorch import SO3, SE3

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from geont_runtime.slam.pgo import cuda_eigen as pgo_cuda_eigen  # noqa: E402
from geont_runtime.slam.pgo.common import (  # noqa: E402
    EPS,
    RelativeEdges,
    _edge_sqrt_information,
    _rotation_residuals,
    _rotation_sqrt_information,
    _scale_prior_residuals,
    _scaled_se3_residuals,
    _so3_hat,
    _so3_right_jacobian_inverse,
    _translation_residuals,
    _translation_sqrt_information,
)
from geont_runtime.slam.pgo.replay import load_pgo_replay_graph  # noqa: E402


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_call(fn, *, warmups: int, repeats: int, device: torch.device) -> tuple[float, float]:
    for _ in range(warmups):
        fn()
    _sync(device)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        _sync(device)
        samples.append(time.perf_counter() - start)
    values = np.asarray(samples, dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=1)) if repeats > 1 else 0.0


def _torch_rotation_blocks(
    rotations: SO3,
    ii: torch.Tensor,
    jj: torch.Tensor,
    meas_rotations: SO3,
    sqrt_info: torch.Tensor,
    robust: torch.Tensor,
):
    residual = _rotation_residuals(rotations, ii, jj, meas_rotations)
    row_weight = sqrt_info * robust
    source_rot = SO3(rotations.data[ii]).matrix()[..., :3, :3]
    right_jac_inv = _so3_right_jacobian_inverse(residual)
    edge_block = row_weight[:, :, None] * (right_jac_inv @ source_rot)
    return -edge_block, edge_block, residual * row_weight


def _torch_translation_scale_blocks(
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
    pose_data = poses.data
    n_edges = ii.numel()
    residual = _translation_residuals(poses, log_s, ii, jj, rel_edges)
    row_weight = sqrt_info * robust
    pred = (poses[jj] * poses[ii].inv()).data[..., :3]
    inv_source_scale = torch.exp(log_s[ii]).clamp_min(EPS).reciprocal()
    pred_t = pred * inv_source_scale[:, None]
    rot_j = SO3(pose_data[jj, 3:7]).matrix()[..., :3, :3]
    translation_block = row_weight[:, :, None] * rot_j * inv_source_scale[:, None, None]

    source_block = torch.zeros(n_edges, 3, 4, device=pose_data.device, dtype=pose_data.dtype)
    target_block = torch.zeros(n_edges, 3, 4, device=pose_data.device, dtype=pose_data.dtype)
    source_block[:, :, :3] = -translation_block
    source_block[:, :, 3:4] = (-pred_t * row_weight)[:, :, None]
    target_block[:, :, :3] = translation_block
    prior_gradient = (log_s - prior_log_s) * scale_prior_diag
    return source_block, target_block, residual * row_weight, prior_gradient


def _torch_se3_scale_blocks(
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
    pose_data = poses.data
    n_edges = ii.numel()
    pred = (poses[jj] * poses[ii].inv()).data
    pred_t_metric = pred[..., :3]
    inv_source_scale = torch.exp(log_s[ii]).clamp_min(EPS).reciprocal()
    pred_t = pred_t_metric * inv_source_scale[:, None]
    pred_q = pred[..., 3:7]
    rot_residual = (SO3(rel_edges.poses.data[..., 3:7]).inv() * SO3(pred_q)).log()
    residual = torch.cat((pred_t - rel_edges.poses.data[..., :3], rot_residual), dim=-1)
    row_weight = sqrt_info * robust

    rot_i = SO3(pose_data[ii, 3:7]).matrix()[..., :3, :3]
    rot_j = SO3(pose_data[jj, 3:7]).matrix()[..., :3, :3]
    source_t_in_source = torch.einsum("eji,ej->ei", rot_i, pose_data[ii, :3])
    trans_rot_block = rot_j @ _so3_hat(source_t_in_source)
    translation_block = rot_j * inv_source_scale[:, None, None]
    rotation_block = _so3_right_jacobian_inverse(rot_residual) @ rot_i

    source_block = torch.zeros(n_edges, 6, 7, device=pose_data.device, dtype=pose_data.dtype)
    target_block = torch.zeros(n_edges, 6, 7, device=pose_data.device, dtype=pose_data.dtype)
    source_block[:, :3, :3] = -translation_block
    source_block[:, :3, 3:6] = -trans_rot_block * inv_source_scale[:, None, None]
    source_block[:, 3:6, 3:6] = -rotation_block
    source_block[:, :3, 6] = -pred_t
    target_block[:, :3, :3] = translation_block
    target_block[:, :3, 3:6] = trans_rot_block * inv_source_scale[:, None, None]
    target_block[:, 3:6, 3:6] = rotation_block
    source_block = row_weight[:, :, None] * source_block
    target_block = row_weight[:, :, None] * target_block
    prior_gradient = (log_s - prior_log_s) * scale_prior_diag
    return source_block, target_block, residual * row_weight, prior_gradient


def _max_abs(a, b) -> float:
    if isinstance(a, tuple):
        return max(_max_abs(x, y) for x, y in zip(a, b, strict=True))
    return float((a - b).abs().max().detach().cpu())


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile native CUDA vs Torch/LieTorch PGO block builders.")
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--huber-delta", type=float, default=0.05)
    parser.add_argument("--scale-conf", type=float, default=0.01)
    args = parser.parse_args()

    device = torch.device("cuda")
    graph = load_pgo_replay_graph(args.graph, device=device, dtype=torch.float32)
    poses = SE3(graph["initial_poses"])
    log_s = graph["initial_log_scales"]
    ii = graph["ii"]
    jj = graph["jj"]
    rel_edges = RelativeEdges(poses=SE3(graph["relative_pose"]), scales=graph["relative_scale"])
    prior_log_s = log_s
    scale_sqrt_info = torch.as_tensor(float(args.scale_conf), device=device, dtype=torch.float32).clamp_min(1e-6).sqrt()
    scale_prior_diag = float((scale_sqrt_info * scale_sqrt_info).cpu())

    rotations = SO3(poses.data[..., 3:7])
    meas_rotations = SO3(rel_edges.poses.data[..., 3:7])

    rot_sqrt = _rotation_sqrt_information(graph["confidence"])
    rot_r = _rotation_residuals(rotations, ii, jj, meas_rotations) * rot_sqrt
    rot_robust = torch.where(
        rot_r.norm(dim=-1).clamp_min(EPS) <= args.huber_delta,
        torch.ones_like(rot_r.norm(dim=-1)),
        args.huber_delta / rot_r.norm(dim=-1).clamp_min(EPS),
    ).sqrt()[:, None]

    ts_sqrt = _translation_sqrt_information(graph["confidence"])
    ts_r = _translation_residuals(poses, log_s, ii, jj, rel_edges) * ts_sqrt
    ts_robust = torch.where(
        ts_r.norm(dim=-1).clamp_min(EPS) <= args.huber_delta,
        torch.ones_like(ts_r.norm(dim=-1)),
        args.huber_delta / ts_r.norm(dim=-1).clamp_min(EPS),
    ).sqrt()[:, None]

    se3_sqrt = _edge_sqrt_information(graph["confidence"])
    se3_r = _scaled_se3_residuals(poses, log_s, ii, jj, rel_edges) * se3_sqrt
    se3_robust = torch.where(
        se3_r.norm(dim=-1).clamp_min(EPS) <= args.huber_delta,
        torch.ones_like(se3_r.norm(dim=-1)),
        args.huber_delta / se3_r.norm(dim=-1).clamp_min(EPS),
    ).sqrt()[:, None]

    cases = [
        (
            "rotation",
            lambda: pgo_cuda_eigen._rotation_cuda_blocks(rotations, ii, jj, meas_rotations, rot_sqrt, rot_robust),
            lambda: _torch_rotation_blocks(rotations, ii, jj, meas_rotations, rot_sqrt, rot_robust),
        ),
        (
            "translation_scale",
            lambda: pgo_cuda_eigen._translation_scale_cuda_blocks(
                poses, log_s, ii, jj, rel_edges, prior_log_s, ts_sqrt, ts_robust, scale_prior_diag
            ),
            lambda: _torch_translation_scale_blocks(
                poses, log_s, ii, jj, rel_edges, prior_log_s, ts_sqrt, ts_robust, scale_prior_diag
            ),
        ),
        (
            "se3_scale",
            lambda: pgo_cuda_eigen._se3_scale_cuda_blocks(
                poses, log_s, ii, jj, rel_edges, prior_log_s, se3_sqrt, se3_robust, scale_prior_diag
            ),
            lambda: _torch_se3_scale_blocks(
                poses, log_s, ii, jj, rel_edges, prior_log_s, se3_sqrt, se3_robust, scale_prior_diag
            ),
        ),
    ]

    rows = []
    for name, native_fn, torch_fn in cases:
        native_out = native_fn()
        torch_out = torch_fn()
        _sync(device)
        native_mean, native_std = _time_call(native_fn, warmups=args.warmups, repeats=args.repeats, device=device)
        torch_mean, torch_std = _time_call(torch_fn, warmups=args.warmups, repeats=args.repeats, device=device)
        row = {
            "case": name,
            "n_nodes": int(graph["n_nodes"]),
            "n_edges": int(ii.numel()),
            "native_cuda_mean_ms": native_mean * 1000.0,
            "native_cuda_std_ms": native_std * 1000.0,
            "torch_lietorch_mean_ms": torch_mean * 1000.0,
            "torch_lietorch_std_ms": torch_std * 1000.0,
            "native_over_torch": native_mean / torch_mean,
            "max_abs_diff": _max_abs(native_out, torch_out),
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    print(json.dumps({"rows": rows}, sort_keys=True))


if __name__ == "__main__":
    main()
