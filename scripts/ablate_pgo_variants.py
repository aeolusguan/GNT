#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import time

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lietorch import SE3
from scripts.evaluate_tartanair_pgo import (
    _load_gt_poses,
    _quat_to_matrix,
    _rotation_matrix_errors_deg,
    _sim3_align,
    _stats,
    _trajectory_metrics,
    _vector_angle_errors_deg,
)
from geont_runtime.slam.pgo import optimizer as pgo
from geont_runtime.slam.pgo.replay import load_pgo_replay_graph


def _config_path(value) -> Path:
    return Path(to_absolute_path(str(value)))


def _resolve_device(name: str, backend: str) -> torch.device:
    if name == "auto":
        if backend == "cuda_eigen":
            return torch.device("cuda")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _relative_pose_np(poses: np.ndarray, ii: np.ndarray, jj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = poses[:, :3]
    R = _quat_to_matrix(poses[:, 3:7])
    Ri_inv = np.swapaxes(R[ii], -1, -2)
    ti_inv = -np.einsum("eij,ej->ei", Ri_inv, t[ii])
    rel_t = t[jj] + np.einsum("eij,ej->ei", R[jj], ti_inv)
    rel_R = R[jj] @ Ri_inv
    return rel_t, rel_R


def _edge_metrics(est_poses: np.ndarray, gt_poses: np.ndarray, ii: np.ndarray, jj: np.ndarray) -> dict[str, float]:
    gt_rel_t, gt_rel_R = _relative_pose_np(gt_poses, ii, jj)
    raw_rel_t, raw_rel_R = _relative_pose_np(est_poses, ii, jj)

    aligned_t, sim3_scale, align_R, _ = _sim3_align(est_poses[:, :3], gt_poses[:, :3])
    aligned_poses = est_poses.copy()
    aligned_poses[:, :3] = aligned_t
    aligned_R = align_R[None] @ _quat_to_matrix(est_poses[:, 3:7])
    gt_R = _quat_to_matrix(gt_poses[:, 3:7])

    aligned_rel_t, aligned_rel_R = _relative_pose_np(aligned_poses, ii, jj)
    aligned_rel_R = aligned_R[jj] @ np.swapaxes(aligned_R[ii], -1, -2)

    out = {"edge_alignment_scale": float(sim3_scale)}
    for prefix, rel_t, rel_R in (
        ("edge_raw", raw_rel_t, raw_rel_R),
        ("edge_sim3_aligned", aligned_rel_t, aligned_rel_R),
    ):
        trans_err = np.linalg.norm(rel_t - gt_rel_t, axis=1)
        trans_dir = _vector_angle_errors_deg(rel_t, gt_rel_t)
        trans_ratio = np.linalg.norm(rel_t, axis=1) / np.maximum(np.linalg.norm(gt_rel_t, axis=1), 1e-12)
        rot_err = _rotation_matrix_errors_deg(rel_R, gt_rel_R)
        out.update(_stats(trans_err, f"{prefix}_translation_error"))
        out.update(_stats(trans_dir, f"{prefix}_translation_direction_error_deg"))
        out.update(_stats(trans_ratio, f"{prefix}_translation_norm_ratio"))
        out.update(_stats(rot_err, f"{prefix}_rotation_error_deg"))
    out["trajectory_aligned_rot_mean_deg"] = float(
        np.mean(_rotation_matrix_errors_deg(aligned_R, gt_R))
    )
    return out


def _run_optimizer(
    graph: dict,
    *,
    mode: str,
    backend: str,
    n_iters: int,
    damping: float,
    lm_max_attempts: int,
    huber_delta: float,
    scale_conf: float,
    device: torch.device,
) -> pgo.Sim3PGOResult:
    _sync(device)
    start = time.perf_counter()
    result = pgo.optimize_sim3_pose_graph(
        n_nodes=int(graph["n_nodes"]),
        ii=graph["ii"],
        jj=graph["jj"],
        rel_poses=graph["relative_pose"],
        rel_scales=graph["relative_scale"],
        edge_conf=graph["confidence"],
        initial_poses=graph["initial_poses"],
        initial_log_scales=graph["initial_log_scales"],
        anchor=int(graph["anchor"]),
        n_iters=n_iters,
        damping=damping,
        lm_max_attempts=lm_max_attempts,
        huber_delta=huber_delta,
        scale_conf=scale_conf,
        mode=mode,
        backend=backend,
    )
    _sync(device)
    result.info["wall_time_sec"] = time.perf_counter() - start
    return result


def _run_joint_refinement(
    staged: pgo.Sim3PGOResult,
    graph: dict,
    rel_edges: pgo.RelativeEdges,
    *,
    backend: str,
    n_iters: int,
    damping: float,
    lm_max_attempts: int,
    huber_delta: float,
    scale_conf: float,
    device: torch.device,
) -> pgo.Sim3PGOResult:
    _sync(device)
    start = time.perf_counter()
    if backend == "cuda_eigen":
        refine = pgo._optimize_se3_scale_pose_graph_cuda(
            SE3(staged.poses),
            staged.log_scales,
            graph["ii"],
            graph["jj"],
            rel_edges,
            graph["confidence"],
            graph["initial_log_scales"],
            int(graph["anchor"]),
            n_iters,
            damping,
            lm_max_attempts,
            huber_delta,
            scale_conf,
        )
    else:
        refine = pgo._optimize_se3_scale_pose_graph(
            SE3(staged.poses),
            staged.log_scales,
            graph["ii"],
            graph["jj"],
            rel_edges,
            graph["confidence"],
            graph["initial_log_scales"],
            int(graph["anchor"]),
            n_iters,
            damping,
            lm_max_attempts,
            huber_delta,
            scale_conf,
        )
    _sync(device)
    refine.info["mode"] = "staged_then_se3_scale"
    refine.info["wall_time_sec"] = float(staged.info["wall_time_sec"]) + time.perf_counter() - start
    refine.info["stage_order"] = ["staged", "se3_scale"]
    refine.info["stage_info"] = {"staged": staged.info, "se3_scale": refine.info.copy()}
    return refine


def _summarize_result(
    *,
    variant: str,
    scale_conf: float,
    huber_delta: float,
    result: pgo.Sim3PGOResult,
    gt_poses: np.ndarray,
    ii: np.ndarray,
    jj: np.ndarray,
) -> dict[str, float | str | bool | int]:
    poses = result.poses.detach().cpu().numpy()
    traj = _trajectory_metrics(poses, gt_poses)
    edge = _edge_metrics(poses, gt_poses, ii, jj)
    info = result.info
    return {
        "variant": variant,
        "scale_conf": float(scale_conf),
        "huber_delta": float(huber_delta),
        "success": bool(info.get("success", False)),
        "mode": str(info.get("mode", "")),
        "pgo_cost": float(info.get("cost", 0.0)),
        "pgo_initial_cost": float(info.get("initial_cost", 0.0)),
        "pgo_edge_residual_mean": float(info.get("edge_residual_mean", 0.0)),
        "pgo_edge_residual_max": float(info.get("edge_residual_max", 0.0)),
        "pgo_scale_prior_residual_mean": float(info.get("scale_prior_residual_mean", 0.0)),
        "pgo_scale_prior_residual_max": float(info.get("scale_prior_residual_max", 0.0)),
        "accepted_iters": int(info.get("accepted_iters", 0)),
        "rejected_attempts": int(info.get("rejected_attempts", 0)),
        "solver_failures": int(info.get("solver_failures", 0)),
        "wall_time_sec": float(info.get("wall_time_sec", info.get("runtime_sec", 0.0))),
        "ate_rmse": float(traj["ate_rmse"]),
        "ate_mean": float(traj["ate_mean"]),
        "ate_max": float(traj["ate_max"]),
        "trajectory_rot_mean_deg": float(traj["rot_mean_deg"]),
        "trajectory_rot_max_deg": float(traj["rot_max_deg"]),
        **edge,
    }


@hydra.main(version_base=None, config_path="../configs", config_name="pgo_variant_ablation")
def main(cfg: DictConfig) -> None:
    graph_path = _config_path(cfg.graph)
    scene_dir = _config_path(cfg.scene)
    output_dir = _config_path(cfg.output)
    backend = str(cfg.backend)
    if backend not in {"cuda_eigen", "torch"}:
        raise ValueError("backend must be 'cuda_eigen' or 'torch'")
    scale_conf_values = [float(value) for value in cfg.scale_conf]
    huber_delta_values = [float(value) for value in cfg.huber_delta]
    if not scale_conf_values:
        raise ValueError("scale_conf must contain at least one value")
    if not huber_delta_values:
        raise ValueError("huber_delta must contain at least one value")

    if not graph_path.exists():
        raise FileNotFoundError(graph_path)
    if not (scene_dir / "pose_left.txt").exists():
        raise FileNotFoundError(scene_dir / "pose_left.txt")
    device = _resolve_device(str(cfg.device), backend)
    dtype = torch.float32
    graph = load_pgo_replay_graph(graph_path, device=device, dtype=dtype)
    rel_edges = pgo.RelativeEdges(poses=SE3(graph["relative_pose"]), scales=graph["relative_scale"])
    with np.load(graph_path, allow_pickle=False) as data:
        timestamps = np.asarray(data["timestamps"]).astype(np.int64)
    gt_poses = _load_gt_poses(scene_dir)[timestamps]
    ii = graph["ii"].detach().cpu().numpy().astype(np.int64)
    jj = graph["jj"].detach().cpu().numpy().astype(np.int64)

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.inference_mode():
        for scale_conf in scale_conf_values:
            for huber_delta in huber_delta_values:
                staged = _run_optimizer(
                    graph,
                    mode="staged",
                    backend=backend,
                    n_iters=int(cfg.iters),
                    damping=float(cfg.damping),
                    lm_max_attempts=int(cfg.lm_max_attempts),
                    huber_delta=huber_delta,
                    scale_conf=scale_conf,
                    device=device,
                )
                joint = _run_optimizer(
                    graph,
                    mode="se3_scale",
                    backend=backend,
                    n_iters=int(cfg.iters),
                    damping=float(cfg.damping),
                    lm_max_attempts=int(cfg.lm_max_attempts),
                    huber_delta=huber_delta,
                    scale_conf=scale_conf,
                    device=device,
                )
                staged_then_joint = _run_joint_refinement(
                    staged,
                    graph,
                    rel_edges,
                    backend=backend,
                    n_iters=int(cfg.iters),
                    damping=float(cfg.damping),
                    lm_max_attempts=int(cfg.lm_max_attempts),
                    huber_delta=huber_delta,
                    scale_conf=scale_conf,
                    device=device,
                )
                for variant, result in (
                    ("A_staged", staged),
                    ("B_se3_scale", joint),
                    ("C_staged_then_se3_scale", staged_then_joint),
                ):
                    row = _summarize_result(
                        variant=variant,
                        scale_conf=scale_conf,
                        huber_delta=huber_delta,
                        result=result,
                        gt_poses=gt_poses,
                        ii=ii,
                        jj=jj,
                    )
                    rows.append(row)
                    print(json.dumps(row, sort_keys=True), flush=True)

    csv_path = output_dir / "ablation.csv"
    json_path = output_dir / "ablation.json"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(
        json.dumps(
            {
                "graph": str(graph_path),
                "scene": str(scene_dir),
                "backend": backend,
                "iters": int(cfg.iters),
                "damping": float(cfg.damping),
                "lm_max_attempts": int(cfg.lm_max_attempts),
                "scale_conf": scale_conf_values,
                "huber_delta": huber_delta_values,
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
