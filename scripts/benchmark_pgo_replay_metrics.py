#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.evaluate_tartanair_pgo import (  # noqa: E402
    _load_gt_poses,
    _quat_to_matrix,
    _rotation_matrix_errors_deg,
    _sim3_align,
    _stats,
    _trajectory_metrics,
    _vector_angle_errors_deg,
)
from geont_runtime.slam.pgo import optimizer as pgo  # noqa: E402
from geont_runtime.slam.pgo.replay import load_pgo_replay_graph  # noqa: E402


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

    aligned_rel_t, _ = _relative_pose_np(aligned_poses, ii, jj)
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
    out["trajectory_aligned_rot_mean_deg"] = float(np.mean(_rotation_matrix_errors_deg(aligned_R, gt_R)))
    return out


def _stage_sum(info: dict, key: str) -> int:
    stages = info.get("stage_info")
    if isinstance(stages, dict):
        return int(sum(int(stage.get(key, 0)) for stage in stages.values()))
    return int(info.get(key, 0))


def _summarize_result(
    *,
    backend: str,
    mode: str,
    run: int,
    measured: bool,
    elapsed: float,
    result: pgo.Sim3PGOResult,
    gt_poses: np.ndarray,
    ii: np.ndarray,
    jj: np.ndarray,
) -> dict[str, float | str | int | bool]:
    poses = result.poses.detach().cpu().numpy()
    traj = _trajectory_metrics(poses, gt_poses)
    edge = _edge_metrics(poses, gt_poses, ii, jj)
    info = result.info
    return {
        "backend": backend,
        "mode": mode,
        "run": int(run),
        "measured": bool(measured),
        "success": bool(info.get("success", False)),
        "wall_time_sec": float(elapsed),
        "runtime_sec_info": float(info.get("runtime_sec", 0.0)),
        "pgo_cost": float(info.get("cost", 0.0)),
        "pgo_initial_cost": float(info.get("initial_cost", 0.0)),
        "pgo_edge_residual_mean": float(info.get("edge_residual_mean", 0.0)),
        "pgo_edge_residual_max": float(info.get("edge_residual_max", 0.0)),
        "pgo_scale_prior_residual_mean": float(info.get("scale_prior_residual_mean", 0.0)),
        "pgo_scale_prior_residual_max": float(info.get("scale_prior_residual_max", 0.0)),
        "accepted_iters": _stage_sum(info, "accepted_iters"),
        "rejected_attempts": _stage_sum(info, "rejected_attempts"),
        "solver_failures": _stage_sum(info, "solver_failures"),
        "linear_solver": str(info.get("linear_solver", "")),
        "linear_solver_impl": str(info.get("linear_solver_impl", "")),
        "normal_equation_assembly": str(info.get("normal_equation_assembly", "")),
        "ate_rmse": float(traj["ate_rmse"]),
        "ate_mean": float(traj["ate_mean"]),
        "ate_max": float(traj["ate_max"]),
        "trajectory_rot_mean_deg": float(traj["rot_mean_deg"]),
        "trajectory_rot_max_deg": float(traj["rot_max_deg"]),
        **edge,
    }


def _run_once(graph: dict, args: argparse.Namespace, *, backend: str, mode: str, device: torch.device) -> pgo.Sim3PGOResult:
    return pgo.optimize_sim3_pose_graph(
        n_nodes=int(graph["n_nodes"]),
        ii=graph["ii"],
        jj=graph["jj"],
        rel_poses=graph["relative_pose"],
        rel_scales=graph["relative_scale"],
        edge_conf=graph["confidence"],
        initial_poses=graph["initial_poses"],
        initial_log_scales=graph["initial_log_scales"],
        anchor=int(graph["anchor"]),
        n_iters=int(args.iters),
        damping=float(args.damping),
        lm_max_attempts=int(args.lm_max_attempts),
        huber_delta=float(args.huber_delta),
        scale_conf=float(args.scale_conf),
        mode=mode,
        backend=backend,
    )


def _numeric_summary(rows: list[dict]) -> dict[str, float | str | int]:
    numeric_keys = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (int, float, np.integer, np.floating)) and key not in {"run", "measured"}
    ]
    out: dict[str, float | str | int] = {
        "backend": str(rows[0]["backend"]),
        "mode": str(rows[0]["mode"]),
        "repeat": len(rows),
    }
    for key in numeric_keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        out[f"{key}_mean"] = float(np.mean(values))
        out[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PGO replay accuracy and runtime.")
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/pgo_replay_benchmark"))
    parser.add_argument("--backends", nargs="+", default=["cuda_eigen", "torch"])
    parser.add_argument("--modes", nargs="+", default=["se3_scale"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--damping", type=float, default=1.0e-3)
    parser.add_argument("--lm-max-attempts", type=int, default=5)
    parser.add_argument("--huber-delta", type=float, default=0.05)
    parser.add_argument("--scale-conf", type=float, default=0.01)
    args = parser.parse_args()

    if not args.graph.exists():
        raise FileNotFoundError(args.graph)
    if not (args.scene / "pose_left.txt").exists():
        raise FileNotFoundError(args.scene / "pose_left.txt")

    with np.load(args.graph, allow_pickle=False) as data:
        timestamps = np.asarray(data["timestamps"]).astype(np.int64)
    gt_poses = _load_gt_poses(args.scene)[timestamps]

    raw_rows = []
    measured_rows = []
    with torch.inference_mode():
        for backend in args.backends:
            device = _resolve_device(args.device, backend)
            graph = load_pgo_replay_graph(args.graph, device=device, dtype=torch.float32)
            ii = graph["ii"].detach().cpu().numpy().astype(np.int64)
            jj = graph["jj"].detach().cpu().numpy().astype(np.int64)
            for mode in args.modes:
                for run in range(args.warmups + args.repeats):
                    measured = run >= args.warmups
                    _sync(device)
                    start = time.perf_counter()
                    result = _run_once(graph, args, backend=backend, mode=mode, device=device)
                    _sync(device)
                    row = _summarize_result(
                        backend=backend,
                        mode=mode,
                        run=run - args.warmups,
                        measured=measured,
                        elapsed=time.perf_counter() - start,
                        result=result,
                        gt_poses=gt_poses,
                        ii=ii,
                        jj=jj,
                    )
                    raw_rows.append(row)
                    if measured:
                        measured_rows.append(row)
                    print(json.dumps(row, sort_keys=True), flush=True)

    summaries = []
    for backend in args.backends:
        for mode in args.modes:
            rows = [row for row in measured_rows if row["backend"] == backend and row["mode"] == mode]
            if rows:
                summaries.append(_numeric_summary(rows))

    args.output.mkdir(parents=True, exist_ok=True)
    raw_csv = args.output / "raw.csv"
    summary_json = args.output / "summary.json"
    summary_csv = args.output / "summary.csv"
    with raw_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        writer.writeheader()
        writer.writerows(raw_rows)
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    summary_json.write_text(
        json.dumps(
            {
                "graph": str(args.graph),
                "scene": str(args.scene),
                "warmups": args.warmups,
                "repeats": args.repeats,
                "iters": args.iters,
                "damping": args.damping,
                "lm_max_attempts": args.lm_max_attempts,
                "huber_delta": args.huber_delta,
                "scale_conf": args.scale_conf,
                "summaries": summaries,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(json.dumps({"raw_csv": str(raw_csv), "summary_csv": str(summary_csv), "summary_json": str(summary_json)}))
    print(json.dumps({"summaries": summaries}, sort_keys=True))


if __name__ == "__main__":
    main()
