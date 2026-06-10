#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from geont_runtime.inference import build_streaming_config
from lietorch import SE3
from geont_runtime.pipeline.default import DefaultAnnotationPipeline
from geont_runtime.slam.pgo.replay import save_pgo_replay_graph
from geont_runtime.streams.frame_dir_stream import FrameDirStream


def _config_path(value) -> Path:
    return Path(to_absolute_path(str(value)))


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    x, y, z, w = [q[..., k] for k in range(4)]
    tx, ty, tz = 2 * x, 2 * y, 2 * z
    xx, yy, zz = tx * x, ty * y, tz * z
    xy, xz, yz = ty * x, tz * x, tz * y
    wx, wy, wz = tx * w, ty * w, tz * w
    row0 = np.stack((1 - (yy + zz), xy - wz, xz + wy), axis=-1)
    row1 = np.stack((xy + wz, 1 - (xx + zz), yz - wx), axis=-1)
    row2 = np.stack((xz - wy, yz + wx, 1 - (xx + yy)), axis=-1)
    return np.stack((row0, row1, row2), axis=-2)


def _resolve_tartanair_scene(root: Path, split_scene: str) -> Path | None:
    split_path = Path(split_scene)
    direct = root / split_path
    if (direct / "pose_left.txt").exists():
        return direct

    parts = split_path.parts
    if len(parts) >= 2 and parts[0] == parts[1]:
        without_duplicate_env = root / Path(*parts[1:])
        if (without_duplicate_env / "pose_left.txt").exists():
            return without_duplicate_env
    return None


def _load_gt_poses(scene_dir: Path) -> np.ndarray:
    poses = np.loadtxt(scene_dir / "pose_left.txt", delimiter=" ")
    poses = poses[:, [1, 2, 0, 4, 5, 3, 6]]
    poses_t = SE3(torch.as_tensor(poses, dtype=torch.float32))
    poses_t = poses_t[[0]].inv() * poses_t
    poses_t = poses_t.inv()
    return poses_t.data.detach().cpu().numpy()


def _sim3_align(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    cov = dst_centered.T @ src_centered / src.shape[0]
    U, singular_values, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    src_var = np.mean(np.sum(src_centered * src_centered, axis=1))
    scale = float(np.trace(np.diag(singular_values) @ S) / max(src_var, 1e-12))
    t = dst_mean - scale * (R @ src_mean)
    aligned = scale * (src @ R.T) + t
    return aligned, scale, R, t


def _rotation_errors_deg(est_q: np.ndarray, gt_q: np.ndarray, align_R: np.ndarray) -> np.ndarray:
    est_R = _quat_to_matrix(est_q)
    gt_R = _quat_to_matrix(gt_q)
    aligned_est_R = align_R[None] @ est_R
    err_R = np.swapaxes(gt_R, -1, -2) @ aligned_est_R
    trace = np.trace(err_R, axis1=-2, axis2=-1)
    cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def _rotation_matrix_errors_deg(est_R: np.ndarray, gt_R: np.ndarray) -> np.ndarray:
    err_R = np.swapaxes(gt_R, -1, -2) @ est_R
    trace = np.trace(err_R, axis1=-2, axis2=-1)
    cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def _relative_pose(poses: np.ndarray, ii: np.ndarray, jj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = poses[:, :3]
    R = _quat_to_matrix(poses[:, 3:7])
    Ri_inv = np.swapaxes(R[ii], -1, -2)
    ti_inv = -np.einsum("eij,ej->ei", Ri_inv, t[ii])
    rel_t = t[jj] + np.einsum("eij,ej->ei", R[jj], ti_inv)
    rel_R = R[jj] @ Ri_inv
    return rel_t, rel_R


def _vector_angle_errors_deg(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred_norm = np.linalg.norm(pred, axis=1)
    gt_norm = np.linalg.norm(gt, axis=1)
    denom = np.maximum(pred_norm * gt_norm, 1e-12)
    cos_theta = np.sum(pred * gt, axis=1) / denom
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_theta))
    invalid = (pred_norm < 1e-8) | (gt_norm < 1e-8)
    angle[invalid] = np.nan
    return angle


def _stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_median": 0.0,
            f"{prefix}_p90": 0.0,
            f"{prefix}_max": 0.0,
        }
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_median": float(np.median(finite)),
        f"{prefix}_p90": float(np.quantile(finite, 0.9)),
        f"{prefix}_max": float(np.max(finite)),
    }


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 2:
        return 0.0
    x = x[mask]
    y = y[mask]
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(x * y) / denom)


def _confidence_tail_means(conf: np.ndarray, error: np.ndarray, prefix: str) -> dict[str, float]:
    mask = np.isfinite(conf) & np.isfinite(error)
    if np.count_nonzero(mask) < 4:
        return {
            f"{prefix}_low_conf_mean": 0.0,
            f"{prefix}_high_conf_mean": 0.0,
        }
    conf = conf[mask]
    error = error[mask]
    low_thr = np.quantile(conf, 0.25)
    high_thr = np.quantile(conf, 0.75)
    return {
        f"{prefix}_low_conf_mean": float(error[conf <= low_thr].mean()),
        f"{prefix}_high_conf_mean": float(error[conf >= high_thr].mean()),
    }


def _trajectory_metrics(est_poses: np.ndarray, gt_poses: np.ndarray) -> dict[str, float]:
    est_t = est_poses[:, :3]
    gt_t = gt_poses[:, :3]
    aligned_t, scale, align_R, _ = _sim3_align(est_t, gt_t)
    trans_err = np.linalg.norm(aligned_t - gt_t, axis=1)
    rot_err = _rotation_errors_deg(est_poses[:, 3:7], gt_poses[:, 3:7], align_R)
    return {
        "keyframes": int(est_poses.shape[0]),
        "sim3_scale": scale,
        "ate_rmse": float(np.sqrt(np.mean(trans_err * trans_err))),
        "ate_mean": float(np.mean(trans_err)),
        "ate_median": float(np.median(trans_err)),
        "ate_max": float(np.max(trans_err)),
        "rot_mean_deg": float(np.mean(rot_err)),
        "rot_median_deg": float(np.median(rot_err)),
        "rot_max_deg": float(np.max(rot_err)),
    }


def _edge_error_rows(
    split_scene: str,
    keyframe_ids: np.ndarray,
    gt_poses: np.ndarray,
    slam_output,
) -> list[dict]:
    finalized_edges = slam_output.finalized_edges or {}
    if not finalized_edges or finalized_edges["ii"].numel() == 0:
        return []

    ii = finalized_edges["ii"].detach().cpu().numpy().astype(np.int64)
    jj = finalized_edges["jj"].detach().cpu().numpy().astype(np.int64)
    rel_pose = finalized_edges["relative_pose"].detach().cpu().numpy()
    edge_scale = finalized_edges["relative_scale"].detach().cpu().numpy()
    confidence = finalized_edges["confidence"].detach().cpu().numpy()
    if confidence.ndim == 1:
        confidence = np.stack((confidence, confidence), axis=1)
    node_scales = slam_output.scales.detach().cpu().numpy() if slam_output.scales is not None else np.ones(len(keyframe_ids))
    optimized_poses = slam_output.trajectory.data.detach().cpu().numpy()

    gt_rel_t, gt_rel_R = _relative_pose(gt_poses, ii, jj)
    optimized_rel_t, optimized_rel_R = _relative_pose(optimized_poses, ii, jj)
    pred_rel_t = rel_pose[:, :3]
    pred_rel_R = _quat_to_matrix(rel_pose[:, 3:7])
    pred_metric_edge_scale = pred_rel_t * edge_scale[:, None]
    pred_metric_node_scale = pred_rel_t * node_scales[ii, None]

    gt_t_norm = np.linalg.norm(gt_rel_t, axis=1)
    pred_edge_norm = np.linalg.norm(pred_metric_edge_scale, axis=1)
    pred_node_norm = np.linalg.norm(pred_metric_node_scale, axis=1)
    optimized_t_norm = np.linalg.norm(optimized_rel_t, axis=1)
    trans_dir_err = _vector_angle_errors_deg(pred_rel_t, gt_rel_t)
    optimized_trans_dir_err = _vector_angle_errors_deg(optimized_rel_t, gt_rel_t)
    trans_edge_scale_err = np.linalg.norm(pred_metric_edge_scale - gt_rel_t, axis=1)
    trans_node_scale_err = np.linalg.norm(pred_metric_node_scale - gt_rel_t, axis=1)
    optimized_trans_err = np.linalg.norm(optimized_rel_t - gt_rel_t, axis=1)
    trans_node_improvement = trans_node_scale_err - optimized_trans_err
    trans_dir_improvement = trans_dir_err - optimized_trans_dir_err
    rot_err = _rotation_matrix_errors_deg(pred_rel_R, gt_rel_R)
    optimized_rot_err = _rotation_matrix_errors_deg(optimized_rel_R, gt_rel_R)
    rot_improvement = rot_err - optimized_rot_err

    rows = []
    for edge_id in range(ii.shape[0]):
        rows.append(
            {
                "scene": split_scene,
                "edge_id": int(edge_id),
                "ii": int(ii[edge_id]),
                "jj": int(jj[edge_id]),
                "timestamp_i": int(keyframe_ids[ii[edge_id]]),
                "timestamp_j": int(keyframe_ids[jj[edge_id]]),
                "span": int(abs(jj[edge_id] - ii[edge_id])),
                "trans_conf": float(confidence[edge_id, 0]),
                "rot_conf": float(confidence[edge_id, 1]),
                "edge_relative_scale": float(edge_scale[edge_id]),
                "node_source_scale": float(node_scales[ii[edge_id]]),
                "gt_translation_norm": float(gt_t_norm[edge_id]),
                "pred_edge_scale_translation_norm": float(pred_edge_norm[edge_id]),
                "pred_node_scale_translation_norm": float(pred_node_norm[edge_id]),
                "optimized_translation_norm": float(optimized_t_norm[edge_id]),
                "translation_direction_error_deg": float(trans_dir_err[edge_id]),
                "optimized_translation_direction_error_deg": float(optimized_trans_dir_err[edge_id]),
                "translation_direction_improvement_deg": float(trans_dir_improvement[edge_id]),
                "translation_edge_scale_error": float(trans_edge_scale_err[edge_id]),
                "translation_node_scale_error": float(trans_node_scale_err[edge_id]),
                "optimized_translation_error": float(optimized_trans_err[edge_id]),
                "translation_node_scale_error_improvement": float(trans_node_improvement[edge_id]),
                "translation_edge_scale_ratio": float(pred_edge_norm[edge_id] / max(gt_t_norm[edge_id], 1e-12)),
                "translation_node_scale_ratio": float(pred_node_norm[edge_id] / max(gt_t_norm[edge_id], 1e-12)),
                "optimized_translation_ratio": float(optimized_t_norm[edge_id] / max(gt_t_norm[edge_id], 1e-12)),
                "rotation_error_deg": float(rot_err[edge_id]),
                "optimized_rotation_error_deg": float(optimized_rot_err[edge_id]),
                "rotation_error_improvement_deg": float(rot_improvement[edge_id]),
                "pred_tx_normed": float(pred_rel_t[edge_id, 0]),
                "pred_ty_normed": float(pred_rel_t[edge_id, 1]),
                "pred_tz_normed": float(pred_rel_t[edge_id, 2]),
                "gt_tx_metric": float(gt_rel_t[edge_id, 0]),
                "gt_ty_metric": float(gt_rel_t[edge_id, 1]),
                "gt_tz_metric": float(gt_rel_t[edge_id, 2]),
                "optimized_tx_metric": float(optimized_rel_t[edge_id, 0]),
                "optimized_ty_metric": float(optimized_rel_t[edge_id, 1]),
                "optimized_tz_metric": float(optimized_rel_t[edge_id, 2]),
            }
        )
    return rows


def _edge_summary(edge_rows: list[dict], prefix: str = "edge") -> dict[str, float]:
    if not edge_rows:
        return {f"{prefix}_count": 0}
    trans_conf = np.asarray([row["trans_conf"] for row in edge_rows], dtype=np.float64)
    rot_conf = np.asarray([row["rot_conf"] for row in edge_rows], dtype=np.float64)
    trans_dir = np.asarray([row["translation_direction_error_deg"] for row in edge_rows], dtype=np.float64)
    opt_trans_dir = np.asarray([row["optimized_translation_direction_error_deg"] for row in edge_rows], dtype=np.float64)
    trans_dir_improvement = np.asarray([row["translation_direction_improvement_deg"] for row in edge_rows], dtype=np.float64)
    trans_edge = np.asarray([row["translation_edge_scale_error"] for row in edge_rows], dtype=np.float64)
    trans_node = np.asarray([row["translation_node_scale_error"] for row in edge_rows], dtype=np.float64)
    opt_trans = np.asarray([row["optimized_translation_error"] for row in edge_rows], dtype=np.float64)
    trans_improvement = np.asarray([row["translation_node_scale_error_improvement"] for row in edge_rows], dtype=np.float64)
    ratio_edge = np.asarray([row["translation_edge_scale_ratio"] for row in edge_rows], dtype=np.float64)
    ratio_node = np.asarray([row["translation_node_scale_ratio"] for row in edge_rows], dtype=np.float64)
    ratio_opt = np.asarray([row["optimized_translation_ratio"] for row in edge_rows], dtype=np.float64)
    rot = np.asarray([row["rotation_error_deg"] for row in edge_rows], dtype=np.float64)
    opt_rot = np.asarray([row["optimized_rotation_error_deg"] for row in edge_rows], dtype=np.float64)
    rot_improvement = np.asarray([row["rotation_error_improvement_deg"] for row in edge_rows], dtype=np.float64)

    out = {f"{prefix}_count": len(edge_rows)}
    out.update(_stats(trans_conf, f"{prefix}_trans_conf"))
    out.update(_stats(rot_conf, f"{prefix}_rot_conf"))
    out.update(_stats(trans_dir, f"{prefix}_translation_direction_error_deg"))
    out.update(_stats(opt_trans_dir, f"{prefix}_optimized_translation_direction_error_deg"))
    out.update(_stats(trans_dir_improvement, f"{prefix}_translation_direction_improvement_deg"))
    out.update(_stats(trans_edge, f"{prefix}_translation_edge_scale_error"))
    out.update(_stats(trans_node, f"{prefix}_translation_node_scale_error"))
    out.update(_stats(opt_trans, f"{prefix}_optimized_translation_error"))
    out.update(_stats(trans_improvement, f"{prefix}_translation_node_scale_error_improvement"))
    out.update(_stats(ratio_edge, f"{prefix}_translation_edge_scale_ratio"))
    out.update(_stats(ratio_node, f"{prefix}_translation_node_scale_ratio"))
    out.update(_stats(ratio_opt, f"{prefix}_optimized_translation_ratio"))
    out.update(_stats(rot, f"{prefix}_rotation_error_deg"))
    out.update(_stats(opt_rot, f"{prefix}_optimized_rotation_error_deg"))
    out.update(_stats(rot_improvement, f"{prefix}_rotation_error_improvement_deg"))
    out[f"{prefix}_optimized_translation_better_fraction"] = float(np.mean(opt_trans < trans_node))
    out[f"{prefix}_optimized_translation_direction_better_fraction"] = float(np.mean(opt_trans_dir < trans_dir))
    out[f"{prefix}_optimized_rotation_better_fraction"] = float(np.mean(opt_rot < rot))
    out[f"{prefix}_trans_conf_vs_direction_error_pearson"] = _pearson(trans_conf, trans_dir)
    out[f"{prefix}_trans_conf_vs_optimized_direction_error_pearson"] = _pearson(trans_conf, opt_trans_dir)
    out[f"{prefix}_trans_conf_vs_edge_scale_error_pearson"] = _pearson(trans_conf, trans_edge)
    out[f"{prefix}_trans_conf_vs_node_scale_error_pearson"] = _pearson(trans_conf, trans_node)
    out[f"{prefix}_trans_conf_vs_optimized_translation_error_pearson"] = _pearson(trans_conf, opt_trans)
    out[f"{prefix}_rot_conf_vs_rotation_error_pearson"] = _pearson(rot_conf, rot)
    out.update(_confidence_tail_means(trans_conf, trans_dir, f"{prefix}_translation_direction_error_deg"))
    out.update(_confidence_tail_means(trans_conf, opt_trans_dir, f"{prefix}_optimized_translation_direction_error_deg"))
    out.update(_confidence_tail_means(trans_conf, trans_edge, f"{prefix}_translation_edge_scale_error"))
    out.update(_confidence_tail_means(trans_conf, trans_node, f"{prefix}_translation_node_scale_error"))
    out.update(_confidence_tail_means(trans_conf, opt_trans, f"{prefix}_optimized_translation_error"))
    out.update(_confidence_tail_means(rot_conf, rot, f"{prefix}_rotation_error_deg"))
    return out


def _write_edge_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _run_scene(cfg: DictConfig, split_scene: str, scene_dir: Path, scene_index: int) -> dict:
    scene_name = split_scene.replace("/", "__")
    output_dir = _config_path(cfg.output) / scene_name
    stream_cfg = build_streaming_config(
        scene_dir / "image_left",
        _config_path(cfg.ckpt),
        list(cfg.intrinsics),
        output_dir=output_dir,
        slam_config=cfg.slam,
    )
    stream = FrameDirStream(
        scene_dir / "image_left",
        seek_range=range(int(cfg.frame_start), int(cfg.frame_end), int(cfg.frame_skip)),
        name=scene_name,
    )
    pipeline = DefaultAnnotationPipeline(
        init=stream_cfg.pipeline.init,
        slam=stream_cfg.pipeline.slam,
        post=stream_cfg.pipeline.post,
        output=stream_cfg.pipeline.output,
    )
    pipeline.return_payload = True

    slam_output = pipeline.run(stream).payload
    assert slam_output is not None
    est_poses = slam_output.trajectory.data.detach().cpu().numpy()
    selected_ids = int(cfg.frame_start) + slam_output.keyframe_ids.astype(np.int64) * int(cfg.frame_skip)
    gt_poses = _load_gt_poses(scene_dir)[selected_ids]

    metrics = _trajectory_metrics(est_poses, gt_poses)
    edge_rows = _edge_error_rows(split_scene, selected_ids, gt_poses, slam_output)
    edge_csv = output_dir / "edge_relative_pose_errors.csv"
    _write_edge_csv(edge_csv, edge_rows)
    pgo_info = slam_output.pgo_info or {}
    replay_graph = output_dir / "pgo_replay_graph.npz"
    if slam_output.pgo_replay:
        save_pgo_replay_graph(
            replay_graph,
            slam_output.pgo_replay,
            pgo_info=pgo_info,
            timestamps=selected_ids,
        )
    metrics.update(
        {
            "scene_index": scene_index,
            "scene": split_scene,
            "resolved_scene": str(scene_dir),
            "frames": int(len(stream)),
            "first_frame": int(cfg.frame_start),
            "frame_skip": int(cfg.frame_skip),
            "edge_count": int((slam_output.finalized_edges or {}).get("ii", torch.empty(0)).numel()),
            "pgo_success": bool(pgo_info.get("success", False)),
            "pgo_mode": str(pgo_info.get("mode", "")),
            "pgo_backend": str(pgo_info.get("backend", "")),
            "pgo_linear_solver": str(pgo_info.get("linear_solver", "")),
            "pgo_normal_equation_assembly": str(pgo_info.get("normal_equation_assembly", "")),
            "pgo_linear_solver_impl": str(pgo_info.get("linear_solver_impl", "")),
            "pgo_runtime_sec": float(pgo_info.get("runtime_sec", 0.0)),
            "pgo_cost": float(pgo_info.get("cost", 0.0)),
            "pgo_edge_residual_mean": float(pgo_info.get("edge_residual_mean", 0.0)),
            "pgo_scale_prior_residual_mean": float(pgo_info.get("scale_prior_residual_mean", 0.0)),
            "edge_error_csv": str(edge_csv),
            "pgo_replay_graph": str(replay_graph) if slam_output.pgo_replay else "",
        }
    )
    metrics.update(_edge_summary(edge_rows))
    return metrics


def _summarize(rows: list[dict], edge_rows: list[dict]) -> dict[str, float]:
    keys = [
        "ate_rmse",
        "ate_mean",
        "ate_median",
        "ate_max",
        "rot_mean_deg",
        "rot_median_deg",
        "rot_max_deg",
        "keyframes",
        "edge_count",
        "pgo_runtime_sec",
    ]
    summary = {"scenes": len(rows)}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_median"] = float(np.median(values))
    summary.update(_edge_summary(edge_rows, prefix="all_edges"))
    return summary


@hydra.main(version_base=None, config_path="../configs", config_name="tartanair_pgo_eval")
def main(cfg: DictConfig) -> None:
    ckpt = _config_path(cfg.ckpt)
    tartanair_root = _config_path(cfg.tartanair_root)
    split_path = _config_path(cfg.split)
    output = _config_path(cfg.output)
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    if not tartanair_root.exists():
        raise FileNotFoundError(f"TartanAir root not found: {tartanair_root}")
    if int(cfg.frame_skip) <= 0:
        raise ValueError("frame_skip must be positive")
    if int(cfg.frame_end) != -1 and int(cfg.frame_end) <= int(cfg.frame_start):
        raise ValueError("frame_end must be greater than frame_start, or -1")

    split_scenes = split_path.read_text().split()
    selected = split_scenes
    if len(cfg.scenes) > 0:
        requested = {str(scene) for scene in cfg.scenes}
        selected = [scene for scene in split_scenes if scene in requested]
        missing_requested = sorted(requested.difference(selected))
        if missing_requested:
            raise ValueError(f"requested scenes are not in the split: {missing_requested}")
    if int(cfg.max_scenes) > 0:
        selected = selected[: int(cfg.max_scenes)]

    output.mkdir(parents=True, exist_ok=True)
    rows = []
    all_edge_rows = []
    for idx, split_scene in enumerate(selected):
        scene_dir = _resolve_tartanair_scene(tartanair_root, split_scene)
        if scene_dir is None:
            raise FileNotFoundError(f"could not resolve split scene under TartanAir root: {split_scene}")
        row = _run_scene(cfg, split_scene, scene_dir, idx)
        rows.append(row)
        edge_csv = Path(row["edge_error_csv"])
        if edge_csv.exists() and edge_csv.stat().st_size > 0:
            with edge_csv.open() as f:
                all_edge_rows.extend(csv.DictReader(f))
        print(json.dumps(row, sort_keys=True))

    summary = _summarize(rows, all_edge_rows)
    _write_edge_csv(output / "edge_relative_pose_errors.csv", all_edge_rows)
    payload = {"summary": summary, "scenes": rows}
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": summary, "summary_path": str(summary_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
