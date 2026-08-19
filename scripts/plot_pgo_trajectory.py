#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_tartanair_pgo import _load_gt_poses, _rotation_errors_deg, _sim3_align


AXIS_NAMES = ("x", "y", "z")
COLORS = {
    "gt": "#1A1A1A",
    "pgo": "#0072B2",
    "grid": "#D9DDE3",
    "text": "#222222",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Sim(3)-aligned GT and PGO trajectories.")
    parser.add_argument("--graph", type=Path, required=True, help="PGO replay graph npz with timestamps.")
    parser.add_argument("--scene", type=Path, required=True, help="TartanAir scene directory with pose_left.txt.")
    parser.add_argument("--result", type=Path, required=True, help="PGO replay result npz with optimized poses.")
    parser.add_argument("--output", type=Path, required=True, help="Output path stem, without extension.")
    parser.add_argument("--axes", choices=("auto", "xy", "xz", "yz"), default="auto")
    parser.add_argument("--title", default="")
    return parser.parse_args()


def _select_axes(gt_t: np.ndarray, mode: str) -> tuple[int, int]:
    if mode != "auto":
        return AXIS_NAMES.index(mode[0]), AXIS_NAMES.index(mode[1])
    return tuple(np.argsort(gt_t.var(axis=0))[::-1][:2].tolist())


def _load_timestamps(graph_path: Path) -> np.ndarray:
    with np.load(graph_path, allow_pickle=False) as data:
        return np.asarray(data["timestamps"], dtype=np.int64)


def _load_optimized_poses(result_path: Path) -> np.ndarray:
    with np.load(result_path, allow_pickle=False) as data:
        poses = np.asarray(data["poses"], dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError(f"expected optimized poses with shape (N, 7), got {poses.shape}")
    return poses


def _trajectory_metrics(est_poses: np.ndarray, gt_poses: np.ndarray) -> tuple[np.ndarray, float, dict[str, float]]:
    aligned_t, scale, align_R, _ = _sim3_align(est_poses[:, :3], gt_poses[:, :3])
    trans_err = np.linalg.norm(aligned_t - gt_poses[:, :3], axis=1)
    rot_err = _rotation_errors_deg(est_poses[:, 3:7], gt_poses[:, 3:7], align_R)
    metrics = {
        "keyframes": int(est_poses.shape[0]),
        "sim3_scale": float(scale),
        "ate_rmse": float(np.sqrt(np.mean(trans_err * trans_err))),
        "ate_mean": float(np.mean(trans_err)),
        "ate_max": float(np.max(trans_err)),
        "rot_mean_deg": float(np.mean(rot_err)),
        "rot_max_deg": float(np.max(rot_err)),
    }
    return aligned_t, scale, metrics


def _set_equal_limits(ax, gt_xy: np.ndarray, est_xy: np.ndarray) -> None:
    pts = np.concatenate((gt_xy, est_xy), axis=0)
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    center = 0.5 * (lo + hi)
    radius = 0.5 * float(np.max(hi - lo))
    radius = max(radius, 1e-6) * 1.06
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)


def _direction_arrow(ax, xy: np.ndarray, color: str, fraction: float) -> None:
    """Draw one temporal-direction arrow on a trajectory polyline."""
    if xy.shape[0] < 2:
        return
    step = max(8, xy.shape[0] // 90)
    start = min(max(int(fraction * xy.shape[0]), 0), xy.shape[0] - step - 1)
    best = start
    best_len = 0.0
    for idx in range(max(0, start - 4 * step), min(xy.shape[0] - step, start + 4 * step)):
        length = float(np.linalg.norm(xy[idx + step] - xy[idx]))
        if length > best_len:
            best = idx
            best_len = length
    if best_len < 1e-9:
        return
    ax.annotate(
        "",
        xy=xy[best + step],
        xytext=xy[best],
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "lw": 1.8,
            "mutation_scale": 16,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=6,
    )


def _label_inset_point(ax, xy: np.ndarray, text: str, color: str, offset: tuple[int, int]) -> None:
    ax.annotate(
        text,
        xy=xy,
        xytext=offset,
        textcoords="offset points",
        color=color,
        fontsize=6.8,
        ha="center",
        va="center",
        arrowprops={"arrowstyle": "-", "color": color, "lw": 0.7, "shrinkA": 1, "shrinkB": 3},
        bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "#FFFFFF", "alpha": 0.88},
        zorder=7,
    )


def _endpoint_inset(ax, gt_xy: np.ndarray, est_xy: np.ndarray) -> None:
    axins = ax.inset_axes([0.59, 0.075, 0.34, 0.27])
    axins.plot(gt_xy[:, 0], gt_xy[:, 1], color=COLORS["gt"], lw=1.35)
    axins.plot(est_xy[:, 0], est_xy[:, 1], color=COLORS["pgo"], lw=1.2)
    axins.scatter(gt_xy[0, 0], gt_xy[0, 1], s=34, marker="o", facecolor="white", edgecolor=COLORS["gt"], lw=1.25, zorder=5)
    axins.scatter(gt_xy[-1, 0], gt_xy[-1, 1], s=39, marker="s", facecolor="white", edgecolor=COLORS["gt"], lw=1.25, zorder=5)
    axins.scatter(est_xy[0, 0], est_xy[0, 1], s=34, marker="o", facecolor="white", edgecolor=COLORS["pgo"], lw=1.25, zorder=5)
    axins.scatter(est_xy[-1, 0], est_xy[-1, 1], s=39, marker="s", facecolor="white", edgecolor=COLORS["pgo"], lw=1.25, zorder=5)

    pts = np.stack((gt_xy[0], gt_xy[-1], est_xy[0], est_xy[-1]), axis=0)
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    center = 0.5 * (lo + hi)
    radius = max(0.5 * float(np.max(hi - lo)) * 1.65, 0.95)
    axins.set_xlim(center[0] - radius, center[0] + radius)
    axins.set_ylim(center[1] - radius, center[1] + radius)
    axins.set_aspect("equal", adjustable="box")
    axins.set_title("Endpoint zoom", fontsize=6.9, pad=2)
    axins.grid(True, color=COLORS["grid"], lw=0.45)
    axins.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in axins.spines.values():
        spine.set_color("#AEB4BD")
        spine.set_linewidth(0.8)

    _label_inset_point(axins, gt_xy[0], "GT start", COLORS["gt"], (-30, 17))
    _label_inset_point(axins, gt_xy[-1], "GT end", COLORS["gt"], (-20, -27))
    _label_inset_point(axins, est_xy[0], "PGO start", COLORS["pgo"], (0, 22))
    _label_inset_point(axins, est_xy[-1], "PGO end", COLORS["pgo"], (0, 21))


def _plot(
    *,
    gt_t: np.ndarray,
    aligned_t: np.ndarray,
    axes: tuple[int, int],
    metrics: dict[str, float],
    output_stem: Path,
    title: str,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11.5,
            "legend.fontsize": 9.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    a, b = axes
    gt_xy = gt_t[:, [a, b]]
    est_xy = aligned_t[:, [a, b]]

    fig, ax = plt.subplots(figsize=(5.2, 5.2), constrained_layout=True)
    ax.plot(gt_xy[:, 0], gt_xy[:, 1], color=COLORS["gt"], lw=2.25, label="GT")
    ax.plot(est_xy[:, 0], est_xy[:, 1], color=COLORS["pgo"], lw=1.8, label="PGO optimized")

    _direction_arrow(ax, gt_xy, COLORS["gt"], 0.68)
    _direction_arrow(ax, est_xy, COLORS["pgo"], 0.73)

    ax.scatter(gt_xy[0, 0], gt_xy[0, 1], s=52, marker="o", facecolor="white", edgecolor=COLORS["gt"], lw=1.7, zorder=8)
    ax.scatter(gt_xy[-1, 0], gt_xy[-1, 1], s=62, marker="s", facecolor="white", edgecolor=COLORS["gt"], lw=1.7, zorder=8)
    ax.scatter(est_xy[0, 0], est_xy[0, 1], s=52, marker="o", facecolor="white", edgecolor=COLORS["pgo"], lw=1.7, zorder=8)
    ax.scatter(est_xy[-1, 0], est_xy[-1, 1], s=64, marker="s", facecolor="white", edgecolor=COLORS["pgo"], lw=1.7, zorder=8)

    _endpoint_inset(ax, gt_xy, est_xy)

    _set_equal_limits(ax, gt_xy, est_xy)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"{AXIS_NAMES[a]} [m]")
    ax.set_ylabel(f"{AXIS_NAMES[b]} [m]")
    if title:
        ax.set_title(title, pad=8)
    ax.grid(True, color=COLORS["grid"], lw=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#FFFFFF")

    text = (
        f"Sim(3)-aligned\n"
        f"ATE RMSE: {metrics['ate_rmse']:.2f} m\n"
        f"Rot. mean: {metrics['rot_mean_deg']:.2f} deg"
    )
    ax.text(
        0.025,
        0.975,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        color=COLORS["text"],
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#FFFFFF", "alpha": 0.88},
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=450, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    timestamps = _load_timestamps(args.graph)
    gt_poses_all = _load_gt_poses(args.scene)
    est_poses = _load_optimized_poses(args.result)
    if est_poses.shape[0] != timestamps.shape[0]:
        raise ValueError(f"pose count {est_poses.shape[0]} does not match timestamp count {timestamps.shape[0]}")
    gt_poses = gt_poses_all[timestamps]
    aligned_t, _, metrics = _trajectory_metrics(est_poses, gt_poses)
    axes = _select_axes(gt_poses[:, :3], args.axes)
    _plot(
        gt_t=gt_poses[:, :3],
        aligned_t=aligned_t,
        axes=axes,
        metrics=metrics,
        output_stem=args.output,
        title=args.title,
    )
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(
        json.dumps(
            {
                "graph": str(args.graph),
                "scene": str(args.scene),
                "result": str(args.result),
                "output_stem": str(args.output),
                "axes": [AXIS_NAMES[axes[0]], AXIS_NAMES[axes[1]]],
                "metrics": metrics,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(json.dumps({"output": str(args.output), "axes": [AXIS_NAMES[axes[0]], AXIS_NAMES[axes[1]]], **metrics}, sort_keys=True))


if __name__ == "__main__":
    main()
