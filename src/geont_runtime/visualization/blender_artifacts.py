from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import os
import subprocess
import sys
import time

import cv2
import numpy as np

from .streaming_artifacts import (
    VideoWriter,
    _render_frame_indices,
    active_keyframe_indices,
    backproject_depth,
    camera_centers_from_tcw,
    camera_points_to_world,
    depth_to_rgb,
    frame_file_for_timestamp,
    invert_pose,
    load_frame_files,
    load_streaming_artifacts,
    metric_keyframe_depth,
    quat_to_matrix,
    read_rgb_frame,
    scale_intrinsics_to_image,
)


@dataclass(frozen=True)
class BlenderExportOptions:
    artifact_root: Path
    frame_dir: Path
    cache_path: Path
    asset_dir: Path
    frame_start: int
    frame_end: int
    frame_stride: int
    source_frame_start: int
    source_frame_skip: int
    point_stride: int
    max_keyframes: int
    max_depth: float
    depth_erode_iterations: int = 0
    depth_discontinuity_threshold: float = 0.0
    voxel_size: float = 0.0
    outlier_filter: str = "none"
    outlier_radius: float = 0.12
    outlier_min_neighbors: int = 4
    outlier_min_retain_ratio: float = 0.25
    sor_neighbors: int = 24
    sor_std_ratio: float = 2.0


OPENCV_WORLD_TO_BLENDER_DISPLAY = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def _matrix_to_quat_xyzw(rotations: np.ndarray) -> np.ndarray:
    rotations = np.asarray(rotations, dtype=np.float64)
    flat = rotations.reshape(-1, 3, 3)
    quats = np.empty((flat.shape[0], 4), dtype=np.float64)
    for idx, matrix in enumerate(flat):
        trace = float(np.trace(matrix))
        if trace > 0.0:
            s = np.sqrt(trace + 1.0) * 2.0
            quats[idx] = [
                (matrix[2, 1] - matrix[1, 2]) / s,
                (matrix[0, 2] - matrix[2, 0]) / s,
                (matrix[1, 0] - matrix[0, 1]) / s,
                0.25 * s,
            ]
        elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
            s = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quats[idx] = [
                0.25 * s,
                (matrix[0, 1] + matrix[1, 0]) / s,
                (matrix[0, 2] + matrix[2, 0]) / s,
                (matrix[2, 1] - matrix[1, 2]) / s,
            ]
        elif matrix[1, 1] > matrix[2, 2]:
            s = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quats[idx] = [
                (matrix[0, 1] + matrix[1, 0]) / s,
                0.25 * s,
                (matrix[1, 2] + matrix[2, 1]) / s,
                (matrix[0, 2] - matrix[2, 0]) / s,
            ]
        else:
            s = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quats[idx] = [
                (matrix[0, 2] + matrix[2, 0]) / s,
                (matrix[1, 2] + matrix[2, 1]) / s,
                0.25 * s,
                (matrix[1, 0] - matrix[0, 1]) / s,
            ]
    quats /= np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-12)
    return quats.reshape(rotations.shape[:-2] + (4,))


def _opencv_world_points_to_blender_display(points: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ OPENCV_WORLD_TO_BLENDER_DISPLAY


def _opencv_twc_to_blender_display(poses_twc: np.ndarray) -> np.ndarray:
    poses_twc = np.asarray(poses_twc, dtype=np.float64)
    rotations_cv = quat_to_matrix(poses_twc[..., 3:7])
    rotations_display = np.einsum("ij,...jk->...ik", OPENCV_WORLD_TO_BLENDER_DISPLAY.T, rotations_cv)
    translations_display = _opencv_world_points_to_blender_display(poses_twc[..., :3])
    quats_display = _matrix_to_quat_xyzw(rotations_display)
    return np.concatenate((translations_display, quats_display), axis=-1)


def _colors_for_keyframe(
    frame_files: list[Path],
    timestamp: int,
    depth: np.ndarray,
    mask: np.ndarray,
    pixels: np.ndarray,
    *,
    source_frame_start: int,
    source_frame_skip: int,
) -> np.ndarray:
    try:
        frame_path = frame_file_for_timestamp(
            frame_files,
            int(timestamp),
            source_frame_start=source_frame_start,
            source_frame_skip=source_frame_skip,
        )
        rgb = read_rgb_frame(frame_path)
        rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA)
    except (IndexError, ValueError):
        rgb = depth_to_rgb(depth, mask)
    return rgb[pixels[:, 0], pixels[:, 1]].astype(np.float32) / 255.0


def _write_rgb_png(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def _enhance_rgb_hud(image_rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(image_rgb, dtype=np.float32) / 255.0
    image = np.clip((image - 0.5) * 1.10 + 0.5 + 0.055, 0.0, 1.0)
    image = np.clip(image ** 0.94, 0.0, 1.0)
    return np.rint(image * 255.0).astype(np.uint8)


def _erode_depth_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    if iterations <= 0:
        return np.asarray(mask, dtype=bool)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(
        np.asarray(mask, dtype=np.uint8),
        kernel,
        iterations=int(iterations),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return eroded.astype(bool)


def _filter_depth_discontinuities(depth: np.ndarray, mask: np.ndarray, threshold: float) -> np.ndarray:
    if threshold <= 0.0:
        return np.asarray(mask, dtype=bool)

    depth = np.asarray(depth, dtype=np.float32)
    filtered = np.asarray(mask, dtype=bool).copy()
    edges = np.zeros(filtered.shape, dtype=bool)
    valid_x = filtered[:, 1:] & filtered[:, :-1]
    jump_x = valid_x & (np.abs(depth[:, 1:] - depth[:, :-1]) > float(threshold))
    edges[:, 1:] |= jump_x
    edges[:, :-1] |= jump_x

    valid_y = filtered[1:, :] & filtered[:-1, :]
    jump_y = valid_y & (np.abs(depth[1:, :] - depth[:-1, :]) > float(threshold))
    edges[1:, :] |= jump_y
    edges[:-1, :] |= jump_y
    return filtered & ~edges


def _point_cloud_filter_enabled(options: BlenderExportOptions) -> bool:
    return options.voxel_size > 0.0 or options.outlier_filter != "none"


def _legacy_open3d_point_cloud(o3d, points: np.ndarray, colors: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return pcd


def _point_cloud_arrays(pcd) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(pcd.points, dtype=np.float32)
    colors = np.asarray(pcd.colors, dtype=np.float32)
    if colors.shape != points.shape:
        colors = np.full(points.shape, 0.55, dtype=np.float32)
    return points, np.clip(colors, 0.0, 1.0).astype(np.float32)


def _open3d_cuda_available(o3d) -> bool:
    cuda = getattr(getattr(o3d, "core", None), "cuda", None)
    return bool(cuda is not None and cuda.is_available())


def _tensor_point_cloud_arrays(pcd) -> tuple[np.ndarray, np.ndarray]:
    points = pcd.point.positions.cpu().numpy().astype(np.float32)
    colors = pcd.point.colors.cpu().numpy().astype(np.float32)
    if colors.shape != points.shape:
        colors = np.full(points.shape, 0.55, dtype=np.float32)
    return points, np.clip(colors, 0.0, 1.0)


def _try_cuda_point_filter(
    o3d,
    points: np.ndarray,
    colors: np.ndarray,
    options: BlenderExportOptions,
) -> tuple[np.ndarray, np.ndarray, str] | None:
    if not _open3d_cuda_available(o3d):
        return None
    try:
        device = o3d.core.Device("CUDA:0")
        pcd = o3d.t.geometry.PointCloud(device)
        pcd.point.positions = o3d.core.Tensor(
            np.asarray(points, dtype=np.float32),
            dtype=o3d.core.Dtype.Float32,
            device=device,
        )
        pcd.point.colors = o3d.core.Tensor(
            np.asarray(colors, dtype=np.float32),
            dtype=o3d.core.Dtype.Float32,
            device=device,
        )
        backend_parts: list[str] = []
        if options.voxel_size > 0.0:
            pcd = pcd.voxel_down_sample(float(options.voxel_size))
            backend_parts.append("cuda_voxel")

        if options.outlier_filter != "none" and int(pcd.point.positions.shape[0]) > 0:
            before_pcd = pcd
            before_count = int(before_pcd.point.positions.shape[0])
            min_retained = int(np.ceil(before_count * float(options.outlier_min_retain_ratio)))
            if options.outlier_filter == "radius":
                candidate_pcd, _ = pcd.remove_radius_outliers(
                    nb_points=int(options.outlier_min_neighbors),
                    search_radius=float(options.outlier_radius),
                )
                backend_name = "cuda_radius"
            elif options.outlier_filter == "sor":
                candidate_pcd, _ = pcd.remove_statistical_outliers(
                    nb_neighbors=int(options.sor_neighbors),
                    std_ratio=float(options.sor_std_ratio),
                )
                backend_name = "cuda_sor"
            else:
                raise ValueError(f"unknown outlier filter {options.outlier_filter}")

            candidate_count = int(candidate_pcd.point.positions.shape[0])
            if options.outlier_min_retain_ratio > 0.0 and candidate_count < max(1, min_retained):
                pcd = before_pcd
                backend_parts.append(f"{backend_name}_skipped_low_retain")
            else:
                pcd = candidate_pcd
                backend_parts.append(backend_name)

        out_points, out_colors = _tensor_point_cloud_arrays(pcd)
        return out_points, out_colors, "+".join(backend_parts) if backend_parts else "cuda"
    except RuntimeError:
        return None
    except AttributeError:
        return None
    except TypeError:
        return None


def _filter_point_cloud_with_open3d(
    points: np.ndarray,
    colors: np.ndarray,
    options: BlenderExportOptions,
) -> tuple[np.ndarray, np.ndarray, str]:
    if not _point_cloud_filter_enabled(options) or points.shape[0] == 0:
        return points.astype(np.float32, copy=False), colors.astype(np.float32, copy=False), "disabled"

    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("Open3D is required when voxel/outlier point filtering is enabled") from exc

    backend_parts: list[str] = []
    filtered_points = points.astype(np.float32, copy=False)
    filtered_colors = np.clip(colors.astype(np.float32, copy=False), 0.0, 1.0)
    cuda_result = _try_cuda_point_filter(o3d, filtered_points, filtered_colors, options)
    if cuda_result is not None:
        return cuda_result

    if options.voxel_size > 0.0:
        pcd = _legacy_open3d_point_cloud(o3d, filtered_points, filtered_colors)
        pcd = pcd.voxel_down_sample(float(options.voxel_size))
        filtered_points, filtered_colors = _point_cloud_arrays(pcd)
        backend_parts.append("cpu_voxel")

    if options.outlier_filter != "none" and filtered_points.shape[0] > 0:
        before_outlier_points = filtered_points
        before_outlier_colors = filtered_colors
        min_retained = int(np.ceil(before_outlier_points.shape[0] * float(options.outlier_min_retain_ratio)))
        pcd = _legacy_open3d_point_cloud(o3d, filtered_points, filtered_colors)
        if options.outlier_filter == "radius":
            pcd, _ = pcd.remove_radius_outlier(
                nb_points=int(options.outlier_min_neighbors),
                radius=float(options.outlier_radius),
            )
            backend_name = "cpu_radius"
        elif options.outlier_filter == "sor":
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=int(options.sor_neighbors),
                std_ratio=float(options.sor_std_ratio),
            )
            backend_name = "cpu_sor"
        else:
            raise ValueError(f"unknown outlier filter {options.outlier_filter}")
        candidate_points, candidate_colors = _point_cloud_arrays(pcd)
        if options.outlier_min_retain_ratio > 0.0 and candidate_points.shape[0] < max(1, min_retained):
            filtered_points = before_outlier_points
            filtered_colors = before_outlier_colors
            backend_parts.append(f"{backend_name}_skipped_low_retain")
        else:
            filtered_points = candidate_points
            filtered_colors = candidate_colors
            backend_parts.append(backend_name)

    return filtered_points, filtered_colors, "+".join(backend_parts) if backend_parts else "open3d"


def export_blender_cache(options: BlenderExportOptions) -> dict[str, object]:
    artifacts = load_streaming_artifacts(options.artifact_root)
    frame_files = load_frame_files(options.frame_dir)
    first_rgb = read_rgb_frame(
        frame_file_for_timestamp(
            frame_files,
            int(artifacts.frame_timestamps[0]),
            source_frame_start=options.source_frame_start,
            source_frame_skip=options.source_frame_skip,
        )
    )
    source_size = first_rgb.shape[:2]
    depth_size = artifacts.depths.shape[-2:]
    depth_intrinsics = scale_intrinsics_to_image(artifacts.intrinsics, source_size, depth_size)
    frame_indices = _render_frame_indices(
        artifacts.frame_trajectory.shape[0],
        options.frame_start,
        options.frame_end,
        options.frame_stride,
    )
    full_active_keyframes = active_keyframe_indices(artifacts.frame_timestamps, artifacts.keyframe_timestamps)
    active_keyframes = full_active_keyframes[frame_indices]
    max_keyframe = int(active_keyframes.max())
    if options.max_keyframes > 0:
        max_keyframe = min(max_keyframe, int(options.max_keyframes) - 1)
    map_start_keyframe = min(int(active_keyframes[0]), max_keyframe)

    options.asset_dir.mkdir(parents=True, exist_ok=True)
    cloud_points: list[np.ndarray] = []
    cloud_colors: list[np.ndarray] = []
    offsets = [0]
    mask_valid_before = 0
    mask_valid_after = 0
    backprojected_points = 0
    filtered_backends: set[str] = set()
    for keyframe_idx in range(map_start_keyframe, max_keyframe + 1):
        depth, mask = metric_keyframe_depth(artifacts, keyframe_idx, max_depth=options.max_depth)
        mask_valid_before += int(mask.sum())
        mask = _erode_depth_mask(mask, options.depth_erode_iterations)
        mask = _filter_depth_discontinuities(depth, mask, options.depth_discontinuity_threshold)
        mask_valid_after += int(mask.sum())
        points_cam, pixels = backproject_depth(depth, mask, depth_intrinsics, stride=options.point_stride)
        backprojected_points += int(points_cam.shape[0])
        if points_cam.shape[0]:
            points_world_cv = camera_points_to_world(points_cam, artifacts.trajectory[keyframe_idx])
            points_world = _opencv_world_points_to_blender_display(points_world_cv).astype(np.float32)
            colors = _colors_for_keyframe(
                frame_files,
                int(artifacts.keyframe_timestamps[keyframe_idx]),
                depth,
                mask,
                pixels,
                source_frame_start=options.source_frame_start,
                source_frame_skip=options.source_frame_skip,
            )
            points_world, colors, backend_name = _filter_point_cloud_with_open3d(points_world, colors, options)
            filtered_backends.add(backend_name)
        else:
            points_world = np.empty((0, 3), dtype=np.float32)
            colors = np.empty((0, 3), dtype=np.float32)
            filtered_backends.add("disabled")
        cloud_points.append(points_world)
        cloud_colors.append(colors)
        offsets.append(offsets[-1] + int(points_world.shape[0]))

    for out_idx, frame_idx in enumerate(frame_indices.tolist()):
        frame_timestamp = int(artifacts.frame_timestamps[frame_idx])
        active_keyframe = min(int(full_active_keyframes[frame_idx]), max_keyframe)
        current_rgb = read_rgb_frame(
            frame_file_for_timestamp(
                frame_files,
                frame_timestamp,
                source_frame_start=options.source_frame_start,
                source_frame_skip=options.source_frame_skip,
            )
        )
        depth, mask = metric_keyframe_depth(artifacts, active_keyframe, max_depth=options.max_depth)
        _write_rgb_png(options.asset_dir / f"rgb_{out_idx:06d}.png", _enhance_rgb_hud(current_rgb))
        _write_rgb_png(options.asset_dir / f"depth_{out_idx:06d}.png", depth_to_rgb(depth, mask))

    packed_points = np.concatenate(cloud_points, axis=0) if cloud_points else np.empty((0, 3), dtype=np.float32)
    packed_colors = np.concatenate(cloud_colors, axis=0) if cloud_colors else np.empty((0, 3), dtype=np.float32)
    active_keyframes = np.minimum(active_keyframes, max_keyframe).astype(np.int64)
    frame_poses_twc = invert_pose(artifacts.frame_trajectory)
    keyframe_poses_twc = invert_pose(artifacts.trajectory)

    options.cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        options.cache_path,
        frame_indices=frame_indices.astype(np.int64),
        active_keyframes=active_keyframes.astype(np.int64),
        render_frame_start=np.asarray(int(frame_indices[0]), dtype=np.int64),
        cloud_keyframe_start=np.asarray(map_start_keyframe, dtype=np.int64),
        cloud_coordinate_frame=np.asarray("blender_display"),
        pose_coordinate_frame=np.asarray("blender_display_from_opencv_camera"),
        frame_timestamps=artifacts.frame_timestamps.astype(np.int64),
        keyframe_timestamps=artifacts.keyframe_timestamps.astype(np.int64),
        frame_poses=_opencv_twc_to_blender_display(frame_poses_twc).astype(np.float32),
        frame_positions=_opencv_world_points_to_blender_display(
            camera_centers_from_tcw(artifacts.frame_trajectory)
        ).astype(np.float32),
        keyframe_poses=_opencv_twc_to_blender_display(keyframe_poses_twc).astype(np.float32),
        intrinsics=artifacts.intrinsics.astype(np.float32),
        source_size=np.asarray(source_size, dtype=np.int32),
        depth_size=np.asarray(depth_size, dtype=np.int32),
        cloud_points=packed_points,
        cloud_colors=packed_colors,
        cloud_offsets=np.asarray(offsets, dtype=np.int64),
    )
    return {
        "cache_path": str(options.cache_path),
        "asset_dir": str(options.asset_dir),
        "frames": int(frame_indices.shape[0]),
        "keyframes": int(max_keyframe + 1),
        "map_start_keyframe": int(map_start_keyframe),
        "raw_points": int(backprojected_points),
        "points": int(packed_points.shape[0]),
        "filter": {
            "mask_valid_before": int(mask_valid_before),
            "mask_valid_after": int(mask_valid_after),
            "depth_erode_iterations": int(options.depth_erode_iterations),
            "depth_discontinuity_threshold": float(options.depth_discontinuity_threshold),
            "voxel_size": float(options.voxel_size),
            "outlier_filter": options.outlier_filter,
            "outlier_radius": float(options.outlier_radius),
            "outlier_min_neighbors": int(options.outlier_min_neighbors),
            "outlier_min_retain_ratio": float(options.outlier_min_retain_ratio),
            "sor_neighbors": int(options.sor_neighbors),
            "sor_std_ratio": float(options.sor_std_ratio),
            "open3d_backends": sorted(filtered_backends),
        },
    }


def run_blender_renderer(
    *,
    blender: Path,
    script: Path,
    cache_path: Path,
    asset_dir: Path,
    render_dir: Path,
    preset: str,
    camera_view: str,
    width: int,
    height: int,
    point_cap: int,
    point_window_keyframes: int,
    point_radius: float,
    point_jitter_scale: float,
    active_cloud_color_mode: str,
    active_depth_colormap: str,
    history_cloud_color_mode: str,
    depth_color_percentiles: tuple[float, float],
    trajectory_radius: float,
    frustum_radius: float,
    frustum_size: float,
    frustum_stride: int,
    trail_length: int,
    render_samples: int,
    gpu_backend: str,
    render_slice_start: int,
    render_slice_end: int,
    teaser_bbox_keyframes: int,
    teaser_shot_frames: int,
    teaser_transition_frames: int,
    teaser_trail_length: int,
    cloud_fade_window: int,
    cloud_min_alpha: float,
    cloud_history_alpha: float,
    cloud_recent_alpha: float,
    cloud_active_alpha: float,
    cloud_history_stride: int,
    cloud_retire_keyframes: int,
    cloud_local_radius: float,
    hide_far_clouds: bool,
    profile_jsonl: Path | None,
) -> dict[str, object]:
    render_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(blender),
        "-b",
        "--gpu-backend",
        gpu_backend,
        "--python",
        str(script),
        "--",
        "--cache",
        str(cache_path),
        "--asset-dir",
        str(asset_dir),
        "--render-dir",
        str(render_dir),
        "--preset",
        preset,
        "--camera-view",
        camera_view,
        "--width",
        str(width),
        "--height",
        str(height),
        "--point-cap",
        str(point_cap),
        "--point-window-keyframes",
        str(point_window_keyframes),
        "--point-radius",
        str(point_radius),
        "--point-jitter-scale",
        str(point_jitter_scale),
        "--active-cloud-color-mode",
        active_cloud_color_mode,
        "--active-depth-colormap",
        active_depth_colormap,
        "--history-cloud-color-mode",
        history_cloud_color_mode,
        "--depth-color-percentiles",
        str(depth_color_percentiles[0]),
        str(depth_color_percentiles[1]),
        "--trajectory-radius",
        str(trajectory_radius),
        "--frustum-radius",
        str(frustum_radius),
        "--frustum-size",
        str(frustum_size),
        "--frustum-stride",
        str(frustum_stride),
        "--trail-length",
        str(trail_length),
        "--render-samples",
        str(render_samples),
        "--render-slice-start",
        str(render_slice_start),
        "--render-slice-end",
        str(render_slice_end),
        "--teaser-bbox-keyframes",
        str(teaser_bbox_keyframes),
        "--teaser-shot-frames",
        str(teaser_shot_frames),
        "--teaser-transition-frames",
        str(teaser_transition_frames),
        "--teaser-trail-length",
        str(teaser_trail_length),
        "--cloud-fade-window",
        str(cloud_fade_window),
        "--cloud-min-alpha",
        str(cloud_min_alpha),
        "--cloud-history-alpha",
        str(cloud_history_alpha),
        "--cloud-recent-alpha",
        str(cloud_recent_alpha),
        "--cloud-active-alpha",
        str(cloud_active_alpha),
        "--cloud-history-stride",
        str(cloud_history_stride),
        "--cloud-retire-keyframes",
        str(cloud_retire_keyframes),
        "--cloud-local-radius",
        str(cloud_local_radius),
    ]
    if hide_far_clouds:
        cmd.append("--hide-far-clouds")
    if profile_jsonl is not None:
        cmd.extend(["--profile-jsonl", str(profile_jsonl)])
    start = time.perf_counter()
    subprocess.run(cmd, check=True)
    return {
        "slice_start": int(render_slice_start),
        "slice_end": int(render_slice_end),
        "wall_s": float(time.perf_counter() - start),
    }


def encode_blender_frames(render_dir: Path, output: Path, fps: float, width: int, height: int, writer: str) -> dict[str, object]:
    frame_paths = sorted(render_dir.glob("frame_*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"no Blender-rendered frames found under {render_dir}")
    output.parent.mkdir(parents=True, exist_ok=True)
    video = VideoWriter(output, fps, (width, height), "cv2" if writer == "auto" else writer)
    try:
        for idx, path in enumerate(frame_paths):
            frame_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame_bgr is None:
                raise ValueError(f"could not read Blender frame {path}")
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if frame_rgb.shape[:2] != (height, width):
                frame_rgb = cv2.resize(frame_rgb, (width, height), interpolation=cv2.INTER_AREA)
            video.append(frame_rgb)
            if idx == 0 or (idx + 1) % 25 == 0:
                print(f"encoded {idx + 1}/{len(frame_paths)} frames", flush=True)
    finally:
        video.close()
    return {"output": str(output), "frames": len(frame_paths)}


def _default_blender_path() -> Path:
    env_path = os.environ.get("BLENDER_BIN")
    if env_path:
        return Path(env_path)
    bundled = Path("/data/disk_7t/tongfan/tools/blender-4.5.4-linux-x64/blender")
    if bundled.exists():
        return bundled
    return Path("blender")


def _default_asset_dir(cache_path: Path) -> Path:
    return cache_path.with_suffix("").with_name(f"{cache_path.stem}_assets")


def _provided_flags(argv: list[str]) -> set[str]:
    flags: set[str] = set()
    for token in argv:
        if token.startswith("--"):
            flags.add(token.split("=", 1)[0])
    return flags


def _apply_recipe_defaults(args: argparse.Namespace, provided_flags: set[str]) -> None:
    recipes: dict[str, dict[str, object]] = {
        "paper-full": {
            "preset": "paper",
            "camera_view": "teaser",
            "width": 960,
            "height": 540,
            "fps": 30.0,
            "frame_stride": 1,
            "point_stride": 8,
            "point_cap": 1_000_000,
            "point_window_keyframes": 80,
            "teaser_bbox_keyframes": 42,
            "cloud_fade_window": 10,
            "cloud_history_stride": 4,
            "cloud_retire_keyframes": 6,
            "active_cloud_color_mode": "hybrid",
            "active_depth_colormap": "turbo",
            "history_cloud_color_mode": "desaturated-rgb",
            "depth_color_percentiles": (2.0, 98.0),
            "frustum_stride": 80,
            "render_samples": 16,
            "render_chunk_size": 500,
            "max_depth": 80.0,
            "writer": "cv2",
        },
        "paper-teaser": {
            "preset": "paper",
            "camera_view": "teaser",
            "width": 1920,
            "height": 1080,
            "fps": 30.0,
            "frame_stride": 1,
            "point_stride": 8,
            "point_cap": 350_000,
            "point_window_keyframes": 48,
            "teaser_bbox_keyframes": 42,
            "cloud_fade_window": 10,
            "cloud_history_stride": 4,
            "cloud_retire_keyframes": 6,
            "active_cloud_color_mode": "hybrid",
            "active_depth_colormap": "turbo",
            "history_cloud_color_mode": "desaturated-rgb",
            "depth_color_percentiles": (2.0, 98.0),
            "frustum_stride": 100,
            "render_samples": 32,
            "max_depth": 80.0,
            "depth_erode_iterations": 1,
            "writer": "cv2",
        },
        "web-preview": {
            "preset": "web",
            "camera_view": "teaser",
            "width": 960,
            "height": 540,
            "fps": 30.0,
            "frame_stride": 1,
            "point_stride": 12,
            "point_cap": 240_000,
            "point_window_keyframes": 48,
            "teaser_bbox_keyframes": 42,
            "cloud_fade_window": 8,
            "cloud_history_stride": 2,
            "cloud_retire_keyframes": 4,
            "active_cloud_color_mode": "hybrid",
            "active_depth_colormap": "turbo",
            "history_cloud_color_mode": "desaturated-rgb",
            "frustum_stride": 100,
            "render_samples": 8,
            "render_chunk_size": 500,
            "writer": "cv2",
        },
        "debug-overview": {
            "preset": "paper",
            "camera_view": "overview",
            "width": 960,
            "height": 540,
            "fps": 30.0,
            "frame_stride": 30,
            "point_stride": 16,
            "point_cap": 180_000,
            "frustum_stride": 80,
            "render_samples": 8,
            "writer": "cv2",
        },
    }
    if args.recipe is None:
        return
    for attr, value in recipes[args.recipe].items():
        flag = "--" + attr.replace("_", "-")
        if flag in provided_flags:
            continue
        setattr(args, attr, value)


def parse_args() -> argparse.Namespace:
    argv = sys.argv[1:]
    provided_flags = _provided_flags(argv)
    parser = argparse.ArgumentParser(description="Render a publication-style Blender video from GeoNT artifacts.")
    parser.add_argument(
        "--recipe",
        choices=("paper-full", "paper-teaser", "web-preview", "debug-overview"),
        help="High-level defaults for common rendering jobs. Explicit CLI flags override recipe values.",
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--frame-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blender", type=Path, default=_default_blender_path(), help=argparse.SUPPRESS)
    parser.add_argument("--cache-path", type=Path)
    parser.add_argument("--asset-dir", type=Path)
    parser.add_argument("--render-dir", type=Path)
    parser.add_argument("--preset", choices=("web", "paper"), default="web", help=argparse.SUPPRESS)
    parser.add_argument("--camera-view", choices=("teaser", "local", "follow", "overview"), default="local", help="Debug override for the camera controller; recipes set this automatically.")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=-1)
    parser.add_argument("--frame-stride", type=int, default=30)
    parser.add_argument("--source-frame-start", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--source-frame-skip", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--point-stride", type=int, default=16)
    parser.add_argument("--point-cap", type=int, default=110_000)
    parser.add_argument("--point-window-keyframes", type=int, default=0, help="Render only the most recent N keyframe point-cloud chunks; 0 uses the view default.")
    parser.add_argument("--point-radius", type=float, default=0.065, help=argparse.SUPPRESS)
    parser.add_argument("--point-jitter-scale", type=float, default=-1.0, help=argparse.SUPPRESS)
    parser.add_argument("--active-cloud-color-mode", choices=("rgb", "depth", "hybrid"), default="hybrid", help=argparse.SUPPRESS)
    parser.add_argument("--active-depth-colormap", choices=("turbo", "viridis", "magma", "cividis"), default="turbo", help=argparse.SUPPRESS)
    parser.add_argument("--history-cloud-color-mode", choices=("rgb", "desaturated-rgb", "gray"), default="desaturated-rgb", help=argparse.SUPPRESS)
    parser.add_argument("--depth-color-percentiles", type=float, nargs=2, default=(2.0, 98.0), metavar=("LOW", "HIGH"), help=argparse.SUPPRESS)
    parser.add_argument("--trajectory-radius", type=float, default=0.035, help=argparse.SUPPRESS)
    parser.add_argument("--frustum-radius", type=float, default=0.022, help=argparse.SUPPRESS)
    parser.add_argument("--frustum-size", type=float, default=0.9, help=argparse.SUPPRESS)
    parser.add_argument("--frustum-stride", type=int, default=120)
    parser.add_argument("--trail-length", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--render-samples", type=int, default=64, help="Blender EEVEE render samples per frame.")
    parser.add_argument("--gpu-backend", choices=("vulkan", "opengl"), default="vulkan", help=argparse.SUPPRESS)
    parser.add_argument("--render-slice-start", type=int, default=0, help="First cached output frame to render; useful with --skip-export.")
    parser.add_argument("--render-slice-end", type=int, default=-1, help="One past last cached output frame to render; -1 renders to cache end.")
    parser.add_argument("--render-chunk-size", type=int, default=0, help="Restart Blender every N output frames; 0 renders in one process.")
    parser.add_argument("--render-workers", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--profile-json", type=Path, help="Write wrapper-level timing summary as JSON.")
    parser.add_argument("--profile-jsonl", type=Path, help="Write Blender per-frame timing records as JSONL.")
    parser.add_argument("--teaser-bbox-keyframes", type=int, default=42, help=argparse.SUPPRESS)
    parser.add_argument("--teaser-shot-frames", type=int, default=300, help=argparse.SUPPRESS)
    parser.add_argument("--teaser-transition-frames", type=int, default=24, help=argparse.SUPPRESS)
    parser.add_argument("--teaser-trail-length", type=int, default=120, help=argparse.SUPPRESS)
    parser.add_argument("--cloud-fade-window", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--cloud-min-alpha", type=float, default=-1.0, help=argparse.SUPPRESS)
    parser.add_argument("--cloud-history-alpha", type=float, default=-1.0, help=argparse.SUPPRESS)
    parser.add_argument("--cloud-recent-alpha", type=float, default=-1.0, help=argparse.SUPPRESS)
    parser.add_argument("--cloud-active-alpha", type=float, default=-1.0, help=argparse.SUPPRESS)
    parser.add_argument("--cloud-history-stride", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--cloud-retire-keyframes", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--cloud-local-radius", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--hide-far-clouds", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--max-depth", type=float, default=80.0, help="Hide predicted depth values at or beyond this distance; <=0 disables.")
    parser.add_argument("--depth-erode-iterations", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--depth-discontinuity-threshold", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--voxel-size", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--outlier-filter", choices=("none", "radius", "sor"), default="none", help=argparse.SUPPRESS)
    parser.add_argument("--outlier-radius", type=float, default=0.12, help=argparse.SUPPRESS)
    parser.add_argument("--outlier-min-neighbors", type=int, default=4, help=argparse.SUPPRESS)
    parser.add_argument("--outlier-min-retain-ratio", type=float, default=0.25, help=argparse.SUPPRESS)
    parser.add_argument("--sor-neighbors", type=int, default=24, help=argparse.SUPPRESS)
    parser.add_argument("--sor-std-ratio", type=float, default=2.0, help=argparse.SUPPRESS)
    parser.add_argument("--max-keyframes", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--writer", choices=("auto", "imageio", "cv2"), default="cv2", help=argparse.SUPPRESS)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-render", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-encode", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    _apply_recipe_defaults(args, provided_flags)
    return args


def main() -> None:
    args = parse_args()
    if args.render_workers != 1:
        raise ValueError("--render-workers is reserved for a later parallel renderer; use --render-workers 1")
    root = Path(__file__).resolve().parents[3]
    blender_script = root / "scripts" / "blender_render_streaming_scene.py"
    cache_path = args.cache_path or args.output.with_suffix(".blender_cache.npz")
    asset_dir = args.asset_dir or _default_asset_dir(cache_path)
    render_dir = args.render_dir or args.output.with_suffix("").with_name(f"{args.output.stem}_blender_frames")
    profile: dict[str, object] = {
        "output": str(args.output),
        "cache_path": str(cache_path),
        "asset_dir": str(asset_dir),
        "render_dir": str(render_dir),
        "gpu_backend": args.gpu_backend,
        "render_chunk_size": int(args.render_chunk_size),
        "render_workers": int(args.render_workers),
    }

    if not args.skip_export:
        export_start = time.perf_counter()
        summary = export_blender_cache(
            BlenderExportOptions(
                artifact_root=args.artifact_root,
                frame_dir=args.frame_dir,
                cache_path=cache_path,
                asset_dir=asset_dir,
                frame_start=args.frame_start,
                frame_end=args.frame_end,
                frame_stride=args.frame_stride,
                source_frame_start=args.source_frame_start,
                source_frame_skip=args.source_frame_skip,
                point_stride=args.point_stride,
                max_keyframes=args.max_keyframes,
                max_depth=args.max_depth,
                depth_erode_iterations=args.depth_erode_iterations,
                depth_discontinuity_threshold=args.depth_discontinuity_threshold,
                voxel_size=args.voxel_size,
                outlier_filter=args.outlier_filter,
                outlier_radius=args.outlier_radius,
                outlier_min_neighbors=args.outlier_min_neighbors,
                outlier_min_retain_ratio=args.outlier_min_retain_ratio,
                sor_neighbors=args.sor_neighbors,
                sor_std_ratio=args.sor_std_ratio,
            )
        )
        summary["wall_s"] = float(time.perf_counter() - export_start)
        profile["export"] = summary
        print(summary, flush=True)

    if not args.skip_render:
        if args.profile_jsonl is not None:
            args.profile_jsonl.parent.mkdir(parents=True, exist_ok=True)
            args.profile_jsonl.write_text("", encoding="utf8")
        with np.load(cache_path, allow_pickle=False) as cache_data:
            total_render_frames = int(cache_data["frame_indices"].shape[0])
        selected_slice_start = max(0, int(args.render_slice_start))
        selected_slice_end = total_render_frames if int(args.render_slice_end) < 0 else min(total_render_frames, int(args.render_slice_end))
        if selected_slice_start >= selected_slice_end:
            raise ValueError(f"empty render slice [{selected_slice_start}, {selected_slice_end}) for {total_render_frames} cached frames")
        chunk_size = int(args.render_chunk_size)
        selected_frames = selected_slice_end - selected_slice_start
        if chunk_size <= 0 or chunk_size >= selected_frames:
            chunks = [(selected_slice_start, selected_slice_end)]
        else:
            chunks = [
                (start, min(start + chunk_size, selected_slice_end))
                for start in range(selected_slice_start, selected_slice_end, chunk_size)
            ]
        render_chunks = []
        render_start = time.perf_counter()
        for chunk_idx, (chunk_slice_start, chunk_slice_end) in enumerate(chunks):
            print(
                f"render chunk {chunk_idx + 1}/{len(chunks)} frames {chunk_slice_start}:{chunk_slice_end}",
                flush=True,
            )
            render_chunks.append(
                run_blender_renderer(
                    blender=args.blender,
                    script=blender_script,
                    cache_path=cache_path,
                    asset_dir=asset_dir,
                    render_dir=render_dir,
                    preset=args.preset,
                    camera_view=args.camera_view,
                    width=args.width,
                    height=args.height,
                    point_cap=args.point_cap,
                    point_window_keyframes=args.point_window_keyframes,
                    point_radius=args.point_radius,
                    point_jitter_scale=args.point_jitter_scale,
                    active_cloud_color_mode=args.active_cloud_color_mode,
                    active_depth_colormap=args.active_depth_colormap,
                    history_cloud_color_mode=args.history_cloud_color_mode,
                    depth_color_percentiles=tuple(args.depth_color_percentiles),
                    trajectory_radius=args.trajectory_radius,
                    frustum_radius=args.frustum_radius,
                    frustum_size=args.frustum_size,
                    frustum_stride=args.frustum_stride,
                    trail_length=args.trail_length,
                    render_samples=args.render_samples,
                    gpu_backend=args.gpu_backend,
                    render_slice_start=chunk_slice_start,
                    render_slice_end=chunk_slice_end,
                    teaser_bbox_keyframes=args.teaser_bbox_keyframes,
                    teaser_shot_frames=args.teaser_shot_frames,
                    teaser_transition_frames=args.teaser_transition_frames,
                    teaser_trail_length=args.teaser_trail_length,
                    cloud_fade_window=args.cloud_fade_window,
                    cloud_min_alpha=args.cloud_min_alpha,
                    cloud_history_alpha=args.cloud_history_alpha,
                    cloud_recent_alpha=args.cloud_recent_alpha,
                    cloud_active_alpha=args.cloud_active_alpha,
                    cloud_history_stride=args.cloud_history_stride,
                    cloud_retire_keyframes=args.cloud_retire_keyframes,
                    cloud_local_radius=args.cloud_local_radius,
                    hide_far_clouds=args.hide_far_clouds,
                    profile_jsonl=args.profile_jsonl,
                )
            )
        profile["render"] = {
            "frames": int(selected_frames),
            "total_cached_frames": int(total_render_frames),
            "slice_start": int(selected_slice_start),
            "slice_end": int(selected_slice_end),
            "chunks": render_chunks,
            "wall_s": float(time.perf_counter() - render_start),
        }

    if not args.skip_encode:
        encode_start = time.perf_counter()
        summary = encode_blender_frames(render_dir, args.output, args.fps, args.width, args.height, args.writer)
        summary["wall_s"] = float(time.perf_counter() - encode_start)
        profile["encode"] = summary
        print(summary, flush=True)
    if args.profile_json is not None:
        args.profile_json.parent.mkdir(parents=True, exist_ok=True)
        args.profile_json.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf8")
        print({"profile_json": str(args.profile_json)}, flush=True)


if __name__ == "__main__":
    main()
