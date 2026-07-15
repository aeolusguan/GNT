from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import cv2
import numpy as np


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
OPENCV_WORLD_UP = np.asarray([0.0, -1.0, 0.0], dtype=np.float64)


@dataclass(frozen=True)
class ArtifactPaths:
    artifact_root: Path
    pose_path: Path
    depth_path: Path


@dataclass
class StreamingArtifacts:
    paths: ArtifactPaths
    trajectory: np.ndarray
    keyframe_timestamps: np.ndarray
    frame_trajectory: np.ndarray
    frame_timestamps: np.ndarray
    intrinsics: np.ndarray
    scales: np.ndarray
    depths: np.ndarray
    masks: np.ndarray


@dataclass(frozen=True)
class RenderOptions:
    artifact_root: Path
    frame_dir: Path
    output: Path | None
    width: int
    height: int
    fps: float
    frame_start: int
    frame_end: int
    frame_stride: int
    source_frame_start: int
    source_frame_skip: int
    point_stride: int
    max_points: int
    frustum_stride: int
    frustum_size: float
    point_size: float
    line_width: float
    trail_length: int
    max_depth: float
    save_frame_dir: Path | None
    writer: str
    view_eye: tuple[float, float, float] | None
    view_center: tuple[float, float, float] | None
    view_up: tuple[float, float, float]


def natural_sort_key(path: Path) -> list[int | str]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def resolve_artifact_paths(artifact_root: Path) -> ArtifactPaths:
    artifact_root = artifact_root.expanduser()
    pose_dir = artifact_root / "pose"
    depth_dir = artifact_root / "depth"
    pose_path = pose_dir / f"{artifact_root.name}.npz"
    depth_path = depth_dir / f"{artifact_root.name}.npz"

    if not pose_path.exists():
        candidates = sorted(pose_dir.glob("*.npz"), key=natural_sort_key)
        if len(candidates) != 1:
            raise FileNotFoundError(f"could not resolve one pose npz under {pose_dir}")
        pose_path = candidates[0]
    if not depth_path.exists():
        candidates = sorted(depth_dir.glob("*.npz"), key=natural_sort_key)
        if len(candidates) != 1:
            raise FileNotFoundError(f"could not resolve one depth npz under {depth_dir}")
        depth_path = candidates[0]

    return ArtifactPaths(artifact_root=artifact_root, pose_path=pose_path, depth_path=depth_path)


def _required_npz_array(data: np.lib.npyio.NpzFile, key: str, path: Path) -> np.ndarray:
    if key not in data:
        raise KeyError(f"{path} is missing required key {key!r}")
    return np.asarray(data[key])


def load_streaming_artifacts(artifact_root: Path) -> StreamingArtifacts:
    paths = resolve_artifact_paths(artifact_root)
    with np.load(paths.pose_path, allow_pickle=True) as pose_npz, np.load(paths.depth_path, allow_pickle=False) as depth_npz:
        trajectory = _required_npz_array(pose_npz, "trajectory", paths.pose_path).astype(np.float64)
        keyframe_timestamps = _required_npz_array(pose_npz, "timestamps", paths.pose_path).astype(np.int64)
        frame_trajectory = _required_npz_array(pose_npz, "frame_trajectory", paths.pose_path).astype(np.float64)
        frame_timestamps = _required_npz_array(pose_npz, "frame_timestamps", paths.pose_path).astype(np.int64)
        intrinsics = _required_npz_array(pose_npz, "intrinsics", paths.pose_path).astype(np.float64)
        if "scales" in pose_npz:
            scales = np.asarray(pose_npz["scales"], dtype=np.float64)
        elif "log_scales" in pose_npz:
            scales = np.exp(np.asarray(pose_npz["log_scales"], dtype=np.float64))
        else:
            raise KeyError(f"{paths.pose_path} is missing required key 'scales' or 'log_scales'")

        depths = _required_npz_array(depth_npz, "depths", paths.depth_path)
        masks = np.asarray(depth_npz["masks"], dtype=bool) if "masks" in depth_npz else np.ones_like(depths, dtype=bool)

    if trajectory.ndim != 2 or trajectory.shape[1] != 7:
        raise ValueError(f"expected trajectory shape (N, 7), got {trajectory.shape}")
    if frame_trajectory.ndim != 2 or frame_trajectory.shape[1] != 7:
        raise ValueError(f"expected frame_trajectory shape (F, 7), got {frame_trajectory.shape}")
    if intrinsics.shape != (4,):
        raise ValueError(f"expected intrinsics shape (4,), got {intrinsics.shape}")
    if depths.ndim != 4 or depths.shape[1] != 1:
        raise ValueError(f"expected depths shape (N, 1, H, W), got {depths.shape}")
    if masks.shape != depths.shape:
        raise ValueError(f"expected masks shape {depths.shape}, got {masks.shape}")
    if trajectory.shape[0] != depths.shape[0] or trajectory.shape[0] != keyframe_timestamps.shape[0]:
        raise ValueError("keyframe pose/depth/timestamp counts do not match")
    if scales.shape[0] != trajectory.shape[0]:
        raise ValueError("scale count does not match keyframe count")
    if frame_trajectory.shape[0] != frame_timestamps.shape[0]:
        raise ValueError("frame pose/timestamp counts do not match")

    return StreamingArtifacts(
        paths=paths,
        trajectory=trajectory,
        keyframe_timestamps=keyframe_timestamps,
        frame_trajectory=frame_trajectory,
        frame_timestamps=frame_timestamps,
        intrinsics=intrinsics,
        scales=scales,
        depths=depths,
        masks=masks,
    )


def load_frame_files(frame_dir: Path) -> list[Path]:
    frame_dir = frame_dir.expanduser()
    files: list[Path] = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(frame_dir.glob(f"*{ext}"))
        files.extend(frame_dir.glob(f"*{ext.upper()}"))
    files = sorted(set(files), key=natural_sort_key)
    if not files:
        raise FileNotFoundError(f"no image files found under {frame_dir}")
    return files


def read_rgb_frame(path: Path) -> np.ndarray:
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"could not read RGB frame {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def frame_file_for_timestamp(
    frame_files: list[Path],
    timestamp: int,
    *,
    source_frame_start: int = 0,
    source_frame_skip: int = 1,
) -> Path:
    raw_index = int(source_frame_start) + int(timestamp) * int(source_frame_skip)
    if raw_index < 0 or raw_index >= len(frame_files):
        raise IndexError(
            f"timestamp {timestamp} maps to frame index {raw_index}, "
            f"but frame directory has {len(frame_files)} images"
        )
    return frame_files[raw_index]


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    q = q / np.maximum(norm, 1e-12)
    x, y, z, w = [q[..., k] for k in range(4)]
    tx, ty, tz = 2.0 * x, 2.0 * y, 2.0 * z
    xx, yy, zz = tx * x, ty * y, tz * z
    xy, xz, yz = ty * x, tz * x, tz * y
    wx, wy, wz = tx * w, ty * w, tz * w
    row0 = np.stack((1.0 - (yy + zz), xy - wz, xz + wy), axis=-1)
    row1 = np.stack((xy + wz, 1.0 - (xx + zz), yz - wx), axis=-1)
    row2 = np.stack((xz - wy, yz + wx, 1.0 - (xx + yy)), axis=-1)
    return np.stack((row0, row1, row2), axis=-2)


def pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    matrix = np.zeros(pose.shape[:-1] + (4, 4), dtype=np.float64)
    matrix[..., :3, :3] = quat_to_matrix(pose[..., 3:7])
    matrix[..., :3, 3] = pose[..., :3]
    matrix[..., 3, 3] = 1.0
    return matrix


def invert_pose(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    q = pose[..., 3:7]
    q_inv = np.concatenate((-q[..., :3], q[..., 3:4]), axis=-1)
    t_inv = -np.einsum("...ij,...j->...i", quat_to_matrix(q_inv), pose[..., :3])
    return np.concatenate((t_inv, q_inv), axis=-1)


def camera_centers_from_tcw(poses_tcw: np.ndarray) -> np.ndarray:
    return invert_pose(poses_tcw)[..., :3]


def transform_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    rotation = quat_to_matrix(pose[3:7])
    return points @ rotation.T + pose[:3]


def camera_points_to_world(points: np.ndarray, pose_tcw: np.ndarray) -> np.ndarray:
    pose_tcw = np.asarray(pose_tcw, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    rotation_cw = quat_to_matrix(pose_tcw[3:7])
    return (points - pose_tcw[:3]) @ rotation_cw


def scale_intrinsics_to_image(
    intrinsics: np.ndarray,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> np.ndarray:
    source_h, source_w = source_size
    target_h, target_w = target_size
    if source_h <= 0 or source_w <= 0:
        raise ValueError(f"invalid source size {source_size}")
    out = np.asarray(intrinsics, dtype=np.float64).copy()
    out[[0, 2]] *= float(target_w) / float(source_w)
    out[[1, 3]] *= float(target_h) / float(source_h)
    return out


def backproject_depth(
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    *,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    if stride <= 0:
        raise ValueError("stride must be positive")
    depth = np.asarray(depth, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if depth.ndim != 2:
        raise ValueError(f"expected depth shape (H, W), got {depth.shape}")
    if mask.shape != depth.shape:
        raise ValueError(f"expected mask shape {depth.shape}, got {mask.shape}")

    h, w = depth.shape
    fx, fy, cx, cy = np.asarray(intrinsics, dtype=np.float64)
    if fx == 0.0 or fy == 0.0:
        raise ValueError("fx and fy must be non-zero")

    ys = np.arange(0, h, stride, dtype=np.int64)
    xs = np.arange(0, w, stride, dtype=np.int64)
    xx, yy = np.meshgrid(xs, ys)
    z = depth[yy, xx]
    valid = mask[yy, xx] & np.isfinite(z) & (z > 0.0)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.int64)

    x = (xx[valid].astype(np.float64) - cx) / fx * z[valid]
    y = (yy[valid].astype(np.float64) - cy) / fy * z[valid]
    points = np.stack((x, y, z[valid]), axis=1)
    pixels = np.stack((yy[valid], xx[valid]), axis=1)
    return points, pixels


def active_keyframe_indices(frame_timestamps: np.ndarray, keyframe_timestamps: np.ndarray) -> np.ndarray:
    keyframe_timestamps = np.asarray(keyframe_timestamps, dtype=np.int64)
    frame_timestamps = np.asarray(frame_timestamps, dtype=np.int64)
    if keyframe_timestamps.ndim != 1 or keyframe_timestamps.size == 0:
        raise ValueError("keyframe_timestamps must be a non-empty 1D array")
    idx = np.searchsorted(keyframe_timestamps, frame_timestamps, side="right") - 1
    return np.clip(idx, 0, keyframe_timestamps.shape[0] - 1).astype(np.int64)


def select_minimap_axes(positions: np.ndarray) -> tuple[int, int]:
    variances = np.var(np.asarray(positions, dtype=np.float64), axis=0)
    axes = np.argsort(variances)[::-1][:2]
    return int(axes[0]), int(axes[1])


def depth_to_rgb(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool) & np.isfinite(depth) & (depth > 0.0)
    if np.any(mask):
        lo, hi = np.percentile(depth[mask], [2.0, 98.0])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(depth[mask].min())
            hi = float(depth[mask].max() + 1e-6)
        norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    else:
        norm = np.zeros_like(depth, dtype=np.float32)
    gray = (norm * 255.0).astype(np.uint8)
    if np.any(mask):
        smoothed = cv2.bilateralFilter(gray, 5, 32.0, 32.0)
        gray = np.where(mask, smoothed, 0).astype(np.uint8)
    cmap = cv2.COLORMAP_TURBO if hasattr(cv2, "COLORMAP_TURBO") else cv2.COLORMAP_VIRIDIS
    rgb = cv2.cvtColor(cv2.applyColorMap(gray, cmap), cv2.COLOR_BGR2RGB)
    rgb[~mask] = np.asarray([5, 8, 14], dtype=np.uint8)
    return rgb


def metric_keyframe_depth(
    artifacts: StreamingArtifacts,
    keyframe_idx: int,
    *,
    max_depth: float = 80.0,
) -> tuple[np.ndarray, np.ndarray]:
    depth = artifacts.depths[keyframe_idx, 0].astype(np.float32) * float(artifacts.scales[keyframe_idx])
    mask = artifacts.masks[keyframe_idx, 0].astype(bool) & np.isfinite(depth) & (depth > 0.0)
    if max_depth > 0.0:
        mask &= depth < float(max_depth)
    return depth, mask


def _resize_panel(panel: np.ndarray, width: int) -> np.ndarray:
    h, w = panel.shape[:2]
    if w == width:
        return panel
    height = max(1, int(round(h * (float(width) / float(w)))))
    return cv2.resize(panel, (width, height), interpolation=cv2.INTER_AREA)


def _place_panel(canvas: np.ndarray, panel: np.ndarray, x: int, y: int, width: int) -> None:
    panel = _resize_panel(panel, width)
    border = 3
    shadow = 5
    ph, pw = panel.shape[:2]
    h, w = canvas.shape[:2]
    x = int(np.clip(x, 0, max(0, w - pw - 2 * border)))
    y = int(np.clip(y, 0, max(0, h - ph - 2 * border)))

    sx0, sy0 = min(w, x + shadow), min(h, y + shadow)
    sx1, sy1 = min(w, sx0 + pw + 2 * border), min(h, sy0 + ph + 2 * border)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0:sy1, sx0:sx1] = (0.72 * canvas[sy0:sy1, sx0:sx1]).astype(np.uint8)

    x0, y0 = x, y
    x1, y1 = min(w, x0 + pw + 2 * border), min(h, y0 + ph + 2 * border)
    canvas[y0:y1, x0:x1] = 255
    inner_x0, inner_y0 = x0 + border, y0 + border
    inner_x1, inner_y1 = min(w, inner_x0 + pw), min(h, inner_y0 + ph)
    canvas[inner_y0:inner_y1, inner_x0:inner_x1] = panel[: inner_y1 - inner_y0, : inner_x1 - inner_x0]


def make_minimap(
    positions: np.ndarray,
    current_idx: int,
    axes: tuple[int, int],
    *,
    size: tuple[int, int] = (360, 240),
) -> np.ndarray:
    width, height = size
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    if positions.shape[0] == 0:
        return canvas

    current_idx = int(np.clip(current_idx, 0, positions.shape[0] - 1))
    history = positions[: current_idx + 1, list(axes)].astype(np.float64)
    all_xy = positions[:, list(axes)].astype(np.float64)
    lo = all_xy.min(axis=0)
    hi = all_xy.max(axis=0)
    center = 0.5 * (lo + hi)
    radius = max(0.5 * float(np.max(hi - lo)), 1e-6) * 1.08
    margin = 18
    scale = min((width - 2 * margin) / (2 * radius), (height - 2 * margin) / (2 * radius))

    xy = (history - center) * scale
    xy[:, 0] += width * 0.5
    xy[:, 1] = height * 0.5 - xy[:, 1]
    pts = np.round(xy).astype(np.int32)
    if pts.shape[0] >= 2:
        cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], False, (40, 95, 175), 2, lineType=cv2.LINE_AA)
    cv2.circle(canvas, tuple(pts[0]), 5, (25, 25, 25), -1, lineType=cv2.LINE_AA)
    cv2.circle(canvas, tuple(pts[-1]), 7, (215, 55, 45), -1, lineType=cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (225, 225, 225), 1)
    return canvas


def compose_visualization_frame(
    base_rgb: np.ndarray,
    current_rgb: np.ndarray,
    keyframe_depth_rgb: np.ndarray,
    minimap_rgb: np.ndarray,
) -> np.ndarray:
    canvas = np.asarray(base_rgb, dtype=np.uint8).copy()
    h, w = canvas.shape[:2]
    margin = max(16, int(round(w * 0.0125)))
    panel_w = max(260, int(round(w * 0.24)))
    map_w = max(260, int(round(w * 0.22)))
    _place_panel(canvas, current_rgb, margin, margin, panel_w)
    _place_panel(canvas, keyframe_depth_rgb, w - margin - panel_w, margin, panel_w)
    map_panel = _resize_panel(minimap_rgb, map_w)
    _place_panel(canvas, map_panel, margin, h - margin - map_panel.shape[0] - 6, map_w)
    return canvas


def _make_lineset(points: np.ndarray, lines: np.ndarray, colors: np.ndarray):
    import open3d as o3d

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    line_set.lines = o3d.utility.Vector2iVector(lines.astype(np.int32))
    line_set.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return line_set


def _trajectory_lineset(positions: np.ndarray, color: tuple[float, float, float]):
    if positions.shape[0] < 2:
        return None
    lines = np.stack((np.arange(positions.shape[0] - 1), np.arange(1, positions.shape[0])), axis=1)
    colors = np.tile(np.asarray(color, dtype=np.float64)[None], (lines.shape[0], 1))
    return _make_lineset(positions, lines, colors)


def _camera_frustum_points(
    pose: np.ndarray,
    intrinsics: np.ndarray,
    image_size: tuple[int, int],
    frustum_size: float,
) -> np.ndarray:
    h, w = image_size
    fx, fy, cx, cy = intrinsics
    corners = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [(0.0 - cx) / fx * frustum_size, (0.0 - cy) / fy * frustum_size, frustum_size],
            [(w - cx) / fx * frustum_size, (0.0 - cy) / fy * frustum_size, frustum_size],
            [(w - cx) / fx * frustum_size, (h - cy) / fy * frustum_size, frustum_size],
            [(0.0 - cx) / fx * frustum_size, (h - cy) / fy * frustum_size, frustum_size],
        ],
        dtype=np.float64,
    )
    return camera_points_to_world(corners, pose)


def _frusta_lineset(
    poses: np.ndarray,
    indices: Iterable[int],
    active_keyframe: int,
    intrinsics: np.ndarray,
    image_size: tuple[int, int],
    frustum_size: float,
):
    points: list[np.ndarray] = []
    lines: list[tuple[int, int]] = []
    colors: list[tuple[float, float, float]] = []
    local_lines = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
    for idx in indices:
        offset = len(points) * 5
        points.append(_camera_frustum_points(poses[idx], intrinsics, image_size, frustum_size))
        color = (0.94, 0.20, 0.12) if int(idx) == int(active_keyframe) else (0.25, 0.25, 0.25)
        for a, b in local_lines:
            lines.append((offset + a, offset + b))
            colors.append(color)
    if not points:
        return None
    return _make_lineset(np.concatenate(points, axis=0), np.asarray(lines), np.asarray(colors))


def _auto_view(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = 0.5 * (positions.min(axis=0) + positions.max(axis=0))
    extent = max(float(np.max(positions.max(axis=0) - positions.min(axis=0))), 1.0)
    eye = center + np.asarray([0.65 * extent, -0.85 * extent, -1.8 * extent], dtype=np.float64)
    up = OPENCV_WORLD_UP.copy()
    return eye, center, up


class Open3DSceneRenderer:
    def __init__(
        self,
        width: int,
        height: int,
        *,
        point_size: float,
        line_width: float,
        eye: np.ndarray,
        center: np.ndarray,
        up: np.ndarray,
    ) -> None:
        import open3d as o3d
        from open3d.visualization import rendering

        self.o3d = o3d
        self.rendering = rendering
        self.width = int(width)
        self.height = int(height)
        self.eye = eye.astype(np.float64)
        self.center = center.astype(np.float64)
        self.up = up.astype(np.float64)
        self.renderer = rendering.OffscreenRenderer(self.width, self.height)
        self.scene = self.renderer.scene
        self.scene.set_background([1.0, 1.0, 1.0, 1.0])

        self.point_material = rendering.MaterialRecord()
        self.point_material.shader = "defaultUnlit"
        self.point_material.point_size = float(point_size)
        self.line_material = rendering.MaterialRecord()
        self.line_material.shader = "unlitLine"
        self.line_material.line_width = float(line_width)

    def render(
        self,
        *,
        points: np.ndarray,
        colors: np.ndarray,
        trajectory_positions: np.ndarray,
        keyframe_poses: np.ndarray,
        frustum_indices: list[int],
        active_keyframe: int,
        intrinsics: np.ndarray,
        image_size: tuple[int, int],
        frustum_size: float,
    ) -> np.ndarray:
        self.scene.clear_geometry()

        if points.shape[0]:
            point_cloud = self.o3d.geometry.PointCloud()
            point_cloud.points = self.o3d.utility.Vector3dVector(points.astype(np.float64))
            point_cloud.colors = self.o3d.utility.Vector3dVector(colors.astype(np.float64))
            self.scene.add_geometry("points", point_cloud, self.point_material)

        trajectory = _trajectory_lineset(trajectory_positions, (0.0, 0.32, 0.78))
        if trajectory is not None:
            self.scene.add_geometry("trajectory", trajectory, self.line_material)

        frusta = _frusta_lineset(keyframe_poses, frustum_indices, active_keyframe, intrinsics, image_size, frustum_size)
        if frusta is not None:
            self.scene.add_geometry("frusta", frusta, self.line_material)

        scene_points = [trajectory_positions]
        if points.shape[0]:
            scene_points.append(points)
        if frusta is not None:
            scene_points.append(np.asarray(frusta.points))
        all_points = np.concatenate([item for item in scene_points if item.shape[0]], axis=0)
        extent = max(float(np.linalg.norm(all_points.max(axis=0) - all_points.min(axis=0))), 1.0)
        far_clip = max(10.0, extent * 8.0, float(np.linalg.norm(self.eye - self.center)) * 4.0)
        self.renderer.setup_camera(60.0, self.center, self.eye, self.up, 0.01, far_clip)
        return np.asarray(self.renderer.render_to_image())


class PointAccumulator:
    def __init__(
        self,
        artifacts: StreamingArtifacts,
        frame_files: list[Path],
        *,
        source_frame_start: int,
        source_frame_skip: int,
        source_size: tuple[int, int],
        point_stride: int,
        max_points: int,
        max_depth: float,
    ) -> None:
        self.artifacts = artifacts
        self.frame_files = frame_files
        self.source_frame_start = int(source_frame_start)
        self.source_frame_skip = int(source_frame_skip)
        self.point_stride = int(point_stride)
        self.max_points = int(max_points)
        self.max_depth = float(max_depth)
        depth_size = artifacts.depths.shape[-2:]
        self.depth_intrinsics = scale_intrinsics_to_image(artifacts.intrinsics, source_size, depth_size)
        self.points = np.empty((0, 3), dtype=np.float32)
        self.colors = np.empty((0, 3), dtype=np.float32)
        self.next_keyframe = 0

    def _colors_for_keyframe(self, keyframe_idx: int, pixels: np.ndarray, depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
        timestamp = int(self.artifacts.keyframe_timestamps[keyframe_idx])
        try:
            frame_path = frame_file_for_timestamp(
                self.frame_files,
                timestamp,
                source_frame_start=self.source_frame_start,
                source_frame_skip=self.source_frame_skip,
            )
            rgb = read_rgb_frame(frame_path)
            rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA)
        except (IndexError, ValueError):
            rgb = depth_to_rgb(depth, mask)
        return rgb[pixels[:, 0], pixels[:, 1]].astype(np.float32) / 255.0

    def _load_keyframe_cloud(self, keyframe_idx: int) -> tuple[np.ndarray, np.ndarray]:
        depth, mask = metric_keyframe_depth(self.artifacts, keyframe_idx, max_depth=self.max_depth)
        points_cam, pixels = backproject_depth(depth, mask, self.depth_intrinsics, stride=self.point_stride)
        if points_cam.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
        points_world = camera_points_to_world(points_cam, self.artifacts.trajectory[keyframe_idx]).astype(np.float32)
        colors = self._colors_for_keyframe(keyframe_idx, pixels, depth, mask)
        return points_world, colors

    def add_through(self, keyframe_idx: int) -> None:
        while self.next_keyframe <= keyframe_idx:
            points, colors = self._load_keyframe_cloud(self.next_keyframe)
            if points.shape[0]:
                self.points = np.concatenate((self.points, points), axis=0)
                self.colors = np.concatenate((self.colors, colors), axis=0)
                if self.points.shape[0] > self.max_points:
                    keep = np.linspace(0, self.points.shape[0] - 1, self.max_points, dtype=np.int64)
                    self.points = self.points[keep]
                    self.colors = self.colors[keep]
            self.next_keyframe += 1


class VideoWriter:
    def __init__(self, path: Path, fps: float, size: tuple[int, int], mode: str) -> None:
        self.path = path
        self.fps = float(fps)
        self.size = size
        self.mode = mode
        self.writer = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if mode == "imageio":
            import imageio

            self.writer = imageio.get_writer(str(path), fps=self.fps, codec="libx264", quality=8, macro_block_size=1)
        elif mode == "cv2":
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(path), fourcc, self.fps, size)
            if not self.writer.isOpened():
                raise RuntimeError(f"could not open cv2 video writer for {path}")
        else:
            raise ValueError(f"unknown writer mode {mode!r}")

    def append(self, frame_rgb: np.ndarray) -> None:
        if self.mode == "imageio":
            self.writer.append_data(frame_rgb)
        else:
            self.writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        if self.writer is None:
            return
        if self.mode == "imageio":
            self.writer.close()
        else:
            self.writer.release()


def _make_video_writer(path: Path, fps: float, size: tuple[int, int], requested: str) -> VideoWriter:
    if requested != "auto":
        return VideoWriter(path, fps, size, requested)
    try:
        return VideoWriter(path, fps, size, "imageio")
    except Exception:
        return VideoWriter(path, fps, size, "cv2")


def _render_frame_indices(n_frames: int, start: int, end: int, stride: int) -> np.ndarray:
    if stride <= 0:
        raise ValueError("frame_stride must be positive")
    if start < 0:
        raise ValueError("frame_start must be non-negative")
    stop = n_frames if end < 0 else min(end, n_frames)
    if stop <= start:
        raise ValueError(f"frame range [{start}, {stop}) selects no frames")
    return np.arange(start, stop, stride, dtype=np.int64)


def preflight_open3d(width: int = 64, height: int = 64) -> dict[str, object]:
    import open3d as o3d

    eye = np.asarray([0.0, -2.0, 1.0], dtype=np.float64)
    center = np.zeros(3, dtype=np.float64)
    up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    renderer = Open3DSceneRenderer(width, height, point_size=10.0, line_width=3.0, eye=eye, center=center, up=up)
    image = renderer.render(
        points=np.asarray([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]], dtype=np.float32),
        colors=np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.7, 0.0]], dtype=np.float32),
        trajectory_positions=np.asarray([[-0.3, 0.0, 0.0], [0.3, 0.0, 0.0]], dtype=np.float64),
        keyframe_poses=np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
        frustum_indices=[0],
        active_keyframe=0,
        intrinsics=np.asarray([1.0, 1.0, 0.5, 0.5], dtype=np.float64),
        image_size=(1, 1),
        frustum_size=0.1,
    )
    image = image[..., :3]
    nonwhite_fraction = float(np.mean(np.any(image < 245, axis=2)))
    return {
        "open3d_version": o3d.__version__,
        "render_shape": list(image.shape),
        "render_dtype": str(image.dtype),
        "render_std": float(image.std()),
        "nonwhite_fraction": nonwhite_fraction,
    }


def render_visualization(options: RenderOptions) -> dict[str, object]:
    if options.output is None and options.save_frame_dir is None:
        raise ValueError("pass --output and/or --save-frame-dir")
    if options.width <= 0 or options.height <= 0:
        raise ValueError("width and height must be positive")
    if options.source_frame_skip <= 0:
        raise ValueError("source_frame_skip must be positive")
    if options.point_stride <= 0:
        raise ValueError("point_stride must be positive")
    if options.max_points <= 0:
        raise ValueError("max_points must be positive")
    if options.frustum_stride <= 0:
        raise ValueError("frustum_stride must be positive")

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
    active_keyframes = active_keyframe_indices(artifacts.frame_timestamps, artifacts.keyframe_timestamps)
    frame_indices = _render_frame_indices(
        artifacts.frame_trajectory.shape[0],
        options.frame_start,
        options.frame_end,
        options.frame_stride,
    )
    positions = camera_centers_from_tcw(artifacts.frame_trajectory)
    minimap_axes = select_minimap_axes(positions)
    eye, center, up = _auto_view(positions)
    if options.view_eye is not None:
        eye = np.asarray(options.view_eye, dtype=np.float64)
    if options.view_center is not None:
        center = np.asarray(options.view_center, dtype=np.float64)
    up = np.asarray(options.view_up, dtype=np.float64)

    renderer = Open3DSceneRenderer(
        options.width,
        options.height,
        point_size=options.point_size,
        line_width=options.line_width,
        eye=eye,
        center=center,
        up=up,
    )
    accumulator = PointAccumulator(
        artifacts,
        frame_files,
        source_frame_start=options.source_frame_start,
        source_frame_skip=options.source_frame_skip,
        source_size=source_size,
        point_stride=options.point_stride,
        max_points=options.max_points,
        max_depth=options.max_depth,
    )

    writer = None
    if options.output is not None:
        writer = _make_video_writer(options.output, options.fps, (options.width, options.height), options.writer)
    if options.save_frame_dir is not None:
        options.save_frame_dir.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    try:
        for output_idx, frame_idx in enumerate(frame_indices.tolist()):
            frame_timestamp = int(artifacts.frame_timestamps[frame_idx])
            active_keyframe = int(active_keyframes[frame_idx])
            accumulator.add_through(active_keyframe)

            if options.trail_length > 0:
                traj_start = max(0, frame_idx - options.trail_length)
            else:
                traj_start = 0
            trajectory_positions = positions[traj_start : frame_idx + 1]
            frustum_indices = list(range(0, active_keyframe + 1, options.frustum_stride))
            if active_keyframe not in frustum_indices:
                frustum_indices.append(active_keyframe)

            base = renderer.render(
                points=accumulator.points,
                colors=accumulator.colors,
                trajectory_positions=trajectory_positions,
                keyframe_poses=artifacts.trajectory,
                frustum_indices=frustum_indices,
                active_keyframe=active_keyframe,
                intrinsics=artifacts.intrinsics,
                image_size=source_size,
                frustum_size=options.frustum_size,
            )
            base = base[..., :3]
            current_rgb = read_rgb_frame(
                frame_file_for_timestamp(
                    frame_files,
                    frame_timestamp,
                    source_frame_start=options.source_frame_start,
                    source_frame_skip=options.source_frame_skip,
                )
            )
            depth, mask = metric_keyframe_depth(artifacts, active_keyframe, max_depth=options.max_depth)
            depth_rgb = depth_to_rgb(depth, mask)
            minimap = make_minimap(positions, frame_idx, minimap_axes)
            frame = compose_visualization_frame(base, current_rgb, depth_rgb, minimap)

            if writer is not None:
                writer.append(frame)
            if options.save_frame_dir is not None:
                out_path = options.save_frame_dir / f"frame_{output_idx:06d}.png"
                cv2.imwrite(str(out_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            frame_count += 1
            if frame_count == 1 or frame_count % 25 == 0:
                print(
                    f"rendered {frame_count}/{len(frame_indices)} frames "
                    f"(frame={frame_idx}, keyframe={active_keyframe}, points={accumulator.points.shape[0]})",
                    flush=True,
                )
    finally:
        if writer is not None:
            writer.close()

    return {
        "artifact_root": str(options.artifact_root),
        "frame_dir": str(options.frame_dir),
        "output": str(options.output) if options.output is not None else "",
        "save_frame_dir": str(options.save_frame_dir) if options.save_frame_dir is not None else "",
        "rendered_frames": frame_count,
        "source_frames": int(artifacts.frame_trajectory.shape[0]),
        "keyframes": int(artifacts.trajectory.shape[0]),
        "point_count": int(accumulator.points.shape[0]),
    }


def _parse_vec3(values: list[float] | None) -> tuple[float, float, float] | None:
    if values is None:
        return None
    return float(values[0]), float(values[1]), float(values[2])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an offline Open3D video from GeoNT streaming artifacts.")
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--frame-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=-1)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--source-frame-start", type=int, default=0)
    parser.add_argument("--source-frame-skip", type=int, default=1)
    parser.add_argument("--point-stride", type=int, default=8)
    parser.add_argument("--max-points", type=int, default=1_500_000)
    parser.add_argument("--frustum-stride", type=int, default=25)
    parser.add_argument("--frustum-size", type=float, default=1.5)
    parser.add_argument("--point-size", type=float, default=4.0)
    parser.add_argument("--line-width", type=float, default=3.0)
    parser.add_argument("--trail-length", type=int, default=0, help="0 keeps the full trajectory history.")
    parser.add_argument("--max-depth", type=float, default=80.0, help="Hide predicted depth values at or beyond this distance; <=0 disables.")
    parser.add_argument("--save-frame-dir", type=Path)
    parser.add_argument("--writer", choices=("auto", "imageio", "cv2"), default="auto")
    parser.add_argument("--view-eye", type=float, nargs=3)
    parser.add_argument("--view-center", type=float, nargs=3)
    parser.add_argument("--view-up", type=float, nargs=3, default=[0.0, -1.0, 0.0])
    parser.add_argument("--preflight", action="store_true", help="Create a tiny Open3D offscreen render and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.preflight:
        print(preflight_open3d())
        return
    if args.artifact_root is None or args.frame_dir is None:
        raise SystemExit("--artifact-root and --frame-dir are required unless --preflight is set")

    summary = render_visualization(
        RenderOptions(
            artifact_root=args.artifact_root,
            frame_dir=args.frame_dir,
            output=args.output,
            width=args.width,
            height=args.height,
            fps=args.fps,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
            frame_stride=args.frame_stride,
            source_frame_start=args.source_frame_start,
            source_frame_skip=args.source_frame_skip,
            point_stride=args.point_stride,
            max_points=args.max_points,
            frustum_stride=args.frustum_stride,
            frustum_size=args.frustum_size,
            point_size=args.point_size,
            line_width=args.line_width,
            trail_length=args.trail_length,
            max_depth=args.max_depth,
            save_frame_dir=args.save_frame_dir,
            writer=args.writer,
            view_eye=_parse_vec3(args.view_eye),
            view_center=_parse_vec3(args.view_center),
            view_up=tuple(float(v) for v in args.view_up),
        )
    )
    print(summary)


if __name__ == "__main__":
    main()
