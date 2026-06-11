from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .components.buffer import GraphBuffer
from .components.factor_graph import PoseGraphEdges
from .pgo.replay import PGO_REPLAY_TENSOR_KEYS, pgo_replay_npz_payload


BUFFER_KEYS = (
    "tstamp",
    "poses",
    "intrinsics",
    "depths",
    "depths_sens_normed",
    "depths_sens_scale",
    "non_sky_masks",
    "fmaps",
    "bases",
)

EDGE_KEYS = (
    "ii",
    "jj",
    "relative_pose",
    "relative_scale",
    "confidence",
)


def _tensor_np(tensor: torch.Tensor, n_frames: int | None = None) -> np.ndarray:
    if n_frames is not None:
        tensor = tensor[:n_frames]
    return tensor.cpu().numpy()


def save_initializer_snapshot(
    path: Path,
    buffer: GraphBuffer,
    edges: PoseGraphEdges,
    *,
    metadata: dict | None = None,
) -> None:
    """Save the one-pass initializer state needed to replay offline frontend."""
    n_frames = int(buffer.n_frames)
    payload = {
        "format": np.array("geont_initializer_frontend_snapshot_v1"),
        "n_frames": np.array(n_frames, dtype=np.int64),
        "height": np.array(int(buffer.height), dtype=np.int64),
        "width": np.array(int(buffer.width), dtype=np.int64),
        "metadata": np.array(json.dumps(metadata or {})),
        "edge_pgo_info": np.array(json.dumps(edges.pgo_info or {})),
        "tstamp": _tensor_np(buffer.tstamp, n_frames),
        "poses": _tensor_np(buffer.poses, n_frames),
        "intrinsics": _tensor_np(buffer.intrinsics),
        "depths": _tensor_np(buffer.depths, n_frames),
        "depths_sens_normed": _tensor_np(buffer.depths_sens_normed, n_frames),
        "depths_sens_scale": _tensor_np(buffer.depths_sens_scale, n_frames),
        "non_sky_masks": _tensor_np(buffer.non_sky_masks, n_frames),
        "fmaps": _tensor_np(buffer.fmaps, n_frames),
        "bases": _tensor_np(buffer.bases, n_frames),
        "edge_ii": _tensor_np(edges.ii),
        "edge_jj": _tensor_np(edges.jj),
        "edge_relative_pose": _tensor_np(edges.relative_pose),
        "edge_relative_scale": _tensor_np(edges.relative_scale),
        "edge_confidence": _tensor_np(edges.confidence),
        **pgo_replay_npz_payload(edges.pgo_replay),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def _required_keys() -> list[str]:
    return [
        "format",
        "n_frames",
        "height",
        "width",
        "metadata",
        "edge_pgo_info",
        *BUFFER_KEYS,
        *(f"edge_{key}" for key in EDGE_KEYS),
    ]


def _load_tensor(data, key: str, *, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
    tensor = torch.as_tensor(data[key], device=device)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _load_pgo_replay(data, *, device: torch.device) -> dict[str, torch.Tensor]:
    if "pgo_n_nodes" not in data.files:
        return {}

    replay = {
        "n_nodes": _load_tensor(data, "pgo_n_nodes", device=device, dtype=torch.long),
        "anchor": _load_tensor(data, "pgo_anchor", device=device, dtype=torch.long),
    }
    for key in PGO_REPLAY_TENSOR_KEYS:
        dtype = torch.long if key in {"ii", "jj"} else torch.float
        replay[key] = _load_tensor(data, f"pgo_{key}", device=device, dtype=dtype)
    return replay


def load_initializer_snapshot(path: Path, *, device: torch.device) -> tuple[GraphBuffer, PoseGraphEdges, dict]:
    """Load a saved initializer snapshot into a frontend-ready buffer and edge store."""
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in _required_keys() if key not in data]
        if missing:
            raise KeyError(f"{path} is missing initializer snapshot keys {missing}")

        snapshot_format = str(np.asarray(data["format"]).item())
        if snapshot_format != "geont_initializer_frontend_snapshot_v1":
            raise ValueError(f"unsupported initializer snapshot format: {snapshot_format}")

        n_frames = int(np.asarray(data["n_frames"]).item())
        height = int(np.asarray(data["height"]).item())
        width = int(np.asarray(data["width"]).item())
        metadata = json.loads(str(np.asarray(data["metadata"]).item()))
        edge_pgo_info = json.loads(str(np.asarray(data["edge_pgo_info"]).item()))

        buffer = GraphBuffer(height=height, width=width, buffer_size=n_frames, device=device)
        buffer.n_frames = n_frames
        buffer.tstamp[:n_frames] = _load_tensor(data, "tstamp", device=device, dtype=buffer.tstamp.dtype)
        buffer.poses[:n_frames] = _load_tensor(data, "poses", device=device, dtype=buffer.poses.dtype)
        buffer.intrinsics[:] = _load_tensor(data, "intrinsics", device=device, dtype=buffer.intrinsics.dtype)
        buffer.depths[:n_frames] = _load_tensor(data, "depths", device=device, dtype=buffer.depths.dtype)
        buffer.depths_sens_normed[:n_frames] = _load_tensor(
            data,
            "depths_sens_normed",
            device=device,
            dtype=buffer.depths_sens_normed.dtype,
        )
        buffer.depths_sens_scale[:n_frames] = _load_tensor(
            data,
            "depths_sens_scale",
            device=device,
            dtype=buffer.depths_sens_scale.dtype,
        )
        buffer.non_sky_masks[:n_frames] = _load_tensor(
            data,
            "non_sky_masks",
            device=device,
            dtype=buffer.non_sky_masks.dtype,
        )
        buffer.fmaps[:n_frames] = _load_tensor(data, "fmaps", device=device, dtype=buffer.fmaps.dtype)
        buffer.bases[:n_frames] = _load_tensor(data, "bases", device=device, dtype=buffer.bases.dtype)

        edges = PoseGraphEdges(device)
        edges.ii = _load_tensor(data, "edge_ii", device=device, dtype=torch.long)
        edges.jj = _load_tensor(data, "edge_jj", device=device, dtype=torch.long)
        edges.relative_pose = _load_tensor(data, "edge_relative_pose", device=device, dtype=torch.float)
        edges.relative_scale = _load_tensor(data, "edge_relative_scale", device=device, dtype=torch.float)
        edges.confidence = _load_tensor(data, "edge_confidence", device=device, dtype=torch.float)
        edges.pgo_info = edge_pgo_info
        edges.pgo_replay = _load_pgo_replay(data, device=device)

    return buffer, edges, metadata
