from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


PGO_REPLAY_TENSOR_KEYS = (
    "ii",
    "jj",
    "relative_pose",
    "relative_scale",
    "confidence",
    "initial_poses",
    "initial_log_scales",
)


def make_pgo_replay_graph(
    *,
    n_nodes: int,
    anchor: int,
    ii: torch.Tensor,
    jj: torch.Tensor,
    relative_pose: torch.Tensor,
    relative_scale: torch.Tensor,
    confidence: torch.Tensor,
    initial_poses: torch.Tensor,
    initial_log_scales: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Pack the exact PGO input graph for later replay.

    Tensor shapes:
        ii, jj: (E,)
        relative_pose: (E, 7)
        relative_scale: (E,)
        confidence: (E, 2)
        initial_poses: (N, 7)
        initial_log_scales: (N,)
    """
    device = relative_pose.device
    return {
        "n_nodes": torch.tensor(int(n_nodes), device=device, dtype=torch.long),
        "anchor": torch.tensor(int(anchor), device=device, dtype=torch.long),
        "ii": ii,
        "jj": jj,
        "relative_pose": relative_pose,
        "relative_scale": relative_scale,
        "confidence": confidence,
        "initial_poses": initial_poses,
        "initial_log_scales": initial_log_scales,
    }


def pgo_replay_npz_payload(replay_graph: dict[str, torch.Tensor] | None) -> dict[str, np.ndarray]:
    if not replay_graph:
        return {}
    payload = {}
    for key, value in replay_graph.items():
        payload[f"pgo_{key}"] = value.cpu().numpy()
    return payload


def save_pgo_replay_graph(
    path: Path,
    replay_graph: dict[str, torch.Tensor],
    *,
    pgo_info: dict | None = None,
    timestamps: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = pgo_replay_npz_payload(replay_graph)
    if pgo_info is not None:
        payload["pgo_info"] = np.array(json.dumps(pgo_info))
    if timestamps is not None:
        payload["timestamps"] = np.asarray(timestamps)
    np.savez_compressed(path, **payload)


def load_pgo_replay_graph(
    path: Path,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> dict[str, int | torch.Tensor]:
    with np.load(path, allow_pickle=False) as data:
        required_keys = ["pgo_n_nodes", "pgo_anchor"] + [f"pgo_{key}" for key in PGO_REPLAY_TENSOR_KEYS]
        missing = [key for key in required_keys if key not in data]
        if missing:
            raise KeyError(
                f"{path} is missing replay keys {missing}; run the updated SLAM/evaluation pipeline once "
                "to save a PGO replay graph."
            )
        graph: dict[str, int | torch.Tensor] = {
            "n_nodes": int(np.asarray(data["pgo_n_nodes"]).item()),
            "anchor": int(np.asarray(data["pgo_anchor"]).item()),
            "ii": torch.as_tensor(data["pgo_ii"], device=device, dtype=torch.long),
            "jj": torch.as_tensor(data["pgo_jj"], device=device, dtype=torch.long),
            "relative_pose": torch.as_tensor(data["pgo_relative_pose"], device=device, dtype=dtype),
            "relative_scale": torch.as_tensor(data["pgo_relative_scale"], device=device, dtype=dtype),
            "confidence": torch.as_tensor(data["pgo_confidence"], device=device, dtype=dtype),
            "initial_poses": torch.as_tensor(data["pgo_initial_poses"], device=device, dtype=dtype),
            "initial_log_scales": torch.as_tensor(data["pgo_initial_log_scales"], device=device, dtype=dtype),
        }
    return graph
