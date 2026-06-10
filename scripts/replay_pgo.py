#!/usr/bin/env python3
from __future__ import annotations

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

from geont_runtime.slam.pgo import optimize_sim3_pose_graph
from geont_runtime.slam.pgo.replay import load_pgo_replay_graph


def _config_path(value) -> Path:
    return Path(to_absolute_path(str(value)))


def _resolve_device(device_name: str, backend: str) -> torch.device:
    if device_name == "auto":
        if backend == "cuda_eigen":
            return torch.device("cuda")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _jsonable_info(info: dict) -> dict:
    out = {}
    for key, value in info.items():
        if isinstance(value, (np.integer, np.floating)):
            out[key] = value.item()
        else:
            out[key] = value
    return out


@hydra.main(version_base=None, config_path="../configs", config_name="pgo_replay")
def main(cfg: DictConfig) -> None:
    graph_path = _config_path(cfg.graph)
    output_path = _config_path(cfg.output)
    if not graph_path.exists():
        raise FileNotFoundError(f"PGO replay graph not found: {graph_path}")
    if int(cfg.repeat) <= 0:
        raise ValueError("repeat must be positive")

    backend = str(cfg.backend)
    if backend not in {"torch", "cuda_eigen"}:
        raise ValueError("backend must be 'cuda_eigen' or 'torch'")
    mode = str(cfg.mode)
    if bool(cfg.rotation_only):
        mode = "rotation_only"
    if mode not in {"rotation_only", "staged", "se3_scale"}:
        raise ValueError("mode must be 'rotation_only', 'staged', or 'se3_scale'")
    device = _resolve_device(str(cfg.device), backend)
    dtype = torch.float32
    graph = load_pgo_replay_graph(graph_path, device=device, dtype=dtype)

    timings = []
    result = None
    with torch.inference_mode():
        for _ in range(int(cfg.repeat)):
            _sync_cuda(device)
            start = time.perf_counter()
            result = optimize_sim3_pose_graph(
                n_nodes=int(graph["n_nodes"]),
                ii=graph["ii"],
                jj=graph["jj"],
                rel_poses=graph["relative_pose"],
                rel_scales=graph["relative_scale"],
                edge_conf=graph["confidence"],
                initial_poses=graph["initial_poses"],
                initial_log_scales=graph["initial_log_scales"],
                anchor=int(graph["anchor"]),
                n_iters=int(cfg.iters),
                damping=float(cfg.damping),
                lm_max_attempts=int(cfg.lm_max_attempts),
                huber_delta=float(cfg.huber_delta),
                scale_conf=float(cfg.scale_conf),
                mode=mode,
                backend=backend,
            )
            _sync_cuda(device)
            timings.append(time.perf_counter() - start)

    assert result is not None
    summary = {
        "graph": str(graph_path),
        "output": str(output_path),
        "backend": backend,
        "device": str(device),
        "mode": mode,
        "rotation_only": bool(cfg.rotation_only),
        "iters": int(cfg.iters),
        "damping": float(cfg.damping),
        "lm_max_attempts": int(cfg.lm_max_attempts),
        "huber_delta": float(cfg.huber_delta),
        "scale_conf": float(cfg.scale_conf),
        "repeat": int(cfg.repeat),
        "elapsed_sec_mean": float(np.mean(timings)),
        "elapsed_sec_median": float(np.median(timings)),
        "elapsed_sec_min": float(np.min(timings)),
        "elapsed_sec_max": float(np.max(timings)),
        "pgo_info": _jsonable_info(result.info),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        poses=result.poses.cpu().numpy(),
        log_scales=result.log_scales.cpu().numpy(),
        scales=torch.exp(result.log_scales).cpu().numpy(),
        summary=np.array(json.dumps(summary)),
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
