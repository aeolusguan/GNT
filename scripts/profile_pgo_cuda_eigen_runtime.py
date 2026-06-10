#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from geont_runtime.slam.pgo import optimizer as pgo
from geont_runtime.slam.pgo import cuda_backend as pgo_cuda
from geont_runtime.slam.pgo import cuda_eigen as pgo_cuda_eigen  # noqa: E402
from geont_runtime.slam.pgo.replay import load_pgo_replay_graph  # noqa: E402


def _sync() -> None:
    torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile high-level cuda_eigen PGO runtime sections.")
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--mode", default="se3_scale", choices=["staged", "se3_scale", "rotation_only"])
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--damping", type=float, default=1.0e-3)
    parser.add_argument("--lm-max-attempts", type=int, default=5)
    parser.add_argument("--huber-delta", type=float, default=0.05)
    parser.add_argument("--scale-conf", type=float, default=0.01)
    args = parser.parse_args()

    timings = defaultdict(float)
    counts = defaultdict(int)

    def wrap_function(module, name: str, label: str):
        original = getattr(module, name)

        def wrapped(*fn_args, **fn_kwargs):
            _sync()
            start = time.perf_counter()
            out = original(*fn_args, **fn_kwargs)
            _sync()
            timings[label] += time.perf_counter() - start
            counts[label] += 1
            return out

        setattr(module, name, wrapped)

    def wrap_method(cls, name: str, label: str):
        original = getattr(cls, name)

        def wrapped(self, *fn_args, **fn_kwargs):
            _sync()
            start = time.perf_counter()
            out = original(self, *fn_args, **fn_kwargs)
            _sync()
            timings[label] += time.perf_counter() - start
            counts[label] += 1
            return out

        setattr(cls, name, wrapped)

    wrap_function(pgo_cuda_eigen, "_rotation_cuda_blocks", "build.rotation_native")
    wrap_function(pgo_cuda_eigen, "_translation_scale_cuda_blocks", "build.translation_scale_native")
    wrap_function(pgo_cuda_eigen, "_se3_scale_cuda_blocks", "build.se3_scale_native")
    wrap_function(pgo_cuda_eigen, "_rotation_residuals", "residual.rotation")
    wrap_function(pgo_cuda_eigen, "_translation_residuals", "residual.translation")
    wrap_function(pgo_cuda_eigen, "_scaled_se3_residuals", "residual.se3_scale")
    wrap_function(pgo_cuda_eigen, "_apply_rotation_delta", "apply.rotation")
    wrap_function(pgo_cuda_eigen, "_apply_translation_scale_delta", "apply.translation_scale")
    wrap_function(pgo_cuda_eigen, "_apply_delta", "apply.se3_scale")
    wrap_method(pgo_cuda.RotationEigenSimplicialLLTSolver, "solve", "solve.rotation_eigen")
    wrap_method(pgo_cuda.TranslationScaleEigenSimplicialLLTSolver, "solve", "solve.translation_scale_eigen")
    wrap_method(pgo_cuda.Se3ScaleEigenSimplicialLLTSolver, "solve", "solve.se3_scale_eigen")

    graph = load_pgo_replay_graph(args.graph, device=torch.device("cuda"), dtype=torch.float32)
    _sync()
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
        n_iters=args.iters,
        damping=args.damping,
        lm_max_attempts=args.lm_max_attempts,
        huber_delta=args.huber_delta,
        scale_conf=args.scale_conf,
        mode=args.mode,
        backend="cuda_eigen",
    )
    _sync()
    total = time.perf_counter() - start

    rows = []
    for label in sorted(timings):
        rows.append(
            {
                "section": label,
                "count": counts[label],
                "total_ms": timings[label] * 1000.0,
                "mean_ms": timings[label] * 1000.0 / max(counts[label], 1),
            }
        )
    print(
        json.dumps(
            {
                "mode": args.mode,
                "total_wall_ms": total * 1000.0,
                "info_runtime_ms": float(result.info.get("runtime_sec", 0.0)) * 1000.0,
                "cost": float(result.info.get("cost", 0.0)),
                "solver_failures": int(result.info.get("solver_failures", 0)),
                "rows": rows,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
