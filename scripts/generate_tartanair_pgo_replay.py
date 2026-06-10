#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys


def _ensure_conda_libstdcpp() -> None:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix or os.environ.get("GNT_CONDA_LIBSTDCXX_READY") == "1":
        return

    conda_lib = Path(conda_prefix) / "lib"
    if not conda_lib.exists():
        return

    current = os.environ.get("LD_LIBRARY_PATH", "")
    paths = [path for path in current.split(":") if path]
    if paths and Path(paths[0]) == conda_lib:
        return

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{conda_lib}:{current}" if current else str(conda_lib)
    env["GNT_CONDA_LIBSTDCXX_READY"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


_ensure_conda_libstdcpp()

import torch


def _override_key(arg: str) -> str:
    if arg.startswith("+"):
        arg = arg[1:]
    return arg.split("=", 1)[0]


def _has_override(args: list[str], key: str) -> bool:
    return any(_override_key(arg) == key for arg in args)


def main() -> None:
    user_overrides = sys.argv[1:]
    evaluator = Path(__file__).with_name("evaluate_tartanair_pgo.py")
    if any(arg in {"-h", "--help"} for arg in user_overrides):
        sys.argv = [str(evaluator), "--config-name=tartanair_pgo_replay_full_p011", "--help"]
        runpy.run_path(str(evaluator), run_name="__main__")
        return
    if not _has_override(user_overrides, "ckpt"):
        raise SystemExit(
            "usage: python scripts/generate_tartanair_pgo_replay.py "
            "ckpt=/path/to/checkpoint.pth output=outputs/pgo_replay_experiment"
        )
    if not _has_override(user_overrides, "output"):
        raise SystemExit("please pass a fresh output=... directory for this checkpoint")

    torch.backends.cudnn.enabled = False
    sys.argv = [str(evaluator), "--config-name=tartanair_pgo_replay_full_p011", *user_overrides]
    runpy.run_path(str(evaluator), run_name="__main__")


if __name__ == "__main__":
    main()
