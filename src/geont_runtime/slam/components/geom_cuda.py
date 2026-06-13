from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_EXTENSION = None


def _source_paths() -> list[str]:
    root = Path(__file__).resolve().parent / "cuda"
    return [
        str(root / "geom_cuda.cpp"),
        str(root / "geom_cuda.cu"),
    ]


def _extension():
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    if not torch.cuda.is_available():
        raise RuntimeError("geometry CUDA requires torch.cuda.is_available()")
    _EXTENSION = load(
        name="gnt_geom_cuda",
        sources=_source_paths(),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
    return _EXTENSION


def is_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        _extension()
    except Exception:
        return False
    return True


def _check_cuda_float32(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type != "cuda":
        raise RuntimeError(f"geometry CUDA requires {name} to be a CUDA tensor")
    if tensor.dtype != torch.float32:
        raise RuntimeError(f"geometry CUDA requires {name} to be torch.float32")
    return tensor.contiguous()


def _check_cuda_bool(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type != "cuda":
        raise RuntimeError(f"geometry CUDA requires {name} to be a CUDA tensor")
    if tensor.dtype != torch.bool:
        raise RuntimeError(f"geometry CUDA requires {name} to be torch.bool")
    return tensor.contiguous()


def _check_cuda_int64(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type != "cuda":
        raise RuntimeError(f"geometry CUDA requires {name} to be a CUDA tensor")
    if tensor.dtype != torch.long:
        raise RuntimeError(f"geometry CUDA requires {name} to be torch.long")
    return tensor.contiguous()


def projection_distance(
    poses: torch.Tensor,
    depths: torch.Tensor,
    scales: torch.Tensor,
    masks: torch.Tensor,
    intrinsics: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    stride: int,
) -> torch.Tensor:
    ext = _extension()
    return ext.projection_distance(
        _check_cuda_float32("poses", poses),
        _check_cuda_float32("depths", depths),
        _check_cuda_float32("scales", scales),
        _check_cuda_bool("masks", masks),
        _check_cuda_float32("intrinsics", intrinsics),
        _check_cuda_int64("ii", ii),
        _check_cuda_int64("jj", jj),
        int(stride),
    )
