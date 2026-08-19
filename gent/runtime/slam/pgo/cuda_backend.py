from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_EXTENSION = None
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_EIGEN_INCLUDE_DIR = _PROJECT_ROOT / "third_party" / "eigen" / "upstream"


def _source_paths() -> list[str]:
    root = Path(__file__).resolve().parent / "cuda"
    return [
        str(root / "pgo_cuda.cpp"),
        str(root / "pgo_cuda_kernel.cu"),
    ]


def _eigen_include_paths() -> list[str]:
    return [os.environ.get("GNT_EIGEN_INCLUDE_DIR", str(_EIGEN_INCLUDE_DIR))]


def _extension():
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    if not torch.cuda.is_available():
        raise RuntimeError("PGO CUDA backend requires torch.cuda.is_available()")
    _EXTENSION = load(
        name="gnt_pgo_cuda",
        sources=_source_paths(),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        extra_include_paths=_eigen_include_paths(),
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
        raise RuntimeError(f"PGO CUDA backend requires {name} to be a CUDA tensor")
    if tensor.dtype != torch.float32:
        raise RuntimeError(f"PGO CUDA backend requires {name} to be torch.float32")
    return tensor.contiguous()


def _check_cuda_int64(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type != "cuda":
        raise RuntimeError(f"PGO CUDA backend requires {name} to be a CUDA tensor")
    if tensor.dtype != torch.long:
        raise RuntimeError(f"PGO CUDA backend requires {name} to be torch.long")
    return tensor.contiguous()


def _check_cpu_float64(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type != "cpu":
        raise RuntimeError(f"PGO Eigen solver requires {name} to be a CPU tensor")
    if tensor.dtype != torch.float64:
        raise RuntimeError(f"PGO Eigen solver requires {name} to be torch.float64")
    return tensor.contiguous()


def build_rotation_blocks(
    rotations: torch.Tensor,
    meas_rotations: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    sqrt_info: torch.Tensor,
    robust: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build rotation-only PGO blocks in the native CUDA extension."""
    ext = _extension()
    return ext.rotation_blocks(
        _check_cuda_float32("rotations", rotations),
        _check_cuda_float32("meas_rotations", meas_rotations),
        _check_cuda_int64("ii", ii),
        _check_cuda_int64("jj", jj),
        _check_cuda_float32("sqrt_info", sqrt_info),
        _check_cuda_float32("robust", robust),
    )


def build_translation_scale_blocks(
    poses: torch.Tensor,
    log_s: torch.Tensor,
    rel_poses: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    sqrt_info: torch.Tensor,
    robust: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build translation+scale PGO blocks in the native CUDA extension."""
    ext = _extension()
    return ext.translation_scale_blocks(
        _check_cuda_float32("poses", poses),
        _check_cuda_float32("log_s", log_s),
        _check_cuda_float32("rel_poses", rel_poses),
        _check_cuda_int64("ii", ii),
        _check_cuda_int64("jj", jj),
        _check_cuda_float32("sqrt_info", sqrt_info),
        _check_cuda_float32("robust", robust),
    )


def build_se3_scale_blocks(
    poses: torch.Tensor,
    log_s: torch.Tensor,
    rel_poses: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    sqrt_info: torch.Tensor,
    robust: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build SE3+scale PGO blocks in the native CUDA extension."""
    ext = _extension()
    return ext.se3_scale_blocks(
        _check_cuda_float32("poses", poses),
        _check_cuda_float32("log_s", log_s),
        _check_cuda_float32("rel_poses", rel_poses),
        _check_cuda_int64("ii", ii),
        _check_cuda_int64("jj", jj),
        _check_cuda_float32("sqrt_info", sqrt_info),
        _check_cuda_float32("robust", robust),
    )


def build_se3_scale_weighted_blocks(
    poses: torch.Tensor,
    log_s: torch.Tensor,
    rel_poses: torch.Tensor,
    rel_log_scales: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    sqrt_info: torch.Tensor,
    scale_sqrt_info: torch.Tensor,
    huber_delta: float,
    buffers: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build Huber-weighted SE3+scale blocks and return current/unrobust costs."""
    ext = _extension()
    source_block, target_block, edge_residual, robust, summary = buffers
    ext.se3_scale_weighted_blocks(
        _check_cuda_float32("poses", poses),
        _check_cuda_float32("log_s", log_s),
        _check_cuda_float32("rel_poses", rel_poses),
        _check_cuda_float32("rel_log_scales", rel_log_scales),
        _check_cuda_int64("ii", ii),
        _check_cuda_int64("jj", jj),
        _check_cuda_float32("sqrt_info", sqrt_info),
        _check_cuda_float32("scale_sqrt_info", scale_sqrt_info),
        float(huber_delta),
        _check_cuda_float32("source_block", source_block),
        _check_cuda_float32("target_block", target_block),
        _check_cuda_float32("edge_residual", edge_residual),
        _check_cuda_float32("robust", robust),
        _check_cuda_float32("summary", summary),
    )
    return buffers


class RotationEigenSimplicialLLTSolver:
    """Cached CPU Eigen LLT solver for one fixed rotation graph."""

    def __init__(self, ii: torch.Tensor, jj: torch.Tensor, n_nodes: int, anchor: int):
        ext = _extension()
        self.n_nodes = int(n_nodes)
        self._solver = ext.RotationEigenSimplicialLLTSolver(
            _check_cuda_int64("ii", ii),
            _check_cuda_int64("jj", jj),
            self.n_nodes,
            int(anchor),
        )

    def solve(
        self,
        source_block: torch.Tensor,
        target_block: torch.Tensor,
        edge_residual: torch.Tensor,
        damping: float,
    ) -> torch.Tensor:
        step = self._solver.solve(
            _check_cuda_float32("source_block", source_block),
            _check_cuda_float32("target_block", target_block),
            _check_cuda_float32("edge_residual", edge_residual),
            float(damping),
        )
        return step.view(self.n_nodes, 3)


class TranslationScaleEigenSimplicialLLTSolver:
    """Cached CPU Eigen LLT solver for one fixed translation+scale graph."""

    def __init__(self, ii: torch.Tensor, jj: torch.Tensor, n_nodes: int, anchor: int):
        ext = _extension()
        self.n_nodes = int(n_nodes)
        self._solver = ext.TranslationScaleEigenSimplicialLLTSolver(
            _check_cuda_int64("ii", ii),
            _check_cuda_int64("jj", jj),
            self.n_nodes,
            int(anchor),
        )

    def solve(
        self,
        source_block: torch.Tensor,
        target_block: torch.Tensor,
        edge_residual: torch.Tensor,
        damping: float,
    ) -> torch.Tensor:
        step = self._solver.solve(
            _check_cuda_float32("source_block", source_block),
            _check_cuda_float32("target_block", target_block),
            _check_cuda_float32("edge_residual", edge_residual),
            float(damping),
        )
        return step.view(self.n_nodes, 4)


class Se3ScaleEigenSimplicialLDLTSolver:
    """Cached CPU Eigen LDLT solver for one fixed SE3+scale graph."""

    def __init__(self, ii: torch.Tensor, jj: torch.Tensor, n_nodes: int, anchor: int):
        ext = _extension()
        self.n_nodes = int(n_nodes)
        self._solver = ext.Se3ScaleEigenSimplicialLDLTSolver(
            _check_cuda_int64("ii", ii),
            _check_cuda_int64("jj", jj),
            self.n_nodes,
            int(anchor),
        )

    def solve(
        self,
        source_block: torch.Tensor,
        target_block: torch.Tensor,
        edge_residual: torch.Tensor,
        damping: float,
    ) -> torch.Tensor:
        step = self._solver.solve(
            _check_cuda_float32("source_block", source_block),
            _check_cuda_float32("target_block", target_block),
            _check_cuda_float32("edge_residual", edge_residual),
            float(damping),
        )
        return step.view(self.n_nodes, 7)

    def solve_multi_rhs(
        self,
        source_block: torch.Tensor,
        target_block: torch.Tensor,
        edge_residual: torch.Tensor,
        scale_rhs: torch.Tensor,
        damping: float,
    ) -> torch.Tensor:
        solutions = self._solver.solve_multi_rhs(
            _check_cuda_float32("source_block", source_block),
            _check_cuda_float32("target_block", target_block),
            _check_cuda_float32("edge_residual", edge_residual),
            _check_cpu_float64("scale_rhs", scale_rhs),
            float(damping),
        )
        return solutions.view(self.n_nodes, 7, scale_rhs.shape[1] + 1)
