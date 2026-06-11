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
    prior_log_s: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    sqrt_info: torch.Tensor,
    robust: torch.Tensor,
    scale_prior_diag: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build translation+scale PGO blocks in the native CUDA extension."""
    ext = _extension()
    return ext.translation_scale_blocks(
        _check_cuda_float32("poses", poses),
        _check_cuda_float32("log_s", log_s),
        _check_cuda_float32("rel_poses", rel_poses),
        _check_cuda_float32("prior_log_s", prior_log_s),
        _check_cuda_int64("ii", ii),
        _check_cuda_int64("jj", jj),
        _check_cuda_float32("sqrt_info", sqrt_info),
        _check_cuda_float32("robust", robust),
        float(scale_prior_diag),
    )


def build_se3_scale_blocks(
    poses: torch.Tensor,
    log_s: torch.Tensor,
    rel_poses: torch.Tensor,
    prior_log_s: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    sqrt_info: torch.Tensor,
    robust: torch.Tensor,
    scale_prior_diag: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build SE3+scale PGO blocks in the native CUDA extension."""
    ext = _extension()
    return ext.se3_scale_blocks(
        _check_cuda_float32("poses", poses),
        _check_cuda_float32("log_s", log_s),
        _check_cuda_float32("rel_poses", rel_poses),
        _check_cuda_float32("prior_log_s", prior_log_s),
        _check_cuda_int64("ii", ii),
        _check_cuda_int64("jj", jj),
        _check_cuda_float32("sqrt_info", sqrt_info),
        _check_cuda_float32("robust", robust),
        float(scale_prior_diag),
    )


def build_se3_scale_weighted_blocks(
    poses: torch.Tensor,
    log_s: torch.Tensor,
    rel_poses: torch.Tensor,
    prior_log_s: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    sqrt_info: torch.Tensor,
    huber_delta: float,
    scale_prior_diag: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    """Build Huber-weighted SE3+scale blocks and return current/unrobust costs."""
    ext = _extension()
    return ext.se3_scale_weighted_blocks(
        _check_cuda_float32("poses", poses),
        _check_cuda_float32("log_s", log_s),
        _check_cuda_float32("rel_poses", rel_poses),
        _check_cuda_float32("prior_log_s", prior_log_s),
        _check_cuda_int64("ii", ii),
        _check_cuda_int64("jj", jj),
        _check_cuda_float32("sqrt_info", sqrt_info),
        float(huber_delta),
        float(scale_prior_diag),
    )


def evaluate_se3_scale_candidate(
    poses: torch.Tensor,
    log_s: torch.Tensor,
    step_full: torch.Tensor,
    rel_poses: torch.Tensor,
    prior_log_s: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    sqrt_info: torch.Tensor,
    robust: torch.Tensor,
    anchor: int,
    scale_prior_diag: float,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """Apply one SE3+scale LM candidate step and return its weighted cost."""
    ext = _extension()
    return ext.se3_scale_candidate(
        _check_cuda_float32("poses", poses),
        _check_cuda_float32("log_s", log_s),
        _check_cuda_float32("step_full", step_full),
        _check_cuda_float32("rel_poses", rel_poses),
        _check_cuda_float32("prior_log_s", prior_log_s),
        _check_cuda_int64("ii", ii),
        _check_cuda_int64("jj", jj),
        _check_cuda_float32("sqrt_info", sqrt_info),
        _check_cuda_float32("robust", robust),
        int(anchor),
        float(scale_prior_diag),
    )


def evaluate_se3_scale_stats(
    poses: torch.Tensor,
    log_s: torch.Tensor,
    rel_poses: torch.Tensor,
    prior_log_s: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    sqrt_info: torch.Tensor,
    scale_prior_diag: float,
) -> tuple[float, float, float, float, float, bool]:
    """Return final SE3+scale weighted cost and residual summary statistics."""
    ext = _extension()
    return ext.se3_scale_stats(
        _check_cuda_float32("poses", poses),
        _check_cuda_float32("log_s", log_s),
        _check_cuda_float32("rel_poses", rel_poses),
        _check_cuda_float32("prior_log_s", prior_log_s),
        _check_cuda_int64("ii", ii),
        _check_cuda_int64("jj", jj),
        _check_cuda_float32("sqrt_info", sqrt_info),
        float(scale_prior_diag),
    )


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
        prior_gradient: torch.Tensor,
        damping: float,
        scale_prior_diag: float,
    ) -> torch.Tensor:
        step = self._solver.solve(
            _check_cuda_float32("source_block", source_block),
            _check_cuda_float32("target_block", target_block),
            _check_cuda_float32("edge_residual", edge_residual),
            _check_cuda_float32("prior_gradient", prior_gradient),
            float(damping),
            float(scale_prior_diag),
        )
        return step.view(self.n_nodes, 4)


class Se3ScaleEigenSimplicialLLTSolver:
    """Cached CPU Eigen LLT solver for one fixed SE3+scale graph."""

    def __init__(self, ii: torch.Tensor, jj: torch.Tensor, n_nodes: int, anchor: int):
        ext = _extension()
        self.n_nodes = int(n_nodes)
        self._solver = ext.Se3ScaleEigenSimplicialLLTSolver(
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
        prior_gradient: torch.Tensor,
        damping: float,
        scale_prior_diag: float,
    ) -> torch.Tensor:
        step = self._solver.solve(
            _check_cuda_float32("source_block", source_block),
            _check_cuda_float32("target_block", target_block),
            _check_cuda_float32("edge_residual", edge_residual),
            _check_cuda_float32("prior_gradient", prior_gradient),
            float(damping),
            float(scale_prior_diag),
        )
        return step.view(self.n_nodes, 7)

    def solve_lm_attempts(
        self,
        source_block: torch.Tensor,
        target_block: torch.Tensor,
        edge_residual: torch.Tensor,
        prior_gradient: torch.Tensor,
        poses: torch.Tensor,
        log_s: torch.Tensor,
        rel_poses: torch.Tensor,
        prior_log_s: torch.Tensor,
        sqrt_info: torch.Tensor,
        robust: torch.Tensor,
        current_cost: float,
        lm: float,
        lm_max_attempts: int,
        scale_prior_diag: float,
    ) -> tuple[torch.Tensor, torch.Tensor, float, float, float, bool, int, int]:
        return self._solver.solve_lm_attempts(
            _check_cuda_float32("source_block", source_block),
            _check_cuda_float32("target_block", target_block),
            _check_cuda_float32("edge_residual", edge_residual),
            _check_cuda_float32("prior_gradient", prior_gradient),
            _check_cuda_float32("poses", poses),
            _check_cuda_float32("log_s", log_s),
            _check_cuda_float32("rel_poses", rel_poses),
            _check_cuda_float32("prior_log_s", prior_log_s),
            _check_cuda_float32("sqrt_info", sqrt_info),
            _check_cuda_float32("robust", robust),
            float(current_cost),
            float(lm),
            int(lm_max_attempts),
            float(scale_prior_diag),
        )
