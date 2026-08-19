"""Scale-graph utilities for offline GeNT/MoGe scale-shape ablations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass(frozen=True)
class ScaleGraphFusionResult:
    x_geo: np.ndarray
    correction: np.ndarray
    x_fused: np.ndarray
    alpha: float
    geo_edge_rmse: float
    fused_edge_rmse: float
    geo_moge_shape_rmse: float
    fused_moge_shape_rmse: float


@dataclass(frozen=True)
class MogeAnchoredScaleGraphResult:
    x: np.ndarray
    edge_rmse: float
    moge_prior_residual_rmse: float
    effective_node_weights: np.ndarray
    beta: float
    rho: str


@dataclass(frozen=True)
class MogeModeNisCorrectionResult:
    x_fused: np.ndarray
    correction: np.ndarray
    alpha: float
    nis_score: float
    mode_nis: np.ndarray
    mode_variance: np.ndarray
    geo_edge_rmse: float
    fused_edge_rmse: float


def _as_vector(values, name: str, dtype=np.float64) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _incidence(n_nodes: int, ii: np.ndarray, jj: np.ndarray) -> sp.csr_matrix:
    rows = np.repeat(np.arange(ii.size, dtype=np.int64), 2)
    cols = np.empty(ii.size * 2, dtype=np.int64)
    cols[0::2] = ii
    cols[1::2] = jj
    data = np.tile(np.asarray([-1.0, 1.0], dtype=np.float64), ii.size)
    return sp.coo_matrix((data, (rows, cols)), shape=(ii.size, n_nodes)).tocsr()


def _solve_mean_constrained(H: sp.csr_matrix, rhs: np.ndarray, target_mean: float) -> np.ndarray:
    n = rhs.size
    ones = np.ones((n, 1), dtype=np.float64)
    kkt = sp.bmat(
        [
            [H.tocsr(), sp.csr_matrix(ones)],
            [sp.csr_matrix(ones.T), sp.csr_matrix((1, 1), dtype=np.float64)],
        ],
        format="csr",
    )
    aug_rhs = np.concatenate([rhs, np.asarray([float(target_mean) * n], dtype=np.float64)])
    sol = spla.spsolve(kkt, aug_rhs)
    if not np.all(np.isfinite(sol)):
        raise RuntimeError("scale graph solve returned non-finite values")
    return np.asarray(sol[:n], dtype=np.float64)


def _center(x: np.ndarray) -> np.ndarray:
    return x - float(x.mean())


def _edge_residual(x: np.ndarray, ii: np.ndarray, jj: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x[jj] - x[ii] - y


def _moge_edge_error(ii: np.ndarray, jj: np.ndarray, y: np.ndarray, m: np.ndarray) -> np.ndarray:
    return np.abs((m[jj] - m[ii]) - y)


def _rmse(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


def _second_difference(n_nodes: int) -> sp.csr_matrix:
    if n_nodes < 3:
        return sp.csr_matrix((0, n_nodes), dtype=np.float64)
    rows = np.repeat(np.arange(n_nodes - 2, dtype=np.int64), 3)
    cols = np.empty((n_nodes - 2) * 3, dtype=np.int64)
    base = np.arange(n_nodes - 2, dtype=np.int64)
    cols[0::3] = base
    cols[1::3] = base + 1
    cols[2::3] = base + 2
    data = np.tile(np.asarray([1.0, -2.0, 1.0], dtype=np.float64), n_nodes - 2)
    return sp.coo_matrix((data, (rows, cols)), shape=(n_nodes - 2, n_nodes)).tocsr()


def _dct_basis(n_nodes: int, n_modes: int) -> np.ndarray:
    samples = np.arange(int(n_nodes), dtype=np.float64) + 0.5
    modes = np.arange(int(n_modes), dtype=np.float64)[:, None]
    basis = np.sqrt(2.0 / float(n_nodes)) * np.cos(np.pi * modes * samples / float(n_nodes))
    basis[0] = 1.0 / np.sqrt(float(n_nodes))
    return basis


def solve_relative_scale_graph(
    n_nodes: int,
    ii,
    jj,
    edge_log_scales,
    *,
    edge_weights=None,
    target_mean: float = 0.0,
) -> np.ndarray:
    ii = _as_vector(ii, "ii", dtype=np.int64)
    jj = _as_vector(jj, "jj", dtype=np.int64)
    y = _as_vector(edge_log_scales, "edge_log_scales")
    if not (ii.shape == jj.shape == y.shape):
        raise ValueError("ii, jj, and edge_log_scales must have the same shape")
    if ii.size == 0:
        raise ValueError("scale graph must contain at least one edge")
    if ii.min() < 0 or jj.min() < 0 or ii.max() >= n_nodes or jj.max() >= n_nodes:
        raise ValueError("edge index is outside the node range")

    weights = np.ones(ii.size, dtype=np.float64) if edge_weights is None else _as_vector(edge_weights, "edge_weights")
    if weights.shape != y.shape:
        raise ValueError("edge_weights must have the same shape as edge_log_scales")
    if np.any(weights < 0.0):
        raise ValueError("edge_weights must be non-negative")

    B = _incidence(int(n_nodes), ii, jj)
    W = sp.diags(weights, format="csr")
    H = B.T @ W @ B
    rhs = B.T @ (weights * y)
    return _solve_mean_constrained(H, np.asarray(rhs).reshape(-1), target_mean)


def fuse_gent_scale_graph_with_moge_mode_nis(
    x_geo,
    moge_log_scales,
    ii,
    jj,
    edge_log_scales,
    *,
    beta: float = 10.0,
    n_modes: int = 8,
    nis_cutoff: float = 0.15,
) -> MogeModeNisCorrectionResult:
    """Apply low-frequency MoGe correction using edge-graph normalized innovation."""
    x_geo = _as_vector(x_geo, "x_geo")
    m = _as_vector(moge_log_scales, "moge_log_scales")
    ii = _as_vector(ii, "ii", dtype=np.int64)
    jj = _as_vector(jj, "jj", dtype=np.int64)
    y = _as_vector(edge_log_scales, "edge_log_scales")
    if x_geo.shape != m.shape:
        raise ValueError("x_geo and moge_log_scales must have the same shape")
    if not (ii.shape == jj.shape == y.shape):
        raise ValueError("ii, jj, and edge_log_scales must have the same shape")
    if not 2 <= int(n_modes) <= x_geo.size:
        raise ValueError("n_modes must be between 2 and the node count")
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    if nis_cutoff <= 0.0:
        raise ValueError("nis_cutoff must be positive")

    B = _incidence(x_geo.size, ii, jj)
    H = float(beta) * (B.T @ B)
    ones = np.ones((x_geo.size, 1), dtype=np.float64)
    kkt = sp.bmat(
        [
            [H.tocsr(), sp.csr_matrix(ones)],
            [sp.csr_matrix(ones.T), sp.csr_matrix((1, 1), dtype=np.float64)],
        ],
        format="csc",
    )
    solve = spla.factorized(kkt)

    basis = _dct_basis(x_geo.size, int(n_modes))
    innovation = _center(m) - _center(x_geo)
    coefficients = basis @ innovation
    mode_variance = np.empty(int(n_modes) - 1, dtype=np.float64)
    for mode in range(1, int(n_modes)):
        rhs = np.concatenate([basis[mode], np.zeros(1, dtype=np.float64)])
        response = solve(rhs)[: x_geo.size]
        mode_variance[mode - 1] = float(basis[mode] @ response)
    if np.any(mode_variance <= 0.0):
        raise RuntimeError("edge scale graph returned non-positive mode variance")

    mode_nis = coefficients[1:] ** 2 / mode_variance
    nis_score = float(np.median(mode_nis))
    alpha = float(nis_cutoff) / (float(nis_cutoff) + nis_score)
    correction = alpha * (basis.T @ coefficients)
    correction = _center(correction)
    x_fused = x_geo + correction
    return MogeModeNisCorrectionResult(
        x_fused=x_fused,
        correction=correction,
        alpha=alpha,
        nis_score=nis_score,
        mode_nis=mode_nis,
        mode_variance=mode_variance,
        geo_edge_rmse=_rmse(_edge_residual(x_geo, ii, jj, y)),
        fused_edge_rmse=_rmse(_edge_residual(x_fused, ii, jj, y)),
    )


def _solve_moge_anchored_l2(
    B: sp.csr_matrix,
    edge_log_scales: np.ndarray,
    moge_log_scales: np.ndarray,
    node_weights: np.ndarray,
    beta: float,
) -> np.ndarray:
    H = sp.diags(node_weights, format="csr") + float(beta) * (B.T @ B)
    rhs = node_weights * moge_log_scales + float(beta) * np.asarray(B.T @ edge_log_scales).reshape(-1)
    x = spla.spsolve(H.tocsr(), rhs)
    if not np.all(np.isfinite(x)):
        raise RuntimeError("MoGe-anchored scale graph solve returned non-finite values")
    return np.asarray(x, dtype=np.float64)


def _huber_weights(residual: np.ndarray, delta: float) -> np.ndarray:
    abs_residual = np.abs(residual)
    weights = np.ones_like(abs_residual, dtype=np.float64)
    outlier = abs_residual > float(delta)
    weights[outlier] = float(delta) / np.maximum(abs_residual[outlier], 1e-12)
    return weights


def solve_moge_anchored_scale_graph(
    n_nodes: int,
    ii,
    jj,
    edge_log_scales,
    moge_log_scales,
    *,
    node_weights=None,
    beta: float = 1.0,
    rho: str = "huber",
    huber_delta: float = 0.05,
    irls_iters: int = 5,
) -> MogeAnchoredScaleGraphResult:
    ii = _as_vector(ii, "ii", dtype=np.int64)
    jj = _as_vector(jj, "jj", dtype=np.int64)
    y = _as_vector(edge_log_scales, "edge_log_scales")
    m = _as_vector(moge_log_scales, "moge_log_scales")
    if m.size != int(n_nodes):
        raise ValueError("moge_log_scales must have one value per node")
    if not (ii.shape == jj.shape == y.shape):
        raise ValueError("ii, jj, and edge_log_scales must have the same shape")
    if ii.size == 0:
        raise ValueError("scale graph must contain at least one edge")
    if ii.min() < 0 or jj.min() < 0 or ii.max() >= n_nodes or jj.max() >= n_nodes:
        raise ValueError("edge index is outside the node range")
    if beta < 0.0:
        raise ValueError("beta must be non-negative")
    if rho not in {"l2", "huber"}:
        raise ValueError("rho must be 'l2' or 'huber'")

    weights = np.ones(int(n_nodes), dtype=np.float64) if node_weights is None else _as_vector(node_weights, "node_weights")
    if weights.shape != m.shape:
        raise ValueError("node_weights must have one value per node")
    if np.any(weights < 0.0):
        raise ValueError("node_weights must be non-negative")

    B = _incidence(int(n_nodes), ii, jj)
    x = _solve_moge_anchored_l2(B, y, m, weights, beta)
    robust_weights = np.ones_like(weights, dtype=np.float64)
    if rho == "huber":
        for _ in range(int(irls_iters)):
            robust_weights = _huber_weights(x - m, float(huber_delta))
            x = _solve_moge_anchored_l2(B, y, m, weights * robust_weights, beta)
        robust_weights = _huber_weights(x - m, float(huber_delta))

    return MogeAnchoredScaleGraphResult(
        x=x,
        edge_rmse=_rmse(_edge_residual(x, ii, jj, y)),
        moge_prior_residual_rmse=_rmse(x - m),
        effective_node_weights=weights * robust_weights,
        beta=float(beta),
        rho=rho,
    )


def moge_consistency_weights(
    n_nodes: int,
    ii,
    jj,
    edge_log_scales,
    moge_log_scales,
    *,
    min_weight: float = 0.0,
) -> np.ndarray:
    ii = _as_vector(ii, "ii", dtype=np.int64)
    jj = _as_vector(jj, "jj", dtype=np.int64)
    y = _as_vector(edge_log_scales, "edge_log_scales")
    m = _as_vector(moge_log_scales, "moge_log_scales")
    if m.size != int(n_nodes):
        raise ValueError("moge_log_scales must have one value per node")

    edge_error = _moge_edge_error(ii, jj, y, m)
    incident_errors: list[list[float]] = [[] for _ in range(int(n_nodes))]
    for i, j, err in zip(ii.tolist(), jj.tolist(), edge_error.tolist(), strict=True):
        incident_errors[int(i)].append(float(err))
        incident_errors[int(j)].append(float(err))

    scene_score = float(np.median(edge_error))
    node_score = np.asarray(
        [
            float(np.mean(errors)) if errors else scene_score
            for errors in incident_errors
        ],
        dtype=np.float64,
    )
    scene_sigma = max(0.48 * float(np.median(node_score)), 1e-8)
    node_weight = 1.0 / (1.0 + (node_score / scene_sigma) ** 2)
    return np.clip(node_weight, float(min_weight), 1.0)


def moge_innovation_weights(
    x_prediction,
    moge_log_scales,
    *,
    sigma: float = 0.05,
    min_weight: float = 0.0,
) -> np.ndarray:
    x = _as_vector(x_prediction, "x_prediction")
    m = _as_vector(moge_log_scales, "moge_log_scales")
    if x.shape != m.shape:
        raise ValueError("x_prediction and moge_log_scales must have the same shape")
    residual = _center(m) - _center(x)
    node_weight = 1.0 / (1.0 + (residual / max(float(sigma), 1e-8)) ** 2)
    return np.clip(node_weight, float(min_weight), 1.0)


def fuse_gent_scale_graph_with_moge(
    *,
    x_geo,
    moge_log_scales,
    ii,
    jj,
    edge_log_scales,
    node_weights=None,
    lambda_smooth: float = 80.0,
    eta_edge: float = 0.01,
    edge_rmse_increase: float = 0.05,
    min_edge_rmse_budget: float = 0.01,
    alpha_grid_steps: int = 200,
) -> ScaleGraphFusionResult:
    x_geo = _as_vector(x_geo, "x_geo")
    m = _as_vector(moge_log_scales, "moge_log_scales")
    ii = _as_vector(ii, "ii", dtype=np.int64)
    jj = _as_vector(jj, "jj", dtype=np.int64)
    y = _as_vector(edge_log_scales, "edge_log_scales")
    if m.shape != x_geo.shape:
        raise ValueError("x_geo and moge_log_scales must have the same shape")
    if not (ii.shape == jj.shape == y.shape):
        raise ValueError("ii, jj, and edge_log_scales must have the same shape")

    n = x_geo.size
    weights = np.ones(n, dtype=np.float64) if node_weights is None else _as_vector(node_weights, "node_weights")
    if weights.shape != x_geo.shape:
        raise ValueError("node_weights must have one value per node")
    if np.any(weights < 0.0):
        raise ValueError("node_weights must be non-negative")

    B = _incidence(n, ii, jj)
    D2 = _second_difference(n)
    q = _center(m) - _center(x_geo)

    H = sp.diags(weights, format="csr")
    rhs = weights * q
    if lambda_smooth > 0.0 and D2.shape[0] > 0:
        H = H + float(lambda_smooth) * (D2.T @ D2)
    if eta_edge > 0.0:
        H = H + float(eta_edge) * (B.T @ B)

    delta = _solve_mean_constrained(H, rhs, 0.0)
    geo_edge_res = _edge_residual(x_geo, ii, jj, y)
    geo_edge_rmse = _rmse(geo_edge_res)
    allowed_edge_rmse = max(
        geo_edge_rmse * (1.0 + float(edge_rmse_increase)),
        float(min_edge_rmse_budget),
    )
    geo_moge_rmse = _rmse(_center(x_geo) - _center(m))

    best_alpha = 0.0
    best_moge_rmse = geo_moge_rmse
    steps = max(int(alpha_grid_steps), 1)
    for alpha in np.linspace(1.0, 0.0, steps + 1):
        candidate = x_geo + alpha * delta
        edge_rmse = _rmse(_edge_residual(candidate, ii, jj, y))
        moge_rmse = _rmse(_center(candidate) - _center(m))
        if edge_rmse <= allowed_edge_rmse + 1e-12 and moge_rmse <= geo_moge_rmse + 1e-12:
            best_alpha = float(alpha)
            best_moge_rmse = float(moge_rmse)
            break

    x_fused = x_geo + best_alpha * delta
    x_fused = x_fused - float(x_fused.mean()) + float(x_geo.mean())
    return ScaleGraphFusionResult(
        x_geo=x_geo,
        correction=best_alpha * delta,
        x_fused=x_fused,
        alpha=best_alpha,
        geo_edge_rmse=geo_edge_rmse,
        fused_edge_rmse=_rmse(_edge_residual(x_fused, ii, jj, y)),
        geo_moge_shape_rmse=geo_moge_rmse,
        fused_moge_shape_rmse=best_moge_rmse,
    )
