from __future__ import annotations

import torch
from lietorch import SO3, SE3

from .common import (
    EPS,
    LM_DAMPING_DECREASE,
    LM_DAMPING_INCREASE,
    LM_DAMPING_MAX,
    LM_DAMPING_MIN,
    RelativeEdges,
    Sim3PGOResult,
    _apply_delta,
    _apply_rotation_delta,
    _apply_translation_scale_delta,
    _dct_scale_modes,
    _edge_sqrt_information,
    _mode_quadratic_cost,
    _moge_mode_prior_covariance,
    _regularize_mode_covariance,
    _relative_scale_residuals,
    _rotation_residuals,
    _rotation_sqrt_information,
    _scale_sqrt_information,
    _scaled_se3_residuals_cached_rotation,
    _so3_hat,
    _so3_right_jacobian_inverse,
    _translation_residuals,
    _translation_sqrt_information,
)


def _solve_lm_step(A: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Solve the damped normal equation A step = -g with Cholesky."""
    chol, info = torch.linalg.cholesky_ex(A)
    if bool((info != 0).any()):
        raise RuntimeError("Cholesky factorization failed")
    return -torch.cholesky_solve(g[:, None], chol).squeeze(-1)


def _pgo_info(
    *,
    success: bool,
    mode: str,
    n_edges: int,
    n_iters: int,
    accepted_iters: int,
    cost: float,
) -> dict:
    return {
        "success": bool(success),
        "mode": mode,
        "backend": "torch",
        "n_edges": int(n_edges),
        "n_iters": int(n_iters),
        "accepted_iters": int(accepted_iters),
        "cost": float(cost),
    }


def _constrain_anchor_scale(
    H: torch.Tensor,
    g: torch.Tensor,
    anchor_scale_col: int,
) -> None:
    H[anchor_scale_col, :] = 0
    H[:, anchor_scale_col] = 0
    H[anchor_scale_col, anchor_scale_col] = 1
    g[anchor_scale_col] = 0


def _rotation_linear_system(
    rotations: SO3,
    ii: torch.Tensor,
    jj: torch.Tensor,
    meas_rotations: SO3,
    anchor: int,
    sqrt_info: torch.Tensor,
    robust: torch.Tensor,
):
    """Assemble rotation-only normal equations without autograd or dense J.

    Unknowns are ordered by non-anchor node id: dphi(3) for each node.
    Returns H: (A * 3, A * 3), g: (A * 3,), and weighted residual r.

    The optimizer applies right perturbations R_k <- R_k * Exp(dphi_k).
    With E_ij = R_ij_meas^{-1} * R_j * R_i^{-1}, the first-order residual is:

        Log(E_ij * Exp( R_i * dphi_j - R_i * dphi_i ))

    so each edge contributes +/- J_r^{-1}(r_ij) R_i blocks.
    """
    rotation_data = rotations.data
    device = rotation_data.device
    dtype = rotation_data.dtype
    n_nodes = rotation_data.shape[0]
    n_edges = ii.numel()
    n_active = n_nodes - 1
    n_vars = n_active * 3

    optimized_nodes = torch.arange(n_nodes, device=device)
    optimized_nodes = optimized_nodes[optimized_nodes != anchor]
    node_vars = torch.full((n_nodes,), -1, device=device, dtype=torch.long)
    node_vars[optimized_nodes] = torch.arange(n_active, device=device)

    residual = _rotation_residuals(rotations, ii, jj, meas_rotations)
    row_weight = sqrt_info * robust
    r = (residual * row_weight).reshape(-1)

    source_vars = node_vars[ii]
    target_vars = node_vars[jj]
    block_cols = torch.arange(3, device=device)

    source_rot = SO3(rotation_data[ii]).matrix()[..., :3, :3]
    right_jac_inv = _so3_right_jacobian_inverse(residual)
    edge_block = row_weight[:, :, None] * (right_jac_inv @ source_rot)
    source_block = -edge_block
    target_block = edge_block

    H = torch.zeros(n_vars, n_vars, device=device, dtype=dtype)
    g = torch.zeros(n_vars, device=device, dtype=dtype)
    r_edges = r.reshape(n_edges, 3)

    def accumulate_gradient(var_ids: torch.Tensor, block: torch.Tensor):
        mask = var_ids >= 0
        cols = var_ids[mask][:, None] * 3 + block_cols[None, :]
        values = torch.einsum("era,er->ea", block[mask], r_edges[mask])
        g.index_add_(0, cols.reshape(-1), values.reshape(-1))

    def accumulate_hessian(var_a: torch.Tensor, block_a: torch.Tensor, var_b: torch.Tensor, block_b: torch.Tensor):
        mask = (var_a >= 0) & (var_b >= 0)
        rows = var_a[mask][:, None, None] * 3 + block_cols[None, :, None]
        cols = var_b[mask][:, None, None] * 3 + block_cols[None, None, :]
        values = torch.einsum("era,erb->eab", block_a[mask], block_b[mask])
        H.reshape(-1).index_add_(0, (rows * n_vars + cols).reshape(-1), values.reshape(-1))

    accumulate_gradient(source_vars, source_block)
    accumulate_gradient(target_vars, target_block)
    accumulate_hessian(source_vars, source_block, source_vars, source_block)
    accumulate_hessian(source_vars, source_block, target_vars, target_block)
    accumulate_hessian(target_vars, target_block, source_vars, source_block)
    accumulate_hessian(target_vars, target_block, target_vars, target_block)
    return H, g, r


def _translation_scale_linear_system(
    poses: SE3,
    log_s: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    rel_edges: RelativeEdges,
    anchor: int,
    sqrt_info: torch.Tensor,
    robust: torch.Tensor,
    scale_sqrt_info: torch.Tensor,
):
    """Assemble translation/scale normal equations without autograd.

    Unknowns are ordered as translation increments for non-anchor nodes followed
    by scale increments for all nodes:

        [dt_non_anchor(A, 3), dlog_s_all_nodes(N)]

    Edge Jacobian blocks are accumulated directly into H/g on GPU rather than
    materializing a dense J. Returns H: (A * 3 + N, A * 3 + N), g:
    (A * 3 + N,), and the weighted residual r.
    """
    pose_data = poses.data
    device = pose_data.device
    dtype = pose_data.dtype
    n_nodes = pose_data.shape[0]
    n_edges = ii.numel()
    n_active = n_nodes - 1
    n_translation_vars = n_active * 3
    scale_offset = n_translation_vars
    n_vars = n_translation_vars + n_nodes

    optimized_nodes = torch.arange(n_nodes, device=device)
    optimized_nodes = optimized_nodes[optimized_nodes != anchor]
    translation_vars = torch.full((n_nodes,), -1, device=device, dtype=torch.long)
    translation_vars[optimized_nodes] = torch.arange(n_active, device=device)
    scale_vars = scale_offset + torch.arange(n_nodes, device=device)

    pred = (poses[jj] * poses[ii].inv()).data[..., :3]
    inv_source_scale = torch.exp(log_s[ii]).clamp_min(EPS).reciprocal()
    pred_t = pred * inv_source_scale[:, None]
    residual = pred_t - rel_edges.poses.data[..., :3]
    row_weight = sqrt_info * robust

    r_edge = (residual * row_weight).reshape(-1)
    r_scale = _relative_scale_residuals(log_s, ii, jj, rel_edges) * scale_sqrt_info
    r = torch.cat((r_edge, r_scale), dim=0)

    H = torch.zeros(n_vars, n_vars, device=device, dtype=dtype)
    g = torch.zeros(n_vars, device=device, dtype=dtype)
    translation_block_cols = torch.arange(3, device=device)
    source_translation_vars = translation_vars[ii]
    target_translation_vars = translation_vars[jj]
    source_scale_cols = scale_vars[ii][:, None]
    source_scale_vec = scale_vars[ii]
    target_scale_vec = scale_vars[jj]

    rot_j = SO3(pose_data[jj, 3:7]).matrix()[..., :3, :3]
    translation_block = row_weight[:, :, None] * rot_j * inv_source_scale[:, None, None]

    source_translation_cols = source_translation_vars[:, None] * 3 + translation_block_cols[None, :]
    target_translation_cols = target_translation_vars[:, None] * 3 + translation_block_cols[None, :]
    source_translation_block = -translation_block
    target_translation_block = translation_block
    source_scale_block = (-pred_t * row_weight)[:, :, None]
    edge_rows = r_edge.reshape(n_edges, 3)
    source_translation_valid = source_translation_vars >= 0
    target_translation_valid = target_translation_vars >= 0

    def accumulate_gradient(cols: torch.Tensor, block: torch.Tensor, valid: torch.Tensor):
        cols = cols[valid]
        block = block[valid]
        rows = edge_rows[valid]
        values = torch.einsum("era,er->ea", block, rows)
        g.index_add_(0, cols.reshape(-1), values.reshape(-1))

    def accumulate_hessian(
        cols_a: torch.Tensor,
        block_a: torch.Tensor,
        cols_b: torch.Tensor,
        block_b: torch.Tensor,
        valid: torch.Tensor,
    ):
        rows = cols_a[valid][:, :, None]
        cols = cols_b[valid][:, None, :]
        values = torch.einsum("era,erb->eab", block_a[valid], block_b[valid])
        H.reshape(-1).index_add_(0, (rows * n_vars + cols).reshape(-1), values.reshape(-1))

    all_edges = torch.ones(n_edges, device=device, dtype=torch.bool)
    accumulate_gradient(source_translation_cols, source_translation_block, source_translation_valid)
    accumulate_gradient(target_translation_cols, target_translation_block, target_translation_valid)
    accumulate_gradient(source_scale_cols, source_scale_block, all_edges)
    accumulate_hessian(
        source_translation_cols,
        source_translation_block,
        source_translation_cols,
        source_translation_block,
        source_translation_valid,
    )
    accumulate_hessian(
        source_translation_cols,
        source_translation_block,
        target_translation_cols,
        target_translation_block,
        source_translation_valid & target_translation_valid,
    )
    accumulate_hessian(
        target_translation_cols,
        target_translation_block,
        source_translation_cols,
        source_translation_block,
        source_translation_valid & target_translation_valid,
    )
    accumulate_hessian(
        target_translation_cols,
        target_translation_block,
        target_translation_cols,
        target_translation_block,
        target_translation_valid,
    )
    accumulate_hessian(
        source_translation_cols,
        source_translation_block,
        source_scale_cols,
        source_scale_block,
        source_translation_valid,
    )
    accumulate_hessian(
        target_translation_cols,
        target_translation_block,
        source_scale_cols,
        source_scale_block,
        target_translation_valid,
    )
    accumulate_hessian(
        source_scale_cols,
        source_scale_block,
        source_translation_cols,
        source_translation_block,
        source_translation_valid,
    )
    accumulate_hessian(
        source_scale_cols,
        source_scale_block,
        target_translation_cols,
        target_translation_block,
        target_translation_valid,
    )
    accumulate_hessian(source_scale_cols, source_scale_block, source_scale_cols, source_scale_block, all_edges)

    scale_info = scale_sqrt_info * scale_sqrt_info
    g.index_add_(0, source_scale_vec, -scale_sqrt_info * r_scale)
    g.index_add_(0, target_scale_vec, scale_sqrt_info * r_scale)
    flat = H.reshape(-1)
    flat.index_add_(0, source_scale_vec * n_vars + source_scale_vec, scale_info)
    flat.index_add_(0, target_scale_vec * n_vars + target_scale_vec, scale_info)
    flat.index_add_(0, source_scale_vec * n_vars + target_scale_vec, -scale_info)
    flat.index_add_(0, target_scale_vec * n_vars + source_scale_vec, -scale_info)
    _constrain_anchor_scale(H, g, scale_offset + int(anchor))
    return H, g, r


def _se3_scale_linear_system(
    poses: SE3,
    log_s: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    rel_edges: RelativeEdges,
    anchor: int,
    sqrt_info: torch.Tensor,
    robust: torch.Tensor,
    scale_sqrt_info: torch.Tensor,
):
    """Assemble SE3+scale normal equations without autograd.

    Unknowns are ordered as pose increments for non-anchor nodes followed by
    scale increments for all nodes:

        [dpose_non_anchor(A, 6), dlog_s_all_nodes(N)]

    The right pose perturbation convention is:

        T_k <- T_k * Exp([dt_k, dphi_k])

    Returns H: (A * 6 + N, A * 6 + N), g: (A * 6 + N,), and weighted
    residual r.
    """
    pose_data = poses.data
    device = pose_data.device
    dtype = pose_data.dtype
    n_nodes = pose_data.shape[0]
    n_edges = ii.numel()
    n_active = n_nodes - 1
    n_pose_vars = n_active * 6
    scale_offset = n_pose_vars
    n_vars = n_pose_vars + n_nodes

    optimized_nodes = torch.arange(n_nodes, device=device)
    optimized_nodes = optimized_nodes[optimized_nodes != anchor]
    pose_vars = torch.full((n_nodes,), -1, device=device, dtype=torch.long)
    pose_vars[optimized_nodes] = torch.arange(n_active, device=device)
    scale_vars = scale_offset + torch.arange(n_nodes, device=device)

    pred = (poses[jj] * poses[ii].inv()).data
    pred_t_metric = pred[..., :3]
    inv_source_scale = torch.exp(log_s[ii]).clamp_min(EPS).reciprocal()
    pred_t = pred_t_metric * inv_source_scale[:, None]
    pred_q = pred[..., 3:7]
    rot_residual = (SO3(rel_edges.poses.data[..., 3:7]).inv() * SO3(pred_q)).log()
    residual = torch.cat((pred_t - rel_edges.poses.data[..., :3], rot_residual), dim=-1)
    row_weight = sqrt_info * robust

    r_edge = (residual * row_weight).reshape(-1)
    r_scale = _relative_scale_residuals(log_s, ii, jj, rel_edges) * scale_sqrt_info
    r = torch.cat((r_edge, r_scale), dim=0)

    H = torch.zeros(n_vars, n_vars, device=device, dtype=dtype)
    g = torch.zeros(n_vars, device=device, dtype=dtype)
    pose_block_cols = torch.arange(6, device=device)
    source_pose_vars = pose_vars[ii]
    target_pose_vars = pose_vars[jj]
    source_pose_cols = source_pose_vars[:, None] * 6 + pose_block_cols[None, :]
    target_pose_cols = target_pose_vars[:, None] * 6 + pose_block_cols[None, :]
    source_scale_cols = scale_vars[ii][:, None]
    source_scale_vec = scale_vars[ii]
    target_scale_vec = scale_vars[jj]

    rot_i = SO3(pose_data[ii, 3:7]).matrix()[..., :3, :3]
    rot_j = SO3(pose_data[jj, 3:7]).matrix()[..., :3, :3]
    source_t_in_source = torch.einsum("eji,ej->ei", rot_i, pose_data[ii, :3])
    trans_rot_block = rot_j @ _so3_hat(source_t_in_source)
    translation_block = rot_j * inv_source_scale[:, None, None]

    right_jac_inv = _so3_right_jacobian_inverse(rot_residual)
    rotation_block = right_jac_inv @ rot_i

    source_pose_block = torch.zeros(n_edges, 6, 6, device=device, dtype=dtype)
    target_pose_block = torch.zeros(n_edges, 6, 6, device=device, dtype=dtype)
    source_pose_block[:, :3, :3] = -translation_block
    source_pose_block[:, :3, 3:6] = -trans_rot_block * inv_source_scale[:, None, None]
    source_pose_block[:, 3:6, 3:6] = -rotation_block
    target_pose_block[:, :3, :3] = translation_block
    target_pose_block[:, :3, 3:6] = trans_rot_block * inv_source_scale[:, None, None]
    target_pose_block[:, 3:6, 3:6] = rotation_block
    source_scale_block = torch.zeros(n_edges, 6, 1, device=device, dtype=dtype)
    source_scale_block[:, :3, 0] = -pred_t

    source_pose_block = row_weight[:, :, None] * source_pose_block
    target_pose_block = row_weight[:, :, None] * target_pose_block
    source_scale_block = row_weight[:, :, None] * source_scale_block
    edge_rows = r_edge.reshape(n_edges, 6)
    source_pose_valid = source_pose_vars >= 0
    target_pose_valid = target_pose_vars >= 0

    def accumulate_gradient(cols: torch.Tensor, block: torch.Tensor, valid: torch.Tensor):
        cols = cols[valid]
        block = block[valid]
        rows = edge_rows[valid]
        values = torch.einsum("era,er->ea", block, rows)
        g.index_add_(0, cols.reshape(-1), values.reshape(-1))

    def accumulate_hessian(
        cols_a: torch.Tensor,
        block_a: torch.Tensor,
        cols_b: torch.Tensor,
        block_b: torch.Tensor,
        valid: torch.Tensor,
    ):
        rows = cols_a[valid][:, :, None]
        cols = cols_b[valid][:, None, :]
        values = torch.einsum("era,erb->eab", block_a[valid], block_b[valid])
        H.reshape(-1).index_add_(0, (rows * n_vars + cols).reshape(-1), values.reshape(-1))

    all_edges = torch.ones(n_edges, device=device, dtype=torch.bool)
    accumulate_gradient(source_pose_cols, source_pose_block, source_pose_valid)
    accumulate_gradient(target_pose_cols, target_pose_block, target_pose_valid)
    accumulate_gradient(source_scale_cols, source_scale_block, all_edges)
    accumulate_hessian(source_pose_cols, source_pose_block, source_pose_cols, source_pose_block, source_pose_valid)
    accumulate_hessian(
        source_pose_cols,
        source_pose_block,
        target_pose_cols,
        target_pose_block,
        source_pose_valid & target_pose_valid,
    )
    accumulate_hessian(
        target_pose_cols,
        target_pose_block,
        source_pose_cols,
        source_pose_block,
        source_pose_valid & target_pose_valid,
    )
    accumulate_hessian(target_pose_cols, target_pose_block, target_pose_cols, target_pose_block, target_pose_valid)
    accumulate_hessian(source_pose_cols, source_pose_block, source_scale_cols, source_scale_block, source_pose_valid)
    accumulate_hessian(target_pose_cols, target_pose_block, source_scale_cols, source_scale_block, target_pose_valid)
    accumulate_hessian(source_scale_cols, source_scale_block, source_pose_cols, source_pose_block, source_pose_valid)
    accumulate_hessian(source_scale_cols, source_scale_block, target_pose_cols, target_pose_block, target_pose_valid)
    accumulate_hessian(source_scale_cols, source_scale_block, source_scale_cols, source_scale_block, all_edges)

    scale_info = scale_sqrt_info * scale_sqrt_info
    g.index_add_(0, source_scale_vec, -scale_sqrt_info * r_scale)
    g.index_add_(0, target_scale_vec, scale_sqrt_info * r_scale)
    flat = H.reshape(-1)
    flat.index_add_(0, source_scale_vec * n_vars + source_scale_vec, scale_info)
    flat.index_add_(0, target_scale_vec * n_vars + target_scale_vec, scale_info)
    flat.index_add_(0, source_scale_vec * n_vars + target_scale_vec, -scale_info)
    flat.index_add_(0, target_scale_vec * n_vars + source_scale_vec, -scale_info)
    _constrain_anchor_scale(H, g, scale_offset + int(anchor))
    return H, g, r


def _optimize_rotation_pose_graph(
    poses: SE3,
    log_s: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    meas_rotations: SO3,
    edge_conf: torch.Tensor,
    anchor: int,
    n_iters: int,
    damping: float,
    lm_max_attempts: int,
    huber_delta: float,
) -> Sim3PGOResult:
    """Rotation stage.

    poses.data: (N, 7), ii/jj: (E,), meas_rotations.data: (E, 4),
    edge_conf: (E, 2). The optimized variable is delta (A, 3).
    """
    pose_data = poses.data
    device = pose_data.device
    dtype = pose_data.dtype
    n_nodes = pose_data.shape[0]
    n_vars = 3 * (n_nodes - 1)
    sqrt_info = _rotation_sqrt_information(edge_conf)
    rotations = SO3(poses.data[..., 3:7])

    if n_vars == 0:
        return Sim3PGOResult(
            poses=poses.data,
            log_scales=log_s,
            info=_pgo_info(
                success=True,
                mode="rotation_only",
                n_edges=int(ii.numel()),
                n_iters=0,
                accepted_iters=0,
                cost=0.0,
            ),
        )

    initial_cost = None
    accepted = 0
    lm = damping
    eye = torch.eye(n_vars, device=device, dtype=dtype)

    for _ in range(n_iters):
        with torch.no_grad():
            r0 = _rotation_residuals(rotations, ii, jj, meas_rotations) * sqrt_info
            edge_norm = r0.norm(dim=-1).clamp_min(EPS)
            robust = torch.where(edge_norm <= huber_delta, torch.ones_like(edge_norm), huber_delta / edge_norm)
            robust = robust.sqrt().unsqueeze(-1)
            weighted_r0 = r0 * robust
            current_cost = 0.5 * (weighted_r0 ** 2).sum()
            if initial_cost is None:
                initial_cost = float((0.5 * (r0 ** 2).sum()).cpu())
            H, g, _ = _rotation_linear_system(
                rotations,
                ii,
                jj,
                meas_rotations,
                anchor,
                sqrt_info,
                robust,
            )

        improved = False
        step_norm = 0.0
        for _ in range(lm_max_attempts):
            A = H + lm * eye
            try:
                step = _solve_lm_step(A, g).to(dtype=dtype)
            except RuntimeError:
                lm = min(lm * LM_DAMPING_INCREASE, LM_DAMPING_MAX)
                continue
            delta = step.reshape(n_nodes - 1, 3)
            cand_rotations = _apply_rotation_delta(rotations, delta, anchor)
            cand_r = _rotation_residuals(cand_rotations, ii, jj, meas_rotations)
            cand_weighted = cand_r * sqrt_info * robust
            cand_cost = 0.5 * (cand_weighted ** 2).sum()
            step_norm = float(step.norm().cpu())
            if cand_cost <= current_cost:
                rotations = cand_rotations
                lm = max(lm * LM_DAMPING_DECREASE, LM_DAMPING_MIN)
                accepted += 1
                improved = True
                break
            lm = min(lm * LM_DAMPING_INCREASE, LM_DAMPING_MAX)

        if not improved:
            break

        if step_norm < 1e-5:
            break

    poses = SE3(torch.cat((pose_data[:, :3], rotations.data), dim=-1))
    final_res = _rotation_residuals(rotations, ii, jj, meas_rotations) * sqrt_info
    final_cost = float((0.5 * (final_res ** 2).sum()).cpu())
    finite = bool(torch.isfinite(final_res).all().cpu())
    start_cost = final_cost if initial_cost is None else initial_cost
    info = _pgo_info(
        success=finite and (n_iters == 0 or accepted > 0 or final_cost <= start_cost + 1e-6),
        mode="rotation_only",
        n_edges=int(ii.numel()),
        n_iters=int(n_iters),
        accepted_iters=accepted,
        cost=final_cost,
    )
    return Sim3PGOResult(poses=poses.data, log_scales=log_s, info=info)


def _optimize_translation_scale_pose_graph(
    poses: SE3,
    log_s: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    rel_edges: RelativeEdges,
    edge_conf: torch.Tensor,
    anchor: int,
    n_iters: int,
    damping: float,
    lm_max_attempts: int,
    huber_delta: float,
) -> Sim3PGOResult:
    """Translation/scale stage.

    poses.data: (N, 7), log_s: (N,), ii/jj: (E,),
    rel_edges.poses.data: (E, 7), edge_conf: (E, 3). The optimized variable
    is [dt_non_anchor(A, 3), dlog_s_all_nodes(N)], and the residual vector has
    E * 4 entries.
    """
    device = poses.data.device
    dtype = poses.data.dtype
    n_nodes = poses.data.shape[0]
    n_translation_vars = 3 * (n_nodes - 1)
    n_vars = n_translation_vars + n_nodes
    sqrt_info = _translation_sqrt_information(edge_conf)
    scale_sqrt_info = _scale_sqrt_information(edge_conf)

    if n_vars == 0:
        return Sim3PGOResult(
            poses=poses.data,
            log_scales=log_s,
            info=_pgo_info(
                success=True,
                mode="translation_scale",
                n_edges=int(ii.numel()),
                n_iters=0,
                accepted_iters=0,
                cost=0.0,
            ),
        )

    initial_cost = None
    accepted = 0
    lm = damping
    eye = torch.eye(n_vars, device=device, dtype=dtype)

    for _ in range(n_iters):
        with torch.no_grad():
            r0 = _translation_residuals(poses, log_s, ii, jj, rel_edges) * sqrt_info
            edge_norm = r0.norm(dim=-1).clamp_min(EPS)
            robust = torch.where(edge_norm <= huber_delta, torch.ones_like(edge_norm), huber_delta / edge_norm)
            robust = robust.sqrt().unsqueeze(-1)
            weighted_r0 = r0 * robust
            s0 = _relative_scale_residuals(log_s, ii, jj, rel_edges) * scale_sqrt_info
            current_cost = 0.5 * ((weighted_r0 ** 2).sum() + (s0 ** 2).sum())
            if initial_cost is None:
                initial_cost = float((0.5 * ((r0 ** 2).sum() + (s0 ** 2).sum())).cpu())
            H, g, _ = _translation_scale_linear_system(
                poses,
                log_s,
                ii,
                jj,
                rel_edges,
                anchor,
                sqrt_info,
                robust,
                scale_sqrt_info,
            )

        improved = False
        step_norm = 0.0
        for _ in range(lm_max_attempts):
            A = H + lm * eye
            try:
                step = _solve_lm_step(A, g).to(dtype=dtype)
            except RuntimeError:
                lm = min(lm * LM_DAMPING_INCREASE, LM_DAMPING_MAX)
                continue
            translation_delta = step[:n_translation_vars].reshape(n_nodes - 1, 3)
            scale_delta = step[n_translation_vars : n_translation_vars + n_nodes]
            cand_poses, cand_log_s = _apply_translation_scale_delta(
                poses,
                log_s,
                translation_delta,
                scale_delta,
                anchor,
            )
            cand_r = _translation_residuals(cand_poses, cand_log_s, ii, jj, rel_edges)
            cand_s = _relative_scale_residuals(cand_log_s, ii, jj, rel_edges) * scale_sqrt_info
            cand_weighted = cand_r * sqrt_info * robust
            cand_cost = 0.5 * ((cand_weighted ** 2).sum() + (cand_s ** 2).sum())
            step_norm = float(step.norm().cpu())
            if cand_cost <= current_cost:
                poses, log_s = cand_poses, cand_log_s
                lm = max(lm * LM_DAMPING_DECREASE, LM_DAMPING_MIN)
                accepted += 1
                improved = True
                break
            lm = min(lm * LM_DAMPING_INCREASE, LM_DAMPING_MAX)

        if not improved:
            break

        if step_norm < 1e-5:
            break

    final_res = _translation_residuals(poses, log_s, ii, jj, rel_edges) * sqrt_info
    final_scale = _relative_scale_residuals(log_s, ii, jj, rel_edges) * scale_sqrt_info
    final_cost = float((0.5 * ((final_res ** 2).sum() + (final_scale ** 2).sum())).cpu())
    finite = bool((torch.isfinite(final_res).all() & torch.isfinite(final_scale).all()).cpu())
    start_cost = final_cost if initial_cost is None else initial_cost
    info = _pgo_info(
        success=finite and (n_iters == 0 or accepted > 0 or final_cost <= start_cost + 1e-6),
        mode="translation_scale",
        n_edges=int(ii.numel()),
        n_iters=int(n_iters),
        accepted_iters=accepted,
        cost=final_cost,
    )
    return Sim3PGOResult(poses=poses.data, log_scales=log_s, info=info)


def _optimize_se3_scale_pose_graph(
    poses: SE3,
    log_s: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    rel_edges: RelativeEdges,
    edge_conf: torch.Tensor,
    anchor: int,
    n_iters: int,
    damping: float,
    lm_max_attempts: int,
    huber_delta: float,
    moge_log_s: torch.Tensor,
    moge_mode_nis: bool,
    moge_mode_count: int,
    moge_mode_nis_cutoff: float,
) -> Sim3PGOResult:
    """Joint SE3+scale PGO with a low-rank MoGe mode factor."""
    device = poses.data.device
    dtype = poses.data.dtype
    n_nodes = poses.data.shape[0]
    n_pose_vars = 6 * (n_nodes - 1)
    n_vars = n_pose_vars + n_nodes
    sqrt_info = _edge_sqrt_information(edge_conf)
    scale_sqrt_info = _scale_sqrt_information(edge_conf)

    if n_vars == 0:
        return Sim3PGOResult(
            poses=poses.data,
            log_scales=log_s,
            info=_pgo_info(
                success=True,
                mode="se3_scale",
                n_edges=int(ii.numel()),
                n_iters=0,
                accepted_iters=0,
                cost=0.0,
            ),
        )

    initial_cost = None
    accepted = 0
    lm = damping
    eye = torch.eye(n_vars, device=device, dtype=dtype)
    node_ids = torch.arange(n_nodes, device=device)
    optimized_nodes = node_ids[node_ids != anchor]
    meas_rotations_inv = SO3(rel_edges.poses.data[..., 3:7]).inv()
    mode_basis = _dct_scale_modes(n_nodes, moge_mode_count, log_s) if moge_mode_nis else None
    mode_identity = (
        torch.eye(mode_basis.shape[0], device=device, dtype=dtype)
        if mode_basis is not None
        else None
    )
    mode_residual = mode_basis @ (log_s - moge_log_s) if mode_basis is not None else None
    mode_prior_covariance = None
    mode_prior_precision = None
    mode_current_cost = None
    mode_nis_score = 0.0
    mode_alpha = 0.0

    for _ in range(n_iters):
        with torch.no_grad():
            r0 = _scaled_se3_residuals_cached_rotation(
                poses,
                log_s,
                ii,
                jj,
                rel_edges,
                meas_rotations_inv,
            ) * sqrt_info
            edge_norm = r0.norm(dim=-1).clamp_min(EPS)
            robust = torch.where(edge_norm <= huber_delta, torch.ones_like(edge_norm), huber_delta / edge_norm)
            robust = robust.sqrt().unsqueeze(-1)
            weighted_r0 = r0 * robust
            s0 = _relative_scale_residuals(log_s, ii, jj, rel_edges) * scale_sqrt_info
            base_current_cost = 0.5 * (
                (weighted_r0 ** 2).sum()
                + (s0 ** 2).sum()
            )
            H, g, _ = _se3_scale_linear_system(
                poses,
                log_s,
                ii,
                jj,
                rel_edges,
                anchor,
                sqrt_info,
                robust,
                scale_sqrt_info,
            )

        improved = False
        step_norm = 0.0
        for _ in range(lm_max_attempts):
            A = H + lm * eye
            chol, chol_info = torch.linalg.cholesky_ex(A)
            if bool((chol_info != 0).any()):
                lm = min(lm * LM_DAMPING_INCREASE, LM_DAMPING_MAX)
                continue
            step = -torch.cholesky_solve(g[:, None], chol).squeeze(-1)
            current_cost = base_current_cost

            if mode_basis is not None:
                mode_rhs = torch.zeros(n_vars, mode_basis.shape[0], device=device, dtype=dtype)
                mode_rhs[n_pose_vars:, :] = mode_basis.transpose(0, 1)
                mode_rhs[n_pose_vars + int(anchor), :] = 0
                mode_response = torch.cholesky_solve(mode_rhs, chol)
                mode_covariance = _regularize_mode_covariance(
                    mode_basis @ mode_response[n_pose_vars:, :],
                    mode_identity,
                )
                innovation = mode_residual + mode_basis @ step[n_pose_vars:]
                if mode_prior_covariance is None:
                    mode_prior_covariance, mode_nis_score, mode_alpha = _moge_mode_prior_covariance(
                        mode_covariance,
                        innovation,
                        moge_mode_nis_cutoff,
                    )
                    mode_prior_precision = torch.linalg.inv(mode_prior_covariance)
                if mode_current_cost is None:
                    mode_current_cost = _mode_quadratic_cost(mode_residual, mode_prior_precision)
                middle = mode_prior_covariance + mode_covariance
                correction = torch.linalg.solve(middle, innovation)
                step = step - mode_response @ correction
                current_cost = current_cost + mode_current_cost

            if initial_cost is None:
                initial_cost = float(current_cost.cpu())
            pose_delta = step[:n_pose_vars].reshape(n_nodes - 1, 6)
            scale_delta = step[n_pose_vars : n_pose_vars + n_nodes]
            cand_poses, cand_log_s = _apply_delta(
                poses,
                log_s,
                pose_delta,
                scale_delta,
                optimized_nodes,
            )
            cand_r = _scaled_se3_residuals_cached_rotation(
                cand_poses,
                cand_log_s,
                ii,
                jj,
                rel_edges,
                meas_rotations_inv,
            )
            cand_s = _relative_scale_residuals(cand_log_s, ii, jj, rel_edges) * scale_sqrt_info
            cand_weighted = cand_r * sqrt_info * robust
            cand_cost = 0.5 * (
                (cand_weighted ** 2).sum()
                + (cand_s ** 2).sum()
            )
            if mode_basis is not None:
                cand_moge = cand_log_s - moge_log_s
                cand_mode_residual = mode_basis @ cand_moge
                cand_mode_cost = _mode_quadratic_cost(cand_mode_residual, mode_prior_precision)
                cand_cost = cand_cost + cand_mode_cost
            cand_cost_value, current_cost_value, step_norm = torch.stack(
                (cand_cost, current_cost, step.norm())
            ).tolist()
            if cand_cost_value <= current_cost_value:
                poses, log_s = cand_poses, cand_log_s
                if mode_basis is not None:
                    mode_residual = cand_mode_residual
                    mode_current_cost = cand_mode_cost
                lm = max(lm * LM_DAMPING_DECREASE, LM_DAMPING_MIN)
                accepted += 1
                improved = True
                break
            lm = min(lm * LM_DAMPING_INCREASE, LM_DAMPING_MAX)

        if not improved:
            break

        if step_norm < 1e-5:
            break

    final_res = _scaled_se3_residuals_cached_rotation(
        poses,
        log_s,
        ii,
        jj,
        rel_edges,
        meas_rotations_inv,
    ) * sqrt_info
    final_scale = _relative_scale_residuals(log_s, ii, jj, rel_edges) * scale_sqrt_info
    final_cost_tensor = 0.5 * (
        (final_res ** 2).sum()
        + (final_scale ** 2).sum()
    )
    if mode_basis is not None and mode_prior_covariance is not None:
        final_cost_tensor = final_cost_tensor + _mode_quadratic_cost(
            mode_residual,
            mode_prior_precision,
        )
    final_cost = float(final_cost_tensor.cpu())
    finite = bool((torch.isfinite(final_res).all() & torch.isfinite(final_scale).all()).cpu())
    start_cost = final_cost if initial_cost is None else initial_cost
    info = _pgo_info(
        success=finite and (n_iters == 0 or accepted > 0 or final_cost <= start_cost + 1e-6),
        mode="se3_scale",
        n_edges=int(ii.numel()),
        n_iters=int(n_iters),
        accepted_iters=accepted,
        cost=final_cost,
    )
    info.update(
        moge_mode_nis=bool(moge_mode_nis),
        moge_mode_count=0 if mode_basis is None else int(mode_basis.shape[0]),
        moge_mode_nis_score=float(mode_nis_score),
        moge_mode_alpha=float(mode_alpha),
    )
    return Sim3PGOResult(poses=poses.data, log_scales=log_s, info=info)
