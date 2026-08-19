from __future__ import annotations

from dataclasses import dataclass

import torch
from lietorch import SE3, SO3

from .common import (
    LM_DAMPING_DECREASE,
    LM_DAMPING_INCREASE,
    LM_DAMPING_MAX,
    LM_DAMPING_MIN,
    RelativeEdges,
    Sim3PGOResult,
    _apply_delta,
    _dct_scale_modes,
    _edge_sqrt_information,
    _mode_quadratic_cost,
    _moge_mode_prior_covariance,
    _regularize_mode_covariance,
    _relative_scale_residuals,
    _scale_sqrt_information,
    _scaled_se3_residuals_cached_rotation,
)
from .cuda_eigen import _require_cuda_float32, _se3_scale_cuda_weighted_blocks


BLOCK_DIM = 7
LOCAL_MOGE_MODE_COUNT = 2


@dataclass
class MarginalPrior:
    start: int
    end: int
    reference_poses: torch.Tensor
    reference_log_scales: torch.Tensor
    hessian: torch.Tensor
    gradient: torch.Tensor


def _dense_normal_system(
    source_block: torch.Tensor,
    target_block: torch.Tensor,
    residual: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    n_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = source_block.device
    n_vars = n_nodes * BLOCK_DIM
    hessian_blocks = torch.zeros(
        n_nodes * n_nodes,
        BLOCK_DIM,
        BLOCK_DIM,
        device=device,
        dtype=torch.float64,
    )
    gradient_blocks = torch.zeros(n_nodes, BLOCK_DIM, device=device, dtype=torch.float64)
    source = source_block.to(dtype=torch.float64)
    target = target_block.to(dtype=torch.float64)
    residual64 = residual.to(dtype=torch.float64)

    def add_hessian(node_a: torch.Tensor, block_a: torch.Tensor, node_b: torch.Tensor, block_b: torch.Tensor) -> None:
        values = torch.einsum("eri,erj->eij", block_a, block_b)
        hessian_blocks.index_add_(0, node_a * n_nodes + node_b, values)

    gradient_blocks.index_add_(0, ii, torch.einsum("eri,er->ei", source, residual64))
    gradient_blocks.index_add_(0, jj, torch.einsum("eri,er->ei", target, residual64))
    add_hessian(ii, source, ii, source)
    add_hessian(ii, source, jj, target)
    add_hessian(jj, target, ii, source)
    add_hessian(jj, target, jj, target)
    hessian = hessian_blocks.view(n_nodes, n_nodes, BLOCK_DIM, BLOCK_DIM)
    hessian = hessian.permute(0, 2, 1, 3).reshape(n_vars, n_vars)
    return hessian, gradient_blocks.reshape(-1)


def _prior_delta(prior: MarginalPrior, poses: SE3, log_scales: torch.Tensor) -> torch.Tensor:
    pose_delta = (SE3(prior.reference_poses).inv() * poses).log()
    scale_delta = log_scales - prior.reference_log_scales
    return torch.cat((pose_delta, scale_delta[:, None]), dim=-1).to(dtype=torch.float64).reshape(-1)


def _add_prior(
    hessian: torch.Tensor,
    gradient: torch.Tensor,
    prior: MarginalPrior,
    poses: SE3,
    log_scales: torch.Tensor,
) -> torch.Tensor:
    delta = _prior_delta(prior, poses, log_scales)
    gradient.add_(prior.gradient + prior.hessian @ delta)
    hessian.add_(prior.hessian)
    return 0.5 * torch.dot(delta, prior.hessian @ delta) + torch.dot(prior.gradient, delta)


def _recenter_prior(
    prior: MarginalPrior,
    poses: SE3,
    log_scales: torch.Tensor,
) -> None:
    delta = _prior_delta(prior, poses, log_scales)
    prior.gradient = prior.gradient + prior.hessian @ delta
    prior.reference_poses = poses.data.clone()
    prior.reference_log_scales = log_scales.clone()


def _constrain_node(hessian: torch.Tensor, gradient: torch.Tensor, node: int) -> None:
    start = int(node) * BLOCK_DIM
    end = start + BLOCK_DIM
    hessian[start:end, :] = 0
    hessian[:, start:end] = 0
    hessian[start:end, start:end] = torch.eye(BLOCK_DIM, device=hessian.device, dtype=hessian.dtype)
    gradient[start:end] = 0


def _schur_complement(
    hessian: torch.Tensor,
    gradient: torch.Tensor,
    marginalized_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    h_dd = 0.5 * (
        hessian[:marginalized_dim, :marginalized_dim]
        + hessian[:marginalized_dim, :marginalized_dim].transpose(0, 1)
    )
    h_dr = hessian[:marginalized_dim, marginalized_dim:]
    h_rd = hessian[marginalized_dim:, :marginalized_dim]
    h_rr = hessian[marginalized_dim:, marginalized_dim:]
    g_d = gradient[:marginalized_dim]
    g_r = gradient[marginalized_dim:]
    jitter = torch.clamp(1e-6 * h_dd.diagonal().abs().mean(), min=1e-9)
    chol = torch.linalg.cholesky(h_dd + jitter * torch.eye(h_dd.shape[0], device=h_dd.device, dtype=h_dd.dtype))
    solve_h = torch.cholesky_solve(h_dr, chol)
    solve_g = torch.cholesky_solve(g_d[:, None], chol).squeeze(-1)
    prior_hessian = h_rr - h_rd @ solve_h
    prior_gradient = g_r - h_rd @ solve_g
    prior_hessian = 0.5 * (prior_hessian + prior_hessian.transpose(0, 1))
    prior_jitter = torch.clamp(1e-6 * prior_hessian.diagonal().abs().mean(), min=1e-9)
    prior_hessian = prior_hessian + prior_jitter * torch.eye(
        prior_hessian.shape[0],
        device=prior_hessian.device,
        dtype=prior_hessian.dtype,
    )
    return prior_hessian, prior_gradient


def _weighted_blocks(
    poses: SE3,
    log_scales: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    rel_edges: RelativeEdges,
    edge_conf: torch.Tensor,
    huber_delta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_edges = int(ii.numel())
    buffers = (
        poses.data.new_empty(n_edges, BLOCK_DIM, BLOCK_DIM),
        poses.data.new_empty(n_edges, BLOCK_DIM, BLOCK_DIM),
        poses.data.new_empty(n_edges, BLOCK_DIM),
        poses.data.new_empty(n_edges, 1),
        poses.data.new_empty(2),
    )
    return _se3_scale_cuda_weighted_blocks(
        poses,
        log_scales,
        ii,
        jj,
        rel_edges,
        _edge_sqrt_information(edge_conf),
        huber_delta,
        _scale_sqrt_information(edge_conf),
        buffers,
    )


class MarginalizedLocalPGO:
    def __init__(self, window_size: int):
        self.window_size = int(window_size)
        self.window_start = 0
        self.window_end = 0
        self.prior: MarginalPrior | None = None
        self.ii: torch.Tensor | None = None
        self.jj: torch.Tensor | None = None
        self.rel_poses: torch.Tensor | None = None
        self.rel_log_scales: torch.Tensor | None = None
        self.edge_conf: torch.Tensor | None = None

    def _append_edges(
        self,
        ii: torch.Tensor,  # [E]
        jj: torch.Tensor,  # [E]
        rel_poses: torch.Tensor,  # [E,7]
        rel_log_scales: torch.Tensor,  # [E]
        edge_conf: torch.Tensor,  # [E,3]
    ) -> None:
        if self.ii is None:
            self.ii = ii
            self.jj = jj
            self.rel_poses = rel_poses
            self.rel_log_scales = rel_log_scales
            self.edge_conf = edge_conf
            return
        self.ii = torch.cat((self.ii, ii))
        self.jj = torch.cat((self.jj, jj))
        self.rel_poses = torch.cat((self.rel_poses, rel_poses))
        self.rel_log_scales = torch.cat((self.rel_log_scales, rel_log_scales))
        self.edge_conf = torch.cat((self.edge_conf, edge_conf))

    def _discard_marginalized_edges(self, window_start: int) -> None:
        keep = (self.ii >= int(window_start)) & (self.jj >= int(window_start))
        self.ii = self.ii[keep]
        self.jj = self.jj[keep]
        self.rel_poses = self.rel_poses[keep]
        self.rel_log_scales = self.rel_log_scales[keep]
        self.edge_conf = self.edge_conf[keep]

    def _marginalize(
        self,
        new_start: int,
        new_end: int,
        poses: torch.Tensor,  # [new_end-window_start,7]
        log_scales: torch.Tensor,  # [new_end-window_start]
        ii: torch.Tensor,  # [E]
        jj: torch.Tensor,  # [E]
        rel_poses: torch.Tensor,  # [E,7]
        rel_log_scales: torch.Tensor,  # [E]
        edge_conf: torch.Tensor,  # [E,3]
        huber_delta: float,
    ) -> MarginalPrior:
        old_start = self.window_start
        union_end = int(new_end)
        in_union = (ii >= old_start) & (ii < union_end) & (jj >= old_start) & (jj < union_end)
        retiring = in_union & ((ii < int(new_start)) | (jj < int(new_start)))
        local_ii = ii[retiring] - old_start
        local_jj = jj[retiring] - old_start
        union_poses = SE3(poses)
        union_log_scales = log_scales
        n_nodes = union_end - old_start
        n_vars = n_nodes * BLOCK_DIM
        hessian = torch.zeros(n_vars, n_vars, device=poses.device, dtype=torch.float64)
        gradient = torch.zeros(n_vars, device=poses.device, dtype=torch.float64)

        if local_ii.numel() > 0:
            rel_edges = RelativeEdges(
                poses=SE3(rel_poses[retiring]),
                log_scales=rel_log_scales[retiring],
            )
            source, target, residual, _, _, _ = _weighted_blocks(
                union_poses,
                union_log_scales,
                local_ii,
                local_jj,
                rel_edges,
                edge_conf[retiring],
                huber_delta,
            )
            edge_hessian, edge_gradient = _dense_normal_system(
                source,
                target,
                residual,
                local_ii,
                local_jj,
                n_nodes,
            )
            hessian.add_(edge_hessian)
            gradient.add_(edge_gradient)

        if self.prior is not None:
            assert self.prior.start == old_start
            assert self.prior.end == self.window_end
            prior_dim = self.prior.hessian.shape[0]
            prior_slice = slice(0, prior_dim)
            prior_start = self.prior.start - old_start
            prior_end = self.prior.end - old_start
            prior_poses = SE3(poses[prior_start:prior_end])
            prior_log_scales = log_scales[prior_start:prior_end]
            delta = _prior_delta(self.prior, prior_poses, prior_log_scales)
            hessian[prior_slice, prior_slice].add_(self.prior.hessian)
            gradient[prior_slice].add_(self.prior.gradient + self.prior.hessian @ delta)
        elif old_start == 0:
            _constrain_node(hessian, gradient, 0)

        marginalized_dim = (int(new_start) - old_start) * BLOCK_DIM
        prior_hessian, prior_gradient = _schur_complement(hessian, gradient, marginalized_dim)
        return MarginalPrior(
            start=int(new_start),
            end=union_end,
            reference_poses=poses[new_start - old_start :].clone(),
            reference_log_scales=log_scales[new_start - old_start :].clone(),
            hessian=prior_hessian,
            gradient=prior_gradient,
        )

    def _optimize_active(
        self,
        poses: SE3,
        log_scales: torch.Tensor,
        moge_log_scales: torch.Tensor,
        ii: torch.Tensor,
        jj: torch.Tensor,
        rel_edges: RelativeEdges,
        edge_conf: torch.Tensor,
        *,
        fix_first_node: bool,
        n_iters: int,
        damping: float,
        lm_max_attempts: int,
        huber_delta: float,
        nis_cutoff: float,
    ) -> Sim3PGOResult:
        _require_cuda_float32(poses)
        n_nodes = poses.data.shape[0]
        if self.prior is not None:
            assert self.prior.end - self.prior.start == n_nodes
        n_vars = n_nodes * BLOCK_DIM
        node_ids = torch.arange(n_nodes, device=poses.data.device)
        optimized_nodes = node_ids[1:] if fix_first_node else node_ids
        mode_basis = _dct_scale_modes(n_nodes, LOCAL_MOGE_MODE_COUNT, log_scales)
        mode_identity = torch.eye(mode_basis.shape[0], device=log_scales.device, dtype=torch.float64)
        mode_residual = (mode_basis @ (log_scales - moge_log_scales)).to(dtype=torch.float64)
        mode_prior_covariance = None
        mode_prior_precision = None
        mode_current_cost = None
        mode_nis_score = 0.0
        mode_alpha = 0.0
        meas_rotations_inv = SO3(rel_edges.poses.data[..., 3:7]).inv()
        sqrt_info = _edge_sqrt_information(edge_conf)
        scale_sqrt_info = _scale_sqrt_information(edge_conf)
        accepted = 0
        initial_cost = None
        lm = float(damping)
        eye = torch.eye(n_vars, device=poses.data.device, dtype=torch.float64)

        for _ in range(n_iters):
            source, target, residual, robust, edge_cost, unrobust_cost = _weighted_blocks(
                poses,
                log_scales,
                ii,
                jj,
                rel_edges,
                edge_conf,
                huber_delta,
            )
            hessian, gradient = _dense_normal_system(source, target, residual, ii, jj, n_nodes)
            prior_cost = hessian.new_zeros(())
            if self.prior is not None:
                prior_cost = _add_prior(hessian, gradient, self.prior, poses, log_scales)
            if fix_first_node:
                _constrain_node(hessian, gradient, 0)
            base_current_cost = edge_cost.to(dtype=torch.float64) + prior_cost

            improved = False
            step_norm = 0.0
            for _ in range(lm_max_attempts):
                chol, info = torch.linalg.cholesky_ex(hessian + lm * eye)
                if bool((info != 0).any()):
                    lm = min(lm * LM_DAMPING_INCREASE, LM_DAMPING_MAX)
                    continue
                step = -torch.cholesky_solve(gradient[:, None], chol).squeeze(-1)
                mode_rhs = torch.zeros(n_vars, mode_basis.shape[0], device=poses.data.device, dtype=torch.float64)
                mode_rhs[6::BLOCK_DIM] = mode_basis.transpose(0, 1).to(dtype=torch.float64)
                if fix_first_node:
                    mode_rhs[:BLOCK_DIM] = 0
                mode_response = torch.cholesky_solve(mode_rhs, chol)
                mode_covariance = _regularize_mode_covariance(
                    mode_basis.to(dtype=torch.float64) @ mode_response[6::BLOCK_DIM],
                    mode_identity,
                )
                innovation = mode_residual + mode_basis.to(dtype=torch.float64) @ step[6::BLOCK_DIM]
                if mode_prior_covariance is None:
                    mode_prior_covariance, mode_nis_score, mode_alpha = _moge_mode_prior_covariance(
                        mode_covariance,
                        innovation,
                        nis_cutoff,
                    )
                    mode_prior_precision = torch.linalg.inv(mode_prior_covariance)
                if mode_current_cost is None:
                    mode_current_cost = _mode_quadratic_cost(mode_residual, mode_prior_precision)
                correction = torch.linalg.solve(mode_prior_covariance + mode_covariance, innovation)
                step = step - mode_response @ correction
                if fix_first_node:
                    step[:BLOCK_DIM] = 0
                current_cost = base_current_cost + mode_current_cost
                if initial_cost is None:
                    initial_cost = float(unrobust_cost + prior_cost + mode_current_cost)

                step_full = step.reshape(n_nodes, BLOCK_DIM).to(dtype=poses.data.dtype)
                cand_poses, cand_log_scales = _apply_delta(
                    poses,
                    log_scales,
                    step_full[optimized_nodes, :6],
                    step_full[:, 6],
                    optimized_nodes,
                )
                cand_residual = _scaled_se3_residuals_cached_rotation(
                    cand_poses,
                    cand_log_scales,
                    ii,
                    jj,
                    rel_edges,
                    meas_rotations_inv,
                )
                cand_scale = _relative_scale_residuals(cand_log_scales, ii, jj, rel_edges) * scale_sqrt_info
                cand_cost = 0.5 * (
                    ((cand_residual * sqrt_info * robust) ** 2).sum()
                    + (cand_scale * cand_scale).sum()
                ).to(dtype=torch.float64)
                if self.prior is not None:
                    cand_delta = _prior_delta(self.prior, cand_poses, cand_log_scales)
                    cand_cost = cand_cost + 0.5 * torch.dot(
                        cand_delta,
                        self.prior.hessian @ cand_delta,
                    ) + torch.dot(self.prior.gradient, cand_delta)
                cand_mode_residual = (
                    mode_basis @ (cand_log_scales - moge_log_scales)
                ).to(dtype=torch.float64)
                cand_mode_cost = _mode_quadratic_cost(cand_mode_residual, mode_prior_precision)
                cand_cost = cand_cost + cand_mode_cost
                cand_value, current_value, step_norm = torch.stack(
                    (cand_cost, current_cost, step.norm())
                ).tolist()
                if cand_value <= current_value:
                    poses, log_scales = cand_poses, cand_log_scales
                    mode_residual = cand_mode_residual
                    mode_current_cost = cand_mode_cost
                    lm = max(lm * LM_DAMPING_DECREASE, LM_DAMPING_MIN)
                    accepted += 1
                    improved = True
                    break
                lm = min(lm * LM_DAMPING_INCREASE, LM_DAMPING_MAX)

            if not improved or step_norm < 1e-5:
                break

        final_residual = _scaled_se3_residuals_cached_rotation(
            poses,
            log_scales,
            ii,
            jj,
            rel_edges,
            meas_rotations_inv,
        ) * sqrt_info
        final_scale = _relative_scale_residuals(log_scales, ii, jj, rel_edges) * scale_sqrt_info
        final_cost_tensor = 0.5 * (
            (final_residual * final_residual).sum()
            + (final_scale * final_scale).sum()
        ).to(dtype=torch.float64)
        if self.prior is not None:
            final_delta = _prior_delta(self.prior, poses, log_scales)
            final_cost_tensor = final_cost_tensor + 0.5 * torch.dot(
                final_delta,
                self.prior.hessian @ final_delta,
            ) + torch.dot(self.prior.gradient, final_delta)
        if mode_current_cost is not None:
            final_cost_tensor = final_cost_tensor + mode_current_cost
        final_cost = float(final_cost_tensor)
        start_cost = final_cost if initial_cost is None else initial_cost
        finite = torch.isfinite(poses.data).all() & torch.isfinite(log_scales).all()
        success = bool(finite) and (
            n_iters == 0 or accepted > 0 or final_cost <= start_cost + 1e-6
        )
        info = {
            "success": success,
            "mode": "marginalized_local_se3_scale",
            "backend": "cuda_dense",
            "n_edges": int(ii.numel()),
            "n_iters": int(n_iters),
            "accepted_iters": accepted,
            "cost": final_cost,
            "moge_mode_nis": True,
            "moge_mode_count": LOCAL_MOGE_MODE_COUNT,
            "moge_mode_dimensions": int(mode_basis.shape[0]),
            "moge_mode_nis_score": float(mode_nis_score),
            "moge_mode_alpha": float(mode_alpha),
        }
        return Sim3PGOResult(poses=poses.data, log_scales=log_scales, info=info)

    def step(
        self,
        *,
        finalized_end: int,
        poses: torch.Tensor,  # [finalized_end-window_start,7]
        log_scales: torch.Tensor,  # [finalized_end-window_start]
        moge_log_scales: torch.Tensor,  # [finalized_end-window_start]
        ii: torch.Tensor,  # [E]
        jj: torch.Tensor,  # [E]
        rel_poses: torch.Tensor,  # [E,7]
        rel_log_scales: torch.Tensor,  # [E]
        edge_conf: torch.Tensor,  # [E,3]
        n_iters: int,
        damping: float,
        lm_max_attempts: int,
        huber_delta: float,
        nis_cutoff: float,
    ) -> tuple[int, Sim3PGOResult]:
        finalized_end = int(finalized_end)
        if finalized_end <= self.window_end:
            raise ValueError("marginalized local PGO finalized_end must increase")
        state_start = self.window_start
        self._append_edges(ii, jj, rel_poses, rel_log_scales, edge_conf)
        ii = self.ii
        jj = self.jj
        rel_poses = self.rel_poses
        rel_log_scales = self.rel_log_scales
        edge_conf = self.edge_conf
        new_start = max(0, finalized_end - self.window_size)
        if self.window_end > 0 and new_start > self.window_start:
            if new_start >= self.window_end:
                raise ValueError("marginalized local PGO updates must retain an overlap")
            self.prior = self._marginalize(
                new_start,
                finalized_end,
                poses,
                log_scales,
                ii,
                jj,
                rel_poses,
                rel_log_scales,
                edge_conf,
                huber_delta,
            )

        active = (ii >= new_start) & (ii < finalized_end) & (jj >= new_start) & (jj < finalized_end)
        local_ii = ii[active] - new_start
        local_jj = jj[active] - new_start
        rel_edges = RelativeEdges(
            poses=SE3(rel_poses[active]),
            log_scales=rel_log_scales[active],
        )
        active_start = new_start - state_start
        result = self._optimize_active(
            SE3(poses[active_start:]),
            log_scales[active_start:],
            moge_log_scales[active_start:],
            local_ii,
            local_jj,
            rel_edges,
            edge_conf[active],
            fix_first_node=new_start == 0,
            n_iters=n_iters,
            damping=damping,
            lm_max_attempts=lm_max_attempts,
            huber_delta=huber_delta,
            nis_cutoff=nis_cutoff,
        )
        if self.prior is not None:
            _recenter_prior(self.prior, SE3(result.poses), result.log_scales)
        self._discard_marginalized_edges(new_start)
        self.window_start = new_start
        self.window_end = finalized_end
        result.info.update(
            window_start=new_start,
            window_end=finalized_end,
            prior_nodes=0 if self.prior is None else self.prior.end - self.prior.start,
            active_factor_count=int(self.ii.numel()),
        )
        return new_start, result
