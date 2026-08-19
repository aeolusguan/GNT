from __future__ import annotations

from dataclasses import dataclass
import heapq

import torch
from lietorch import SO3, SE3


EPS = 1e-8
DEFAULT_LM_MAX_ATTEMPTS = 5
LM_DAMPING_DECREASE = 0.5
LM_DAMPING_INCREASE = 10.0
LM_DAMPING_MIN = 1e-7
LM_DAMPING_MAX = 1e7
PGO_MODES = {"rotation_only", "staged", "se3_scale"}
PGO_BACKENDS = {"torch", "cuda_eigen"}

# Shape notation:
#   N: number of pose-graph nodes, E: number of directed edges.
#   A: number of optimized nodes after excluding the anchor, A = N - 1.
#   Pose tensors use (tx, ty, tz, qx, qy, qz, qw), so SE3.data is (..., 7).



@dataclass
class Sim3PGOResult:
    poses: torch.Tensor  # [N,7]
    log_scales: torch.Tensor  # [N]
    info: dict
    initial_poses: torch.Tensor | None = None


@dataclass
class RelativeEdges:
    poses: SE3  # data: [E,7]
    log_scales: torch.Tensor  # [E]


def _apply_delta(
    poses: SE3,  # data: [N,7]
    log_s: torch.Tensor,  # [N]
    pose_delta: torch.Tensor,  # [A,6]
    scale_delta: torch.Tensor,  # [N]
    optimized_nodes: torch.Tensor,  # [A]
):
    """Apply full pose/scale increments."""
    if pose_delta.numel() == 0 and scale_delta.numel() == 0:
        return poses, log_s

    pose_data = poses.data
    pose_new = pose_data.clone()
    if pose_delta.numel() > 0:
        delta_poses = SE3(torch.cat((pose_delta[:, :3], SO3.exp(pose_delta[:, 3:6]).data), dim=-1))
        pose_new[optimized_nodes] = (SE3(pose_data[optimized_nodes]) * delta_poses).data
    return SE3(pose_new), log_s + scale_delta


def _scaled_se3_residuals(
    poses: SE3,  # data: [N,7]
    log_s: torch.Tensor,  # [N]
    ii: torch.Tensor,  # [E]
    jj: torch.Tensor,  # [E]
    rel_edges: RelativeEdges,  # poses: [E,7]
) -> torch.Tensor:
    meas_rotations_inv = SO3(rel_edges.poses.data[..., 3:7]).inv()
    return _scaled_se3_residuals_cached_rotation(
        poses,
        log_s,
        ii,
        jj,
        rel_edges,
        meas_rotations_inv,
    )


def _scaled_se3_residuals_cached_rotation(
    poses: SE3,  # data: [N,7]
    log_s: torch.Tensor,  # [N]
    ii: torch.Tensor,  # [E]
    jj: torch.Tensor,  # [E]
    rel_edges: RelativeEdges,  # poses: [E,7]
    meas_rotations_inv: SO3,  # data: [E,4]
) -> torch.Tensor:
    """Residual for GNT-normalized relative poses.

    GNT is trained with relative translations divided by the source depth
    scale. The optimized state keeps metric SE(3) camera poses plus a coherent
    per-node pose scale:

        normalize(T_j * inv(T_i).translation, scale_i)
          ~= edge_relative_pose_ij[:3]
    """
    pred = (poses[jj] * poses[ii].inv()).data
    pred_t = pred[..., :3] / torch.exp(log_s[ii]).clamp_min(EPS)[..., None]
    pred_q = pred[..., 3:7]
    err_t = pred_t - rel_edges.poses.data[..., :3]
    err_r = (meas_rotations_inv * SO3(pred_q)).log()
    return torch.cat((err_t, err_r), dim=-1)


def _rotation_residuals(
    rotations: SO3,  # data: [N,4]
    ii: torch.Tensor,  # [E]
    jj: torch.Tensor,  # [E]
    meas_rotations: SO3,  # data: [E,4]
) -> torch.Tensor:
    """Rotation-only edge residuals.

    This is the rotation-averaging objective for our edge convention:

        r_ij = Log(R_ij_meas^{-1} * R_j * R_i^{-1}).
    """
    pred = rotations[jj] * rotations[ii].inv()
    return (meas_rotations.inv() * pred).log()


def _so3_hat(
    phi: torch.Tensor,  # [...,3]
) -> torch.Tensor:
    """Hat operator for SO(3)."""
    x, y, z = phi.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    return torch.stack(
        (
            torch.stack((zeros, -z, y), dim=-1),
            torch.stack((z, zeros, -x), dim=-1),
            torch.stack((-y, x, zeros), dim=-1),
        ),
        dim=-2,
    )


def _so3_right_jacobian_inverse(
    phi: torch.Tensor,  # [E,3]
) -> torch.Tensor:
    """SO(3) right-Jacobian inverse for Log(Exp(phi) * Exp(delta))."""
    device = phi.device
    dtype = phi.dtype
    theta_sq = (phi * phi).sum(dim=-1)
    theta = torch.sqrt(theta_sq.clamp_min(EPS))
    hat = _so3_hat(phi)
    hat_sq = hat @ hat
    eye = torch.eye(3, device=device, dtype=dtype).expand(phi.shape[0], 3, 3)

    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    denom = (2.0 * theta * sin_theta).clamp_min(EPS)
    coeff = 1.0 / theta_sq.clamp_min(EPS) - (1.0 + cos_theta) / denom
    taylor = 1.0 / 12.0 + theta_sq / 720.0 + theta_sq * theta_sq / 30240.0
    coeff = torch.where(theta_sq > 1e-8, coeff, taylor)
    return eye + 0.5 * hat + coeff[:, None, None] * hat_sq


def _edge_sqrt_information(
    edge_conf: torch.Tensor,  # [E,3]
):
    info = edge_conf[..., :2].clamp_min(0.0)
    return torch.sqrt(torch.cat((info[:, :1].expand(-1, 3), info[:, 1:2].expand(-1, 3)), dim=-1))


def _rotation_sqrt_information(
    edge_conf: torch.Tensor,  # [E,3]
):
    rot_info = edge_conf[..., 1]
    return torch.sqrt(rot_info.clamp_min(0.0))[:, None].expand(-1, 3)


def _translation_sqrt_information(
    edge_conf: torch.Tensor,  # [E,3]
):
    trans_info = edge_conf[..., 0]
    return torch.sqrt(trans_info.clamp_min(0.0))[:, None].expand(-1, 3)


def _scale_sqrt_information(
    edge_conf: torch.Tensor,  # [E,3]
):
    return torch.sqrt(edge_conf[..., 2].clamp_min(0.0))


def _dct_scale_modes(n_nodes: int, n_modes: int, reference: torch.Tensor) -> torch.Tensor:
    """Mean-zero orthonormal DCT modes, excluding the global scale gauge."""
    mode_count = min(int(n_modes), int(n_nodes))
    samples = torch.arange(n_nodes, device=reference.device, dtype=reference.dtype) + 0.5
    modes = torch.arange(1, mode_count, device=reference.device, dtype=reference.dtype)[:, None]
    return torch.sqrt(reference.new_tensor(2.0 / float(n_nodes))) * torch.cos(
        torch.pi * modes * samples[None, :] / float(n_nodes)
    )


def _regularize_mode_covariance(
    covariance: torch.Tensor,
    identity: torch.Tensor,
) -> torch.Tensor:
    """Symmetrize a small marginalized covariance and add numerical jitter."""
    covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
    jitter = (1e-6 * covariance.diagonal().abs().mean()).clamp_min(EPS)
    return covariance + jitter * identity


def _moge_mode_prior_covariance(
    marginalized_covariance: torch.Tensor,
    innovation: torch.Tensor,
    nis_cutoff: float,
) -> tuple[torch.Tensor, float, float]:
    """Convert modal NIS into the covariance of a fixed MoGe mode factor."""
    nis = innovation.square() / marginalized_covariance.diagonal().clamp_min(EPS)
    score = nis.median()
    alpha = float(nis_cutoff) / (float(nis_cutoff) + score)
    alpha = alpha.clamp(min=1e-6, max=1.0 - 1e-4)
    prior_covariance = ((1.0 - alpha) / alpha) * marginalized_covariance
    return prior_covariance, float(score.cpu()), float(alpha.cpu())


def _mode_quadratic_cost(residual: torch.Tensor, precision: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.dot(residual, precision @ residual)


def _relative_scale_residuals(
    log_s: torch.Tensor,  # [N]
    ii: torch.Tensor,  # [E]
    jj: torch.Tensor,  # [E]
    rel_edges: RelativeEdges,  # log_scales: [E]
) -> torch.Tensor:
    """Relative scale residuals: log_s[j] - log_s[i] ~= edge log-scale."""
    return log_s[jj] - log_s[ii] - rel_edges.log_scales


def _initial_from_edges(
    n_nodes: int,
    ii: torch.Tensor,  # [E]
    jj: torch.Tensor,  # [E]
    rel_edges: RelativeEdges,  # poses: [E,7]
    anchor: int,
    log_s: torch.Tensor,  # [N]
    edge_conf: torch.Tensor,  # [E,2]
):
    """Build initial poses from pose graph edges."""
    edge_data = rel_edges.poses.data
    device = edge_data.device
    dtype = edge_data.dtype
    edge_ii = ii.cpu().tolist()
    edge_jj = jj.cpu().tolist()
    edge_values = edge_data.cpu().tolist()
    scales = torch.exp(log_s).cpu().tolist()
    pose_conf = edge_conf[..., :2].min(dim=-1).values.cpu().tolist()

    def qmul(a, b):
        ax, ay, az, aw = a
        bx, by, bz, bw = b
        return (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        )

    def qinv(q):
        x, y, z, w = q
        norm_sq = max(x * x + y * y + z * z + w * w, EPS)
        return (-x / norm_sq, -y / norm_sq, -z / norm_sq, w / norm_sq)

    def qnorm(q):
        x, y, z, w = q
        norm = max((x * x + y * y + z * z + w * w) ** 0.5, EPS)
        return (x / norm, y / norm, z / norm, w / norm)

    def qact(q, v):
        x, y, z, w = q
        vx, vy, vz = v
        tx = 2.0 * (y * vz - z * vy)
        ty = 2.0 * (z * vx - x * vz)
        tz = 2.0 * (x * vy - y * vx)
        return (
            vx + w * tx + y * tz - z * ty,
            vy + w * ty + z * tx - x * tz,
            vz + w * tz + x * ty - y * tx,
        )

    def vadd(a, b):
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

    def vsub(a, b):
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    def vmul(v, scale):
        return (v[0] * scale, v[1] * scale, v[2] * scale)

    adjacency = [[] for _ in range(n_nodes)]
    for edge_id, (i, j) in enumerate(zip(edge_ii, edge_jj, strict=True)):
        key = (abs(i - j), -float(pose_conf[edge_id]), edge_id)
        adjacency[i].append((key, edge_id))
        adjacency[j].append((key, edge_id))

    pose_q = [(0.0, 0.0, 0.0, 1.0) for _ in range(n_nodes)]
    pose_t = [(0.0, 0.0, 0.0) for _ in range(n_nodes)]
    visited = [False for _ in range(n_nodes)]
    visited[anchor] = True
    heap = []
    for key, edge_id in adjacency[anchor]:
        heapq.heappush(heap, (key, edge_id))

    n_visited = 1
    while heap and n_visited < n_nodes:
        _, edge_id = heapq.heappop(heap)
        i = edge_ii[edge_id]
        j = edge_jj[edge_id]
        i_visited = visited[i]
        j_visited = visited[j]
        if i_visited == j_visited:
            continue
        meas_t = edge_values[edge_id][:3]
        meas_q = edge_values[edge_id][3:7]
        if i_visited:
            pose_q[j] = qnorm(qmul(meas_q, pose_q[i]))
            pose_t[j] = vadd(vmul(meas_t, scales[i]), qact(meas_q, pose_t[i]))
            visited[j] = True
            new_node = j
        else:
            meas_inv = qinv(meas_q)
            pose_q[i] = qnorm(qmul(meas_inv, pose_q[j]))
            pose_t[i] = qact(meas_inv, vsub(pose_t[j], vmul(meas_t, scales[i])))
            visited[i] = True
            new_node = i
        n_visited += 1
        for key, next_edge_id in adjacency[new_node]:
            heapq.heappush(heap, (key, next_edge_id))

    pose_data = torch.zeros(n_nodes, 7, device=device, dtype=dtype)
    pose_data[:, 6] = 1
    pose_data[:, :3] = torch.as_tensor(pose_t, device=device, dtype=dtype)
    pose_data[:, 3:7] = torch.as_tensor(pose_q, device=device, dtype=dtype)
    return SE3(pose_data), torch.as_tensor(visited, device=device, dtype=torch.bool)


def _apply_rotation_delta(
    rotations: SO3,  # data: [N,4]
    delta: torch.Tensor,  # [A,3]
    anchor: int,
) -> SO3:
    """Apply rotation increments."""
    if delta.numel() == 0:
        return rotations

    rotation_data = rotations.data
    node_ids = torch.arange(rotation_data.shape[0], device=rotation_data.device)
    optimized_nodes = node_ids[node_ids != anchor]
    q_new = rotation_data.clone()
    q_new[optimized_nodes] = (SO3(rotation_data[optimized_nodes]) * SO3.exp(delta)).data
    return SO3(q_new)


def _apply_translation_scale_delta(
    poses: SE3,  # data: [N,7]
    log_s: torch.Tensor,  # [N]
    translation_delta: torch.Tensor,  # [A,3]
    scale_delta: torch.Tensor,  # [N]
    anchor: int,
):
    """Apply translation/scale increments."""
    if translation_delta.numel() == 0 and scale_delta.numel() == 0:
        return poses, log_s

    pose_data = poses.data
    node_ids = torch.arange(pose_data.shape[0], device=pose_data.device)
    optimized_nodes = node_ids[node_ids != anchor]
    pose_new = pose_data.clone()
    if translation_delta.numel() > 0:
        pose_new[optimized_nodes, :3] = (
            pose_data[optimized_nodes, :3]
            + SO3(pose_data[optimized_nodes, 3:7]).act(translation_delta)
        )
    return SE3(pose_new), log_s + scale_delta


def _translation_residuals(
    poses: SE3,
    log_s: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    rel_edges: RelativeEdges,
) -> torch.Tensor:
    """Translation-only edge residuals; returns (E, 3)."""
    pred_t = (poses[jj] * poses[ii].inv()).data[..., :3]
    pred_t = pred_t / torch.exp(log_s[ii]).clamp_min(EPS)[..., None]
    return pred_t - rel_edges.poses.data[..., :3]
