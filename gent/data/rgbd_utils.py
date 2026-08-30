import numpy as np
import torch
from lietorch import SE3

import gent.geometry.projective_ops as pops


def compute_distance_matrix_flow(
    poses,
    disps,
    intrinsics,
    valid,
):
    """Compute symmetric flow distance for every pair in a trajectory."""
    poses = SE3(torch.from_numpy(poses).float().cuda()[None]).inv()
    disps = torch.from_numpy(disps).float().cuda()[None]
    intrinsics = torch.from_numpy(intrinsics).float().cuda()[None]
    valid = torch.from_numpy(valid).bool().cuda()[None]

    num_frames = poses.shape[1]
    ii, jj = torch.triu_indices(
        num_frames,
        num_frames,
        offset=1,
        device=disps.device,
    )

    max_flow = 100.0
    matrix = np.full((num_frames, num_frames), np.inf, dtype=np.float32)

    chunk_size = 2048
    for start in range(0, ii.shape[0], chunk_size):
        edge_ii = ii[start:start + chunk_size]
        edge_jj = jj[start:start + chunk_size]
        flow1, val1 = pops.induced_flow(poses, disps, intrinsics, edge_ii, edge_jj)
        flow2, val2 = pops.induced_flow(poses, disps, intrinsics, edge_jj, edge_ii)
        val1 = val1 * valid[:, edge_ii, :, :, None]
        val2 = val2 * valid[:, edge_jj, :, :, None]
        
        flow = torch.stack([flow1, flow2], dim=2)
        val = torch.stack([val1, val2], dim=2)
        
        mag = flow.norm(dim=-1).clamp(max=max_flow)
        mag = mag.view(mag.shape[1], -1)
        val = val.view(val.shape[1], -1)

        valid_ratio = val.mean(-1)
        mag = (mag * val).mean(-1) / valid_ratio.clamp_min(1e-8)
        mag[valid_ratio < 0.7] = torch.inf

        edge_ii = edge_ii.cpu().numpy()
        edge_jj = edge_jj.cpu().numpy()
        distance = mag.cpu().numpy()
        matrix[edge_ii, edge_jj] = distance
        matrix[edge_jj, edge_ii] = distance

    return matrix


def compute_sparse_distance_matrix_flow(
    poses,
    coords,
    depths,
    valid,
    intrinsics,
):
    """Compute the dense flow-distance semantics on sampled depth points."""
    poses = SE3(torch.from_numpy(poses).float().cuda()[None]).inv()
    coords = torch.from_numpy(coords).float().cuda()[None]
    depths = torch.from_numpy(depths).float().cuda()[None]
    valid = torch.from_numpy(valid).bool().cuda()[None]
    intrinsics = torch.from_numpy(intrinsics).float().cuda()[None]

    num_frames = poses.shape[1]
    ii, jj = torch.triu_indices(
        num_frames,
        num_frames,
        offset=1,
        device=depths.device,
    )
    max_flow = 1600.0
    matrix = np.full((num_frames, num_frames), np.inf, dtype=np.float32)

    def project(edge_ii, edge_jj):
        flow, projected_valid = pops.induced_flow_sparse(
            poses,
            coords,
            depths,
            intrinsics,
            edge_ii,
            edge_jj,
        )
        return flow, projected_valid.squeeze(-1).float()

    chunk_size = 2048
    for start in range(0, ii.shape[0], chunk_size):
        edge_ii = ii[start:start + chunk_size]
        edge_jj = jj[start:start + chunk_size]
        flow1, val1 = project(edge_ii, edge_jj)
        flow2, val2 = project(edge_jj, edge_ii)
        val1 = val1 * valid[:, edge_ii]
        val2 = val2 * valid[:, edge_jj]

        flow = torch.stack([flow1, flow2], dim=2)
        val = torch.stack([val1, val2], dim=2)

        mag = flow.norm(dim=-1).clamp(max=max_flow)
        mag = mag.view(mag.shape[1], -1)
        val = val.view(val.shape[1], -1)

        valid_ratio = val.mean(-1)
        mag = (mag * val).mean(-1) / valid_ratio.clamp_min(1e-8)
        mag[valid_ratio < 0.7] = torch.inf

        edge_ii = edge_ii.cpu().numpy()
        edge_jj = edge_jj.cpu().numpy()
        distance = mag.cpu().numpy()
        matrix[edge_ii, edge_jj] = distance
        matrix[edge_jj, edge_ii] = distance

    return matrix
