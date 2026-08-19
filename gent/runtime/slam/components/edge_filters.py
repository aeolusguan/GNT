from dataclasses import replace

import torch

from .measurements import PoseMeasurement


def relative_pose_rotation_angle_deg(pose: torch.Tensor) -> torch.Tensor:
    q = pose[..., 3:7]
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    sin_half = q[..., :3].norm(dim=-1)
    cos_half = q[..., 3].abs()
    return 2.0 * torch.atan2(sin_half, cos_half) * (180.0 / torch.pi)


def relative_pose_translation_norm(pose: torch.Tensor) -> torch.Tensor:
    return pose[..., :3].norm(dim=-1)


def filter_local_edge_measurement(
    result: PoseMeasurement,
    *,
    max_rotation_deg: float,
    max_translation: float,
    trans_conf_thresh: float,
) -> PoseMeasurement:
    ii = result.ii
    jj = result.jj
    pose = result.relative_pose
    confidence = result.confidence
    n_edges = int(ii.numel())

    if n_edges == 0:
        keep_idx = torch.as_tensor([], dtype=torch.long, device=ii.device)
    else:
        rotation_deg = relative_pose_rotation_angle_deg(pose)
        translation_norm = relative_pose_translation_norm(pose)
        keep_rotation = rotation_deg <= float(max_rotation_deg)
        keep_translation = translation_norm <= float(max_translation)
        keep_pose = keep_rotation & keep_translation

        pose_idx = torch.nonzero(keep_pose, as_tuple=False).flatten()
        if pose_idx.numel() == 0:
            keep_idx = pose_idx
        else:
            trans_conf = confidence[pose_idx, 0]
            keep_conf = trans_conf >= float(trans_conf_thresh)
            keep_idx = pose_idx[keep_conf]

    return replace(
        result,
        ii=ii[keep_idx],
        jj=jj[keep_idx],
        relative_pose=pose[keep_idx],
        relative_log_scale=result.relative_log_scale[keep_idx],
        confidence=confidence[keep_idx],
    )
