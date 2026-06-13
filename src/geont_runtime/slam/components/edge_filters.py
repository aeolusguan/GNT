import torch


def make_local_edge_outlier_info(trans_conf_thresh: float) -> dict:
    return {
        "local_edge_outlier_trans_conf_thresh": float(trans_conf_thresh),
        "local_edge_outlier_candidates": 0,
        "local_edge_outlier_accepted": 0,
        "local_edge_outlier_dropped": 0,
        "local_edge_outlier_duplicate_filtered_after_gate": 0,
    }


def filter_edges_by_confidence(
    ii: torch.Tensor,
    jj: torch.Tensor,
    pose: torch.Tensor,
    confidence: torch.Tensor,
    trans_conf_thresh: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    info = make_local_edge_outlier_info(trans_conf_thresh)
    info["local_edge_outlier_candidates"] = int(ii.numel())
    if ii.numel() == 0:
        return ii, jj, pose, confidence, info

    trans_conf = confidence[:, 0]
    keep = trans_conf >= float(trans_conf_thresh)
    counts = torch.stack((keep.sum(), (~keep).sum())).cpu().tolist()
    info["local_edge_outlier_accepted"] = int(counts[0])
    info["local_edge_outlier_dropped"] = int(counts[1])
    return ii[keep], jj[keep], pose[keep], confidence[keep], info


def relative_pose_rotation_angle_deg(pose: torch.Tensor) -> torch.Tensor:
    q = pose[..., 3:7]
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    sin_half = q[..., :3].norm(dim=-1)
    cos_half = q[..., 3].abs()
    return 2.0 * torch.atan2(sin_half, cos_half) * (180.0 / torch.pi)


def relative_pose_translation_norm(pose: torch.Tensor) -> torch.Tensor:
    return pose[..., :3].norm(dim=-1)


def filter_edges_by_pose_magnitude(
    ii: torch.Tensor,
    jj: torch.Tensor,
    pose: torch.Tensor,
    confidence: torch.Tensor,
    max_rotation_deg: float,
    max_translation: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    info = {
        "pose_gate_max_rotation_deg": float(max_rotation_deg),
        "pose_gate_max_translation": float(max_translation),
        "pose_gate_candidates": int(ii.numel()),
        "pose_gate_accepted": 0,
        "pose_gate_dropped_rotation": 0,
        "pose_gate_dropped_translation": 0,
    }
    if ii.numel() == 0:
        return ii, jj, pose, confidence, info

    rotation_deg = relative_pose_rotation_angle_deg(pose)
    translation_norm = relative_pose_translation_norm(pose)
    keep_rotation = rotation_deg <= float(max_rotation_deg)
    keep_translation = translation_norm <= float(max_translation)
    keep = keep_rotation & keep_translation
    info["pose_gate_accepted"] = int(keep.sum().item())
    info["pose_gate_dropped_rotation"] = int((~keep_rotation).sum().item())
    info["pose_gate_dropped_translation"] = int((keep_rotation & ~keep_translation).sum().item())
    return ii[keep], jj[keep], pose[keep], confidence[keep], info


def filter_local_edge_measurement(
    result: dict,
    *,
    max_rotation_deg: float,
    max_translation: float,
    trans_conf_thresh: float,
) -> dict:
    ii = result["ii"]
    jj = result["jj"]
    pose = result["relative_pose"]
    confidence = result["confidence"]
    n_edges = int(ii.numel())

    pose_gate_info = {
        "pose_gate_max_rotation_deg": float(max_rotation_deg),
        "pose_gate_max_translation": float(max_translation),
        "pose_gate_candidates": n_edges,
        "pose_gate_accepted": 0,
        "pose_gate_dropped_rotation": 0,
        "pose_gate_dropped_translation": 0,
    }
    outlier_info = make_local_edge_outlier_info(trans_conf_thresh)
    if n_edges == 0:
        keep_idx = torch.as_tensor([], dtype=torch.long, device=ii.device)
    else:
        rotation_deg = relative_pose_rotation_angle_deg(pose)
        translation_norm = relative_pose_translation_norm(pose)
        keep_rotation = rotation_deg <= float(max_rotation_deg)
        keep_translation = translation_norm <= float(max_translation)
        keep_pose = keep_rotation & keep_translation
        pose_gate_info["pose_gate_accepted"] = int(keep_pose.sum().item())
        pose_gate_info["pose_gate_dropped_rotation"] = int((~keep_rotation).sum().item())
        pose_gate_info["pose_gate_dropped_translation"] = int((keep_rotation & ~keep_translation).sum().item())

        pose_idx = torch.nonzero(keep_pose, as_tuple=False).flatten()
        outlier_info["local_edge_outlier_candidates"] = int(pose_idx.numel())
        if pose_idx.numel() == 0:
            keep_idx = pose_idx
        else:
            trans_conf = confidence[pose_idx, 0]
            keep_conf = trans_conf >= float(trans_conf_thresh)
            outlier_info["local_edge_outlier_accepted"] = int(keep_conf.sum().item())
            outlier_info["local_edge_outlier_dropped"] = int((~keep_conf).sum().item())
            keep_idx = pose_idx[keep_conf]

    filtered = dict(result)
    for key, value in result.items():
        if isinstance(value, torch.Tensor) and value.shape[:1] == (n_edges,):
            filtered[key] = value[keep_idx]
    filtered["local_edge_outlier_info"] = dict(outlier_info)
    filtered["pose_gate_info"] = dict(pose_gate_info)
    return filtered


def record_local_edge_outlier_info(
    accumulator: dict,
    info: dict,
    duplicate_filtered_after_gate: int,
) -> None:
    for key, value in info.items():
        if key in accumulator and isinstance(value, int):
            accumulator[key] += int(value)
        else:
            accumulator[key] = value
    accumulator["local_edge_outlier_duplicate_filtered_after_gate"] += int(duplicate_filtered_after_gate)
