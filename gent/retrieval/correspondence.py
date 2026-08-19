from __future__ import annotations

import torch
from lietorch import SE3

from gent.geometry.projective_ops import projective_transform


def patch_correspondences_from_geometry(
    poses: torch.Tensor,  # [S,7]
    depths: torch.Tensor,  # [S,H,W]
    valid: torch.Tensor,  # [S,H,W]
    intrinsics: torch.Tensor,  # [S,4]
    *,
    source: int,
    target: int,
    fmap_height: int,
    fmap_width: int,
    min_depth: float = 0.2,
    depth_consistency: float = 0.1,
    max_correspondences: int = -1,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project source fmap patch centers into target fmap patch indices."""

    device = depths.device
    source = int(source)
    target = int(target)
    image_height, image_width = int(depths.shape[-2]), int(depths.shape[-1])
    fmap_height = int(fmap_height)
    fmap_width = int(fmap_width)
    assert fmap_height > 0 and fmap_width > 0

    depth_valid = valid.bool() & torch.isfinite(depths) & (depths > min_depth)
    disps = torch.zeros_like(depths, dtype=torch.float32)
    disps[depth_valid] = 1.0 / depths.float()[depth_valid]

    ii = torch.as_tensor([source], device=device, dtype=torch.long)
    jj = torch.as_tensor([target], device=device, dtype=torch.long)
    coords, projection_valid = projective_transform(
        SE3(poses[None].float()),
        disps[None],
        intrinsics[None].float(),
        ii,
        jj,
        return_depth=True,
    )

    ys = ((torch.arange(fmap_height, device=device, dtype=torch.float32) + 0.5) * image_height / fmap_height - 0.5)
    xs = ((torch.arange(fmap_width, device=device, dtype=torch.float32) + 0.5) * image_width / fmap_width - 0.5)
    py, px = torch.meshgrid(ys, xs, indexing="ij")
    src_y = py.round().long().clamp(0, image_height - 1)
    src_x = px.round().long().clamp(0, image_width - 1)

    target_x = coords[0, 0, src_y, src_x, 0].float().reshape(-1)
    target_y = coords[0, 0, src_y, src_x, 1].float().reshape(-1)
    target_disp = coords[0, 0, src_y, src_x, 2].float().reshape(-1)
    target_px = torch.floor(target_x * fmap_width / image_width).long()
    target_py = torch.floor(target_y * fmap_height / image_height).long()

    projected_valid = projection_valid[0, 0, src_y, src_x, 0].bool().reshape(-1)
    source_valid = depth_valid[source, src_y, src_x].reshape(-1)
    in_bounds = (
        (target_disp > 0.0)
        & (target_x >= 0.0)
        & (target_x < image_width)
        & (target_y >= 0.0)
        & (target_y < image_height)
        & (target_px >= 0)
        & (target_px < fmap_width)
        & (target_py >= 0)
        & (target_py < fmap_height)
    )

    nearest_x = target_x.round().long().clamp(0, image_width - 1)
    nearest_y = target_y.round().long().clamp(0, image_height - 1)
    projected_depth = 1.0 / target_disp.clamp_min(1.0e-8)
    target_depth = depths[target, nearest_y, nearest_x].float()
    target_valid = depth_valid[target, nearest_y, nearest_x]
    keep = source_valid & projected_valid & in_bounds & target_valid
    if depth_consistency > 0.0:
        denom = torch.maximum(target_depth, projected_depth).clamp_min(1.0e-6)
        keep = keep & ((target_depth - projected_depth).abs() / denom <= float(depth_consistency))

    source_linear = torch.arange(fmap_height * fmap_width, device=device, dtype=torch.long)
    target_linear = target_py * fmap_width + target_px
    source_linear = source_linear[keep]
    target_linear = target_linear[keep]

    if max_correspondences > 0 and source_linear.numel() > int(max_correspondences):
        order = torch.randperm(source_linear.numel(), device=device, generator=generator)[: int(max_correspondences)]
        source_linear = source_linear[order]
        target_linear = target_linear[order]
    return source_linear.contiguous(), target_linear.contiguous()
