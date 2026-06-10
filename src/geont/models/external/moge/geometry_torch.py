from typing import *

import torch
import torch.nn.functional as F

from .geometry_numpy import solve_optimal_focal_shift, solve_optimal_shift
from .helpers import totensor, batched


@totensor(_others=torch.float32)
@batched(_others=0)
def intrinsics_from_focal_center(
    fx: Union[float, torch.Tensor],
    fy: Union[float, torch.Tensor],
    cx: Union[float, torch.Tensor],
    cy: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Get OpenCV intrinsics matrix

    ## Parameters
        focal_x (float | Tensor): focal length in x axis
        focal_y (float | Tensor): focal length in y axis
        cx (float | Tensor): principal point in x axis
        cy (float | Tensor): principal point in y axis

    ## Returns
        (Tensor): [..., 3, 3] OpenCV intrinsics matrix
    """
    zeros, ones = torch.zeros_like(fx), torch.ones_like(fx)
    ret = torch.stack([
        fx, zeros, cx, 
        zeros, fy, cy, 
        zeros, zeros, ones
    ], dim=-1).unflatten(-1, (3, 3))
    return ret


def depth_map_to_point_map(depth: torch.Tensor, intrinsics: torch.Tensor, extrinsics: torch.Tensor = None):
    height, width = depth.shape[-2:]
    uv = uv_map(height, width, dtype=depth.dtype, device=depth.device)
    pts = unproject_cv(uv, depth, intrinsics=intrinsics[..., None, :, :], extrinsics=extrinsics[..., None, :, :] if extrinsics is not None else None)
    return pts


def uv_map(
    *size: Union[int, Tuple[int, int]],
    top: float = 0.,
    left: float = 0.,
    bottom: float = 1.,
    right: float = 1.,
    dtype: torch.dtype = torch.float32,
    device: torch.device = None
) -> torch.Tensor:
    """
    Get image UV space coordinate map, where (0., 0.) is the top-left corner of the image, and (1., 1.) is the bottom-right corner of the image.
    This is commonly used as normalized image coordinates in texture mapping (when image is not flipped vertically).

    ## Parameters
    - `*size`: `Tuple[int, int]` or two integers of map size `(height, width)`
    - `top`: `float`, optional top boundary in uv space. Defaults to 0.
    - `left`: `float`, optional left boundary in uv space. Defaults to 0.
    - `bottom`: `float`, optional bottom boundary in uv space. Defaults to 1.
    - `right`: `float`, optional right boundary in uv space. Defaults to 1.
    - `dtype`: `np.dtype`, optional data type of the output uv map. Defaults to torch.float32.
    - `device`: `torch.device`, optional device of the output uv map. Defaults to None.

    ## Returns
    - `uv (Tensor)`: shape `(height, width, 2)`

    ## Example Usage

    >>> uv_map(10, 10):
    [[[0.05, 0.05], [0.15, 0.05], ..., [0.95, 0.05]],
     [[0.05, 0.15], [0.15, 0.15], ..., [0.95, 0.15]],
      ...             ...                  ...
     [[0.05, 0.95], [0.15, 0.95], ..., [0.95, 0.95]]]
    """
    if len(size) == 1 and isinstance(size[0], tuple):
        height, width = size[0]
    else:
        height, width = size
    u = torch.linspace(left + 0.5 / width, right - 0.5 / width, width, dtype=dtype, device=device)
    v = torch.linspace(top + 0.5 / height, bottom - 0.5 / height, height, dtype=dtype, device=device)
    u, v = torch.meshgrid(u, v, indexing='xy')
    return torch.stack([u, v], dim=2)


def unproject_cv(
    uv: torch.Tensor,
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor = None,
) -> torch.Tensor:
    """
    Unproject uv coordinates to 3D view space following the OpenCV convention

    ## Parameters
        uv (Tensor): [..., N, 2] uv coordinates, value ranging in [0, 1].
            The origin (0., 0.) is corresponding to the left & top
        depth (Tensor): [..., N] depth value
        extrinsics (Tensor): [..., 4, 4] extrinsics matrix
        intrinsics (Tensor): [..., 3, 3] intrinsics matrix

    ## Returns
        points (Tensor): [..., N, 3] 3d points
    """
    intrinsics = torch.cat([
        torch.cat([intrinsics, torch.zeros((*intrinsics.shape[:-2], 3, 1), dtype=intrinsics.dtype, device=intrinsics.device)], dim=-1),
        torch.tensor([[0, 0, 0, 1]], dtype=intrinsics.dtype, device=intrinsics.device).expand(*intrinsics.shape[:-2], 1, 4)
    ], dim=-2)
    transform = intrinsics @ extrinsics if extrinsics is not None else intrinsics
    points = torch.cat([uv, torch.ones((*uv.shape[:-1], 1), dtype=uv.dtype, device=uv.device)], dim=-1) * depth[..., None]
    points = torch.cat([points, torch.ones((*points.shape[:-1], 1), dtype=uv.dtype, device=uv.device)], dim=-1)
    points = points @ torch.linalg.inv(transform).mT
    points = points[..., :3]
    return points


def angle_diff_vec3(v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-12):
    return torch.atan2(torch.cross(v1, v2, dim=-1).norm(dim=-1) + eps, (v1 * v2).sum(dim=-1))


def normalized_view_plane_uv(width: int, height: int, aspect_ratio: float = None, dtype: torch.dtype = None, device: torch.device = None) -> torch.Tensor:
    "UV with left-top corner as (-width / diagonal, -height / diagonal) and right-bottom corner as (width / diagonal, height / diagonal)"
    if aspect_ratio is None:
        aspect_ratio = width / height
    
    span_x = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5
    span_y = 1 / (1 + aspect_ratio ** 2) ** 0.5

    u = torch.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=dtype, device=device)
    v = torch.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=dtype, device=device)
    u, v = torch.meshgrid(u, v, indexing='xy')
    uv = torch.stack([u, v], dim=-1)
    return uv


def recover_focal_shift(points: torch.Tensor, mask: torch.Tensor = None, focal: torch.Tensor = None, downsample_size: Tuple[int, int] = (64, 64)):
    """
    Recover the depth map and FoV from a point map with unknown z shift and focal.

    Note that it assumes:
    - the optical center is at the center of the map
    - the map is undistorted
    - the map is isometric in the x and y directions

    ### Parameters:
    - `points: torch.Tensor` of shape (..., H, W, 3)
    - `downsample_size: Tuple[int, int]` in (height, width), the size of the downsampled map. Downsampling produces approximate solution and is efficient for large maps.

    ### Returns:
    - `focal`: torch.Tensor of shape (...) the estimated focal length, relative to the half diagonal of the map
    - `shift`: torch.Tensor of shape (...) Z-axis shift to translate the point map to camera space
    """
    shape = points.shape
    height, width = points.shape[-3], points.shape[-2]
    diagonal = (height ** 2 + width ** 2) ** 0.5

    points = points.reshape(-1, *shape[-3:])
    mask = None if mask is None else mask.reshape(-1, *shape[-3:-1])
    focal = focal.reshape(-1) if focal is not None else None
    uv = normalized_view_plane_uv(width, height, dtype=points.dtype, device=points.device)  # (H, W, 2)

    points_lr = F.interpolate(points.permute(0, 3, 1, 2), downsample_size, mode='nearest').permute(0, 2, 3, 1)
    uv_lr = F.interpolate(uv.unsqueeze(0).permute(0, 3, 1, 2), downsample_size, mode='nearest').squeeze(0).permute(1, 2, 0)
    mask_lr = None if mask is None else F.interpolate(mask.to(torch.float32).unsqueeze(1), downsample_size, mode='nearest').squeeze(1) > 0
    
    uv_lr_np = uv_lr.cpu().numpy()
    points_lr_np = points_lr.detach().cpu().numpy()
    focal_np = focal.cpu().numpy() if focal is not None else None
    mask_lr_np = None if mask is None else mask_lr.cpu().numpy()
    optim_shift, optim_focal = [], []
    for i in range(points.shape[0]):
        points_lr_i_np = points_lr_np[i] if mask is None else points_lr_np[i][mask_lr_np[i]]
        uv_lr_i_np = uv_lr_np if mask is None else uv_lr_np[mask_lr_np[i]]
        if uv_lr_i_np.shape[0] < 2:
            optim_focal.append(1)
            optim_shift.append(0)
            continue
        if focal is None:
            optim_shift_i, optim_focal_i = solve_optimal_focal_shift(uv_lr_i_np, points_lr_i_np)
            optim_focal.append(float(optim_focal_i))
        else:
            optim_shift_i = solve_optimal_shift(uv_lr_i_np, points_lr_i_np, focal_np[i])
        optim_shift.append(float(optim_shift_i))
    optim_shift = torch.tensor(optim_shift, device=points.device, dtype=points.dtype).reshape(shape[:-3])

    if focal is None:
        optim_focal = torch.tensor(optim_focal, device=points.device, dtype=points.dtype).reshape(shape[:-3])
    else:
        optim_focal = focal.reshape(shape[:-3])

    return optim_focal, optim_shift