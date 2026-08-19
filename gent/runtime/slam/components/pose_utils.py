import torch
from lietorch import SE3


def multiply_scaled_se3(
    T_ij_meas: torch.Tensor,  # [N,7]
    T_j: torch.Tensor,  # [N,7]
    scale: torch.Tensor,  # [N]
) -> torch.Tensor:
    """Return inv(T_ij_scaled) * T_j for batched SE3 poses.

    GeNT pose edges store source-scale-normalized relative translations:
    T_ij_meas[:3] = translation(T_j * inv(T_i)) / scale_i. Pass scale_i to recover T_i.
    """
    T_ij_scaled = T_ij_meas.float().clone()
    T_ij_scaled[:, :3] = T_ij_scaled[:, :3] * scale.float()[:, None]
    return (SE3(T_ij_scaled).inv() * SE3(T_j.float())).data
