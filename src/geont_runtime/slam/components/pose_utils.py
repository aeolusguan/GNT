import torch
from lietorch import SE3


def multiply_scaled_se3(
    T_ij_meas: torch.Tensor,
    T_j: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Return inv(T_ij_scaled) * T_j for batched SE3 poses.

    Shape contract: T_ij_meas is (N, 7), T_j is (N, 7), scale is (N,), and
    the returned T_i is (N, 7). T_ij_scaled uses T_ij_meas rotation and
    scale[:, None] * T_ij_meas[:, :3].

    GeoNT pose edges store source-scale-normalized relative translations:
    T_ij_meas[:3] = translation(T_j * inv(T_i)) / scale_i. Pass scale_i to recover T_i.
    """
    assert T_ij_meas.ndim == 2 and T_ij_meas.shape[1] == 7
    assert T_j.shape == T_ij_meas.shape
    assert scale.shape == (T_ij_meas.shape[0],)
    T_ij_scaled = T_ij_meas.float().clone()
    T_ij_scaled[:, :3] = T_ij_scaled[:, :3] * scale.float()[:, None]
    return (SE3(T_ij_scaled).inv() * SE3(T_j.float())).data
