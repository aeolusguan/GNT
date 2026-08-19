import torch
import torch.nn as nn

from .dinov2.layers import Block, Mlp


class CameraEnc(nn.Module):
    """Encode a source-centered camera set into backbone camera tokens."""

    def __init__(
        self,
        dim_out: int,
        trunk_depth: int = 4,
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
    ):
        super().__init__()
        self.pose_branch = Mlp(
            in_features=9,
            hidden_features=dim_out // 2,
            out_features=dim_out,
            drop=0,
        )
        self.token_norm = nn.LayerNorm(dim_out)
        self.trunk = nn.Sequential(
            *[
                Block(
                    dim=dim_out,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    init_values=init_values,
                    ln_eps=1e-5,
                )
                for _ in range(trunk_depth)
            ]
        )
        self.trunk_norm = nn.LayerNorm(dim_out)

    @staticmethod
    def pose_encoding(
        poses: torch.Tensor,  # [B,S,7], source-centered translation and quaternion xyzw
        intrinsics: torch.Tensor,  # [B,S,4], fx/fy/cx/cy
        image_shape: tuple[int, int],
    ) -> torch.Tensor:
        height, width = image_shape
        translation = poses[..., :3]
        quaternion = poses[..., 3:7]
        quaternion = quaternion / quaternion.norm(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-6)
        quaternion = torch.where(quaternion[..., 3:4] < 0, -quaternion, quaternion)
        fov_h = 2 * torch.atan((height / 2) / intrinsics[..., 1])
        fov_w = 2 * torch.atan((width / 2) / intrinsics[..., 0])
        return torch.cat((translation, quaternion, fov_h[..., None], fov_w[..., None]), dim=-1).float()

    def forward(
        self,
        poses: torch.Tensor,  # [B,S,7]
        intrinsics: torch.Tensor,  # [B,S,4]
        image_shape: tuple[int, int],
    ) -> torch.Tensor:
        tokens = self.pose_branch(self.pose_encoding(poses, intrinsics, image_shape))
        tokens = self.token_norm(tokens)
        tokens = self.trunk(tokens)
        return self.trunk_norm(tokens)
