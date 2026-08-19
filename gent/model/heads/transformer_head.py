from functools import partial

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ..dinov2.layers import Block, PositionGetter


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        patch_size,
        output_dim=1,
        activation: str = "inv_log",
        conf_activation: str = "expp1",
        dec_embed_dim=512,
        depth=5,
        dec_num_heads=8,
        mlp_ratio=4,
        rope=None,
        use_checkpoint=False,
    ):
        super().__init__()

        self.projects = nn.Sequential(
            nn.Linear(in_dim, 2 * dec_embed_dim),
            nn.GELU(),
            nn.Linear(2 * dec_embed_dim, 4 * dec_embed_dim),
        )
        self.fuse = nn.Linear(dec_embed_dim + in_dim // 2, dec_embed_dim)
        self.use_checkpoint = use_checkpoint
        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=dec_embed_dim,
                    num_heads=dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    drop_path=0.0,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    act_layer=nn.GELU,
                    init_values=None,
                    qk_norm=False,
                    rope=rope,
                )
                for _ in range(depth)
            ]
        )

        self.linear_out = nn.Linear(dec_embed_dim, output_dim * (self.patch_size // 2) ** 2)
        self.pos_getter = PositionGetter() if rope is not None else None

    def forward(self, aggregated_tokens_list, res_feat, img_shape):
        # Depth is decoded from the reference stream only.
        reference_tokens = aggregated_tokens_list[-1][0][:, 0]
        hidden = self.projects(reference_tokens)

        H, W = img_shape
        B = hidden.shape[0]
        hidden = hidden.transpose(-1, -2).view(B, -1, H // self.patch_size, W // self.patch_size)
        hidden = nn.functional.pixel_shuffle(hidden, 2)
        patch_size = self.patch_size // 2
        if self.pos_getter is not None:
            pos = self.pos_getter(B, H // patch_size, W // patch_size, device=hidden.device)
        else:
            pos = None
        hidden = hidden.flatten(2).transpose(-1, -2)
        hidden = self.fuse(torch.cat([hidden, res_feat], dim=-1))
        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                hidden = checkpoint(blk, hidden, pos=pos, use_reentrant=False)
            else:
                hidden = blk(hidden, pos=pos)
        out = self.linear_out(hidden)
        out = out.transpose(-1, -2).view(B, -1, H // patch_size, W // patch_size)
        out = nn.functional.pixel_shuffle(out, patch_size)
        out = out.permute(0, 2, 3, 1)
        pred = self._apply_activation_single(out[..., :-1], activation=self.activation)
        conf = self._apply_activation_single(out[..., -1], activation=self.conf_activation)
        return pred.squeeze(-1), conf

    def _apply_activation_single(self, x: torch.Tensor, activation: str = "linear") -> torch.Tensor:
        act = activation.lower() if isinstance(activation, str) else activation
        if act == "exp":
            return torch.exp(x)
        if act == "expm1":
            return torch.expm1(x)
        if act == "expp1":
            return torch.exp(x) + 1
        if act == "relu":
            return torch.relu(x)
        if act == "sigmoid":
            return torch.sigmoid(x)
        if act == "softplus":
            return torch.nn.functional.softplus(x)
        if act == "tanh":
            return torch.tanh(x)
        return x
