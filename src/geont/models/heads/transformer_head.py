from functools import partial
from typing import Callable
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ..dinov2.layers import Block, Attention, Mlp, DropPath, PositionGetter

class CrossAttention(nn.Module):
    def __init__(
        self, 
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        rope=None, 
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, qpos: torch.Tensor, kpos: torch.Tensor):
        B, _, C = query.shape

        q = self.q(query).reshape(B, -1, self.num_heads, C//self.num_heads).permute(0, 2, 1, 3)
        k = self.k(key).reshape(B, -1, self.num_heads, C//self.num_heads).permute(0, 2, 1, 3)
        v = self.v(value).reshape(B, -1, self.num_heads, C//self.num_heads).permute(0, 2, 1, 3)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None:
            q = self.rope(q, qpos)
            k = self.rope(k, kpos)
        x = nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool = False,
        rope=None,
        ln_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim, eps=ln_eps)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            rope=rope,
        )
        self.cross_attn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            rope=rope,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim, eps=ln_eps)
        self.norm3 = norm_layer(dim, eps=ln_eps)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.norm_y = norm_layer(dim, eps=ln_eps)

    def forward(self, x, y, xpos, ypos):
        y_ = self.norm_y(y)
        x = x + self.drop_path(self.cross_attn(self.norm1(x), y_, y_, xpos, ypos))
        x = x + self.drop_path(self.attn(self.norm2(x), xpos))
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x

        
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
            nn.Linear(in_dim, 2*dec_embed_dim),
            nn.GELU(),
            nn.Linear(2*dec_embed_dim, 4*dec_embed_dim),
        )
        self.fuse = nn.Linear(dec_embed_dim + in_dim // 2, dec_embed_dim)
        self.use_checkpoint = use_checkpoint
        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation

        self.blocks = nn.ModuleList([
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
            ) for _ in range(depth)
        ])

        self.linear_out = nn.Linear(dec_embed_dim, output_dim*(self.patch_size//2)**2)

        self.gate = nn.Sequential(
            nn.Linear(in_dim, dec_embed_dim),
            nn.GELU(),
            nn.Linear(dec_embed_dim, 1),
        )

        self.pos_getter = PositionGetter() if rope is not None else None

    def forward(self, aggregated_tokens_list, res_feat, img_shape):
        # only use the last layer's output
        aggregated_tokens = aggregated_tokens_list[-1]
        gates = torch.softmax(self.gate(aggregated_tokens[0]), dim=1)
        hidden = (aggregated_tokens[0] * gates).sum(dim=1)  # B,N,C
        hidden = self.projects(hidden)  # B,N,4C

        H, W = img_shape
        B = hidden.shape[0]
        hidden = hidden.transpose(-1, -2).view(B, -1, H//self.patch_size, W//self.patch_size)
        hidden = nn.functional.pixel_shuffle(hidden, 2)
        patch_size = self.patch_size // 2
        if self.pos_getter is not None:
            pos = self.pos_getter(
                B, H//patch_size, W//patch_size, device=hidden.device
            )
        else:
            pos = None
        hidden = hidden.flatten(2).transpose(-1, -2)  # B,N,C
        # fuse with residual features
        hidden = self.fuse(torch.cat([hidden, res_feat], dim=-1))
        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                hidden = checkpoint(blk, hidden, pos=pos, use_reentrant=False)
            else:
                hidden = blk(hidden, pos=pos)
        out = self.linear_out(hidden)
        out = out.transpose(-1, -2).view(B, -1, H//patch_size, W//patch_size)
        out = nn.functional.pixel_shuffle(out, patch_size)  # B,C,H,W
        out = out.permute(0, 2, 3, 1)  # B,H,W,C
        pred = self._apply_activation_single(out[..., :-1], activation=self.activation)
        conf = self._apply_activation_single(out[..., -1], activation=self.conf_activation)
        return pred.squeeze(-1), conf
    
    def _apply_activation_single(
        self, x: torch.Tensor, activation: str = "linear"
    ) -> torch.Tensor:
        """
        Apply activation to single channel output, maintaining semantic consistency with value branch in multi-channel case.
        Supports: exp / relu / sigmoid / softplus / tanh / linear / expp1
        """
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
        # Default linear
        return x

# class TransformerDecoder(nn.Module):
#     def __init__(
#         self,
#         in_dim,
#         patch_size,
#         output_dim=1,
#         activation: str = "inv_log",
#         dec_embed_dim=512,
#         depth=3,
#         dec_num_heads=8,
#         mlp_ratio=4,
#         rope=None,
#         use_checkpoint=False,
#     ):
#         super().__init__()

#         self.project = nn.Linear(in_dim, dec_embed_dim)
#         self.use_checkpoint = use_checkpoint
#         self.patch_size = patch_size
#         self.activation = activation

#         self.blocks = nn.ModuleList([
#             DecoderBlock(
#                 dim=dec_embed_dim,
#                 num_heads=dec_num_heads,
#                 mlp_ratio=mlp_ratio,
#                 qkv_bias=True,
#                 proj_bias=True,
#                 ffn_bias=True,
#                 drop_path=0.0,
#                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
#                 act_layer=nn.GELU,
#                 qk_norm=False,
#                 rope=rope,
#             ) for _ in range(depth)
#         ])

#         self.linear_out = nn.Linear(dec_embed_dim, output_dim*(self.patch_size//2)**2)

#         self.pos_getter = PositionGetter() if rope is not None else None

#     def forward(self, aggregated_tokens_list, query, img_shape):
#         # only use the last layer's output
#         aggregated_tokens = aggregated_tokens_list[-1][0]
#         H, W = img_shape
#         B, S, N, C = aggregated_tokens.shape
#         aggregated_tokens = self.project(aggregated_tokens.reshape(B, S*N, C))
#         device = aggregated_tokens.device

#         patch_size = self.patch_size // 2
#         if self.pos_getter is not None:
#             qpos = self.pos_getter(
#                 B, H//patch_size, W//patch_size, device=device
#             )
#             kpos = self.pos_getter(
#                 B*S, H//self.patch_size, W//self.patch_size, device=device
#             )
#             kpos = (kpos.view(B, -1, 2) * 2).to(kpos.dtype)  # scale up for cross attention
#         else:
#             qpos, kpos = None, None
#         x = query
#         for blk in self.blocks:
#             if self.use_checkpoint and self.training:
#                 x = checkpoint(blk, x, aggregated_tokens, qpos, kpos, use_reentrant=False)
#             else:
#                 x = blk(x, aggregated_tokens, qpos, kpos)
#         out = self.linear_out(x)
#         out = out.transpose(-1, -2).view(B, -1, H//patch_size, W//patch_size)
#         out = nn.functional.pixel_shuffle(out, patch_size)  # B,1,H,W
#         pred = self._apply_activation_single(out, activation=self.activation)
#         return pred
    
#     def _apply_activation_single(
#         self, x: torch.Tensor, activation: str = "linear"
#     ) -> torch.Tensor:
#         """
#         Apply activation to single channel output, maintaining semantic consistency with value branch in multi-channel case.
#         Supports: exp / relu / sigmoid / softplus / tanh / linear / expp1
#         """
#         act = activation.lower() if isinstance(activation, str) else activation
#         if act == "exp":
#             return torch.exp(x)
#         if act == "expm1":
#             return torch.expm1(x)
#         if act == "expp1":
#             return torch.exp(x) + 1
#         if act == "relu":
#             return torch.relu(x)
#         if act == "sigmoid":
#             return torch.sigmoid(x)
#         if act == "softplus":
#             return torch.nn.functional.softplus(x)
#         if act == "tanh":
#             return torch.tanh(x)
#         # Default linear
#         return x