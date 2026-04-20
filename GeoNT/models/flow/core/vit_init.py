import torch
import torch.nn as nn
import timm
from timm.layers import Mlp
from einops import rearrange

from .layer import resconv, conv1x1

class TinyDPT(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.project = conv1x1(in_channels, out_channels)
        self.resize_layer = nn.ConvTranspose2d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=2, stride=2, padding=0
        )
        self.resConfUnit = resconv(out_channels, out_channels)
        self.out_conv = conv1x1(out_channels, out_channels)

        self.res_conv = resconv(out_channels, out_channels)
        self.merge = Mlp(2*out_channels, out_channels, out_channels, use_conv=True)

    def forward(self, hidden, x):
        hidden = self.resize_layer(self.project(hidden))
        hidden = self.out_conv(self.resConfUnit(hidden))

        res_x = self.res_conv(x)
        return self.merge(torch.cat([hidden, res_x], dim=1))

class ViTInit(nn.Module):
    def __init__(self, model_name, input_dim):
        super().__init__()
        self.configs = {
            "vitl": {"encoder": "vit_large_patch16_224", "n_layers": 24, "dim": 1024},
            "vitb": {"encoder": "vit_base_patch16_224", "n_layers": 24, "dim": 768},
            "vits": {"encoder": "vit_small_patch16_224", "n_layers": 12, "dim": 384},
            'vitt': {"encoder": "vit_tiny_patch16_224", "n_layers": 12, "dim": 192}
        }
        self.dim = self.configs[model_name]["dim"]
        vit = timm.create_model(
            self.configs[model_name]["encoder"],
            pretrained=True
        )
        self.blks = vit.blocks
        self.patch_embed = resconv(input_dim, self.dim // 4 * 3, stride=2)
        base_dim = self.dim - self.dim // 4 * 3
        grow_factor = (base_dim / 16.0) ** 0.25
        self.base_dims = [16] + [int(16 * grow_factor ** i) // 8 * 8 for i in range(1, 5)]
        self.patch_embed_base = nn.Sequential(
            *[resconv(self.base_dims[i], self.base_dims[i+1], stride=2) for i in range(4)]
        )
        
        self.hidden = TinyDPT(self.dim, input_dim)
        self.net = TinyDPT(self.dim, input_dim)

    def forward(self, x, bases):
        vit_x = self.patch_embed(x)
        vit_bases = self.patch_embed_base(bases)
        vit_x = torch.cat((vit_x, vit_bases), dim=1)

        h, w = vit_x.shape[-2:]
        vit_x = rearrange(vit_x, 'b c h w -> b (h w) c')
        for blk in self.blks:
            vit_x = blk(vit_x)
        hidden = rearrange(vit_x, 'b (h w) c -> b c h w', h=h, w=w)
        
        return self.hidden(hidden, x), self.net(hidden, x)