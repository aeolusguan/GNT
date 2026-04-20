# This file includes code from SEA-RAFT (https://github.com/princeton-vl/SEA-RAFT)
# and flowseek (https://github.com/mattpoggi/flowseek)
# Copyright (c) 2024, Princeton Vision & Learning Lab
# Licensed under the BSD 3-Clause License

import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import Mlp
import einops

from .update import BasicUpdateBlock
from .corr import CorrBlock
from .utils.utils import coords_grid, InputPadder
from .extractor import ResNetFPN
from .vit_init import ViTInit

class FlowModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.args.corr_levels = 4
        self.args.corr_radius = args.radius
        self.args.corr_channel = args.corr_levels * (args.radius * 2 + 1) ** 2

        self.merge_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
        )

        self.fnet = ResNetFPN(args, input_dim=3, output_dim=self.args.dim, norm_layer=nn.BatchNorm2d, init_weight=True)
        hidden_dim = 64
        self.init_decoder = ViTInit(model_name='vits', input_dim=hidden_dim)
        self.init_proj = Mlp(2*(args.dim + 128), hidden_dim, hidden_dim, use_conv=True)
        self.init_pred_head = Mlp(hidden_dim, hidden_dim, 8*8*2, use_conv=True)
        self.net_init = nn.Conv2d(hidden_dim, args.dim, 1, 1, 0)

        self.upsample_weight = nn.Sequential(
            # convex combination of 3x3 patches
            nn.Conv2d(args.dim, args.dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(args.dim * 2, 64 * 9, 1, padding=0)
        )
        self.flow_head = nn.Sequential(
            # flow (2) + weight (2) + log_b(2)
            nn.Conv2d(args.dim, args.dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(args.dim * 2, 6, 3, padding=1)
        )
        if args.iters > 0:
            self.update_block = BasicUpdateBlock(args, hdim=args.dim, cdim=args.dim)

    def create_bases(self, disp):
        B, C, H, W = disp.shape
        device = disp.device
        assert C == 1
        cx = 0.5
        cy = 0.5

        ys = torch.linspace(0.5 / H, 1.0 - 0.5 / H, H)
        xs = torch.linspace(0.5 / W, 1.0 - 0.5 / W, W)
        u, v = torch.meshgrid(xs, ys, indexing='xy')
        u = u - cx
        v = v - cy
        u = u.unsqueeze(0).unsqueeze(0)
        v = v.unsqueeze(0).unsqueeze(0)
        u = u.repeat(B, 1, 1, 1).to(device)
        v = v.repeat(B, 1, 1, 1).to(device)
        
        aspect_ratio = W / H
        u = u * aspect_ratio

        Tx = torch.cat([-torch.ones_like(disp), torch.zeros_like(disp)], dim=1)
        Ty = torch.cat([torch.zeros_like(disp), -torch.ones_like(disp)], dim=1)
        Tz = torch.cat([u, v], dim=1)

        Tx = Tx / torch.linalg.vector_norm(Tx, dim=(1,2,3), keepdim=True)
        Ty = Ty / torch.linalg.vector_norm(Ty, dim=(1,2,3), keepdim=True)
        Tz = Tz / torch.linalg.vector_norm(Tz, dim=(1,2,3), keepdim=True)

        Tx = 2 * disp * Tx
        Ty = 2 * disp * Ty
        Tz = 2 * disp * Tz

        R1x = torch.cat([torch.zeros_like(disp), torch.ones_like(disp)], dim=1)
        R2x = torch.cat([u * v, v * v], dim=1)
        R1y = torch.cat([-torch.ones_like(disp), torch.zeros_like(disp)], dim=1)
        R2y = torch.cat([-u * u, -u * v], dim=1)
        Rz =  torch.cat([-v, u], dim=1)

        R1x = R1x / torch.linalg.vector_norm(R1x, dim=(1,2,3), keepdim=True)
        R2x = R2x / torch.linalg.vector_norm(R2x, dim=(1,2,3), keepdim=True)
        R1y = R1y / torch.linalg.vector_norm(R1y, dim=(1,2,3), keepdim=True)
        R2y = R2y / torch.linalg.vector_norm(R2y, dim=(1,2,3), keepdim=True)
        Rz =  Rz  / torch.linalg.vector_norm(Rz,  dim=(1,2,3), keepdim=True)
        
        M = torch.cat([Tx, Ty, Tz, R1x, R2x, R1y, R2y, Rz], dim=1) # Bx(8x2)xHxW
        return M
    
    def upsample_data(self, flow, info, mask):
        """ Upsample [H/8, W/8, C] -> [H, W, C] using convex combination """
        N, C, H, W = info.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
        up_info = F.unfold(info, [3,3], padding=1)
        up_info = up_info.view(N, C, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        up_info = torch.sum(mask * up_info, dim=2)
        up_info = up_info.permute(0, 1, 4, 2, 5, 3)

        return up_flow.reshape(N, 2, 8*H, 8*W), up_info.reshape(N, C, 8*H, 8*W)
    
    def forward(self, image1, image2, disp, mono1, mono2, iters=None, flow_gt=None, test_mode=False):
        """ Estimate optical flow between pair of frames """
        if iters is None:
            iters = self.args.iters
        
        bases = self.create_bases(disp.unsqueeze(1))

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0
        image1 = image1.contiguous()
        image2 = image2.contiguous()

        # padding
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        bases = padder.pad(bases)[0]
        mono1, mono2 = padder.pad(mono1, mono2)
        N, _, H, W = image1.shape
        dilation = torch.ones(N, 1, H//8, W//8, device=image1.device)

        # run the feature network
        fmap1_8x = self.fnet(image1)
        fmap2_8x = self.fnet(image2)
        

        mono1 = self.merge_head(mono1)
        mono2 = self.merge_head(mono2)

        fmap1_8x = torch.cat((fmap1_8x, mono1), 1)
        fmap2_8x = torch.cat((fmap2_8x, mono2), 1)

        # Initialization
        x = self.init_proj(torch.cat([fmap1_8x, fmap2_8x], dim=1))
        x, net = self.init_decoder(x, bases)
        init = self.init_pred_head(x)
        net = self.net_init(net)
        init = einops.rearrange(init, 'b (c sh sw) h w -> b c (sh sw) h w', sh=8, sw=8)
        flow_8x = torch.median(init, dim=2, keepdim=False)[0]
        init = einops.rearrange(init, 'b c (sh sw) h w -> b c (h sh) (w sw)', sh=8, sw=8)
        

        if iters > 0:
            corr_fn = CorrBlock(fmap1_8x, fmap2_8x, self.args)

        flow_predictions = []
        info_predictions = []
        for itr in range(iters):
            N, _, H, W = flow_8x.shape
            flow_8x = flow_8x.detach()
            coords2 = coords_grid(N, H, W, device=image1.device) + flow_8x
            corr = corr_fn(coords2, dilation=dilation)
            net = self.update_block(net, corr, flow_8x)
            flow_update = self.flow_head(net)
            weight_update = .25 * self.upsample_weight(net)
            flow_8x = flow_8x + flow_update[:, :2]
            info_8x = flow_update[:, 2:]
            # upsample predictions
            flow_up, info_up = self.upsample_data(flow_8x, info_8x, weight_update)
            flow_predictions.append(flow_up)
            info_predictions.append(info_up)
        flow_predictions = [padder.unpad(flow) for flow in flow_predictions]
        info_predictions = [padder.unpad(info) for info in info_predictions]

        if test_mode == False:
            # exclude invalid pixels and extremely large displacements
            nf_predictions = []
            for i in range(len(info_predictions)):
                if not self.args.use_var:
                    var_max = var_min = 0
                else:
                    var_max = self.args.var_max
                    var_min = self.args.var_min
                
                raw_b = info_predictions[i][:, 2:]
                log_b = torch.zeros_like(raw_b)
                weight = info_predictions[i][:, :2]
                # Large b Component
                log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=var_max)
                # Small b Component
                log_b[:, 1] = torch.clamp(raw_b[:, 1], min=var_min, max=0)
                # term2: [N, 2, m, H, W]
                term2 = ((flow_gt - flow_predictions[i]).abs().unsqueeze(2)) * (torch.exp(-log_b).unsqueeze(1))
                # term1: [N, m, H, W]
                term1 = weight - math.log(2) - log_b
                nf_loss = torch.logsumexp(weight, dim=1, keepdim=True) - torch.logsumexp(term1.unsqueeze(1) - term2, dim=2)
                nf_predictions.append(nf_loss)

            return {'final': flow_predictions[-1], 'flow': flow_predictions, 'info': info_predictions, 'nf': nf_predictions, 'init': init * 8}
        else:
            return {'final': flow_predictions[-1], 'flow': flow_predictions, 'info': info_predictions, 'nf': None, 'init': init * 8}