import torch
import torch.nn as nn
import torch.nn.functional as F

from .dinov2.dinov2 import DinoV2
from .dinov2.layers import PatchEmbed
from .heads.dpt_head import DPTHead
from .cam_dec import CameraDec
from .heads.linear_head import LinearDepth

from GeoNT.geom.graph_utils import graph_to_edge_list, keyframe_indices
from ..external import load_moge, load_raft, load_romav2
from ..external import InputPadder, _interpolate_warp_and_confidence, to_pixel
from GeoNT.geom.projective_ops import projective_transform
from lietorch import SE3


def flow_jacobian(flow):
    """
    Compute spatial Jacobian of optical flow.

    Args:
        flow: (B, 2, H, W) tensor, flow in normalized coordinates

    Returns:
        jacobian: (b, 4, H, W) tensor
                  channels = [dfx_dx, dfx_dy, dfy_dx, dfy_dy]
    """

    dx = flow[:, 0:1]
    dy = flow[:, 1:2]

    # Pad for central differences
    dx_pad = F.pad(dx, (1, 1, 1, 1), mode="replicate")
    dy_pad = F.pad(dy, (1, 1, 1, 1), mode="replicate")

    # Central differences
    dfx_dx = (dx_pad[:, :, 1:-1, 2:] - dx_pad[:, :, 1:-1, :-2]) * 0.5
    dfx_dy = (dx_pad[:, :, 2:, 1:-1] - dx_pad[:, :, :-2, 1:-1]) * 0.5

    dfy_dx = (dy_pad[:, :, 1:-1, 2:] - dy_pad[:, :, 1:-1, :-2]) * 0.5
    dfy_dy = (dy_pad[:, :, 2:, 1:-1] - dy_pad[:, :, :-2, 1:-1]) * 0.5

    jacobian = torch.cat(
        [dfx_dx, dfx_dy, dfy_dx, dfy_dy],
        dim=1,
    )
    
    return jacobian


class GeoNT(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = DinoV2(
            name='vitb',
            out_layers=[5, 7, 9, 11],
            alt_start=4,
            qknorm_start=4,
            rope_start=-1,
            cat_token=True,
        )
        self.embed_dim = self.backbone.pretrained.embed_dim
        self.patch_size = self.backbone.pretrained.patch_size
        # self.depth_head = DPTHead(
        #     dim_in=2*self.embed_dim,
        #     patch_size=self.patch_size,
        #     output_dim=2,
        #     activation="exp",
        #     conf_activation="sigmoid",
        #     out_channels=[96, 192, 384, 768],
        #     features=128,
        # )
        self.depth_head = LinearDepth(
            patch_size=self.patch_size,
            dec_embed_dim=2*self.embed_dim,
            activation="exp",
        )
        self.depth_patch_embed = PatchEmbed(in_chans=2, patch_size=self.patch_size, embed_dim=self.embed_dim - self.embed_dim // 4 * 3, flatten_embedding=False)
        self.motion_patch_embed = PatchEmbed(in_chans=5, patch_size=self.patch_size, embed_dim=self.embed_dim // 4 * 3, flatten_embedding=False)
        self.cam_dec = CameraDec(dim_in=1536)

    def forward(
        self, 
        flow_predictions, 
        depth_predictions, 
        intrinsics: torch.Tensor,
        export_feat_layers: list[int] | None = [],
        use_fp16: bool = False,
    ):
        # ---- motion tokenization ---- #
        flow = flow_predictions['final']
        flow_info = flow_predictions['info']

        # normalized coordinate
        ht, wd = flow.shape[-2:]
        fx, fy, cx, cy = intrinsics[..., None, None, :].unbind(dim=-1)
        v, u = torch.meshgrid(
            torch.arange(ht, device=flow.device, dtype=torch.float),
            torch.arange(wd, device=flow.device, dtype=torch.float),
            indexing="ij",
        )
        x, y = (u - cx) / fx, (v - cy) / fy

        # make flow intrinsic-invariant
        dx, dy = flow[:, 0] / fx, flow[:, 1] / fy

        # flow jacobian
        # jacobian = flow_jacobian(torch.stack((dx, dy), dim=1)) * 320

        x = x.expand(dx.shape[0], -1, -1)
        y = y.expand(dy.shape[0], -1, -1)

        # motion field encoder
        motion_field = torch.stack((x, y, dx * 50, dy * 50, flow_info), dim=1)
        # motion_field = torch.cat((motion_field, jacobian), dim=1)
        motion_token = self.motion_patch_embed(motion_field)

        # ---- depth tokenization ---- #
        depth = depth_predictions['depth']
        mask = depth_predictions['mask']
        assert depth.ndim == 2
        depthmap = torch.stack([depth, mask.to(depth.dtype)], dim=0)[None]
        depth_token = self.depth_patch_embed(depthmap)

        # expand the edge size
        depth_token = depth_token.expand(motion_token.shape[0], -1, -1, -1)

        patch_token = torch.cat((depth_token, motion_token), dim=-1)[None]  # [1,E,H,W,C]

        # multi-view transformer aggregation
        with torch.autocast(device_type=patch_token.device.type, enabled=use_fp16):
            feats, aux_feats = self.backbone(patch_token, export_feat_layers=export_feat_layers)

        # process features through depth head
        with torch.autocast(device_type=patch_token.device.type, enabled=False):
            depth = self.depth_head(feats, img_shape=(ht, wd))
            depth = depth.squeeze(0)
            pose_enc = self.cam_dec(feats[-1][1]).squeeze(0)

        output = {
            "depth": depth,  # L,H,W (L=1 in inference stage, L=num_layers in training stage for auxiliary supervision)
            "pose_enc": pose_enc,  # E,7
            "aux": self._extract_auxiliary_features(aux_feats, export_feat_layers, ht, wd),
        }
        
        return output

    def _extract_auxiliary_features(
        self, feats: list[torch.Tensor], feat_layers: list[int], H: int, W: int
    ) -> dict[str, torch.Tensor]:
        """Extract auxiliary features from specified layers."""
        aux_features = {}
        assert len(feats) == len(feat_layers)
        for feat, feat_layer in zip(feats, feat_layers):
            # Reshape features to spatial dimensions
            feat_reshaped = feat.reshape(
                [
                    feat.shape[0],
                    feat.shape[1],
                    H // self.patch_size,
                    W // self.patch_size,
                    feat.shape[-1],
                ]
            )
            aux_features[f"feat_layer_{feat_layer}"] = feat_reshaped
        
        return aux_features
    
    @property
    def num_out_layers(self):
        return len(self.backbone.out_layers)
    

class GeoNTWrapper(nn.Module):
    def __init__(self):
        super().__init__()

        # self.matcher = load_romav2()
        self.raft = load_raft()
        self.mono = load_moge('v2')
        self.model = GeoNT()

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)
    
    def load_state_dict(self, state_dict, strict=True, assign=False):
        return self.model.load_state_dict(state_dict, strict, assign)
    
    def normalize_depth(self, depth, mask, eps=1e-8):
        """
        depth: [B,H,W]
        mask:  [B,H,W]
        """
        assert depth.shape == mask.shape, "mask and depth must have the same dimensions"

        scaled_depth = torch.zeros_like(depth)
        scale = depth.new_zeros((depth.shape[0],))
        # invalid depth to zeros, required for MoGE
        depth[~mask] = 0
        for b in range(depth.shape[0]):
            valid = depth[b][mask[b]]
            if valid.numel() == 0:
                continue

            mean = valid.mean() + eps
            depth_b = depth[b] / mean
            scaled_depth[b] = depth_b
            scale[b] = mean

        scale[scale==0] = scale[scale>0].mean()

        return scaled_depth, scale
    
    # @torch.no_grad()
    # def extract_matches(self, images, ii, jj):
    #     # images: B,S,3,H,W
    #     B, S, _, H, W = images.shape
    #     assert B == 1
    #     images = images.flatten(0, 1)
    #     padder = InputPadder(images.shape)
    #     images = padder.pad(images)[0]
    #     # assume images between [0, 1]
    #     fmaps = self.matcher.f(images)
    #     refiner_features = self.matcher.refiner_features(images)

    #     # match feats
    #     img_A, img_B = images[ii], images[jj]
    #     matcher_output = self.matcher.matcher(
    #         [x[ii] for x in fmaps], [x[jj] for x in fmaps], img_A=img_A, img_B=img_B, bidirectional=False
    #     )
    #     warp_AB, confidence_AB = (
    #         matcher_output["warp_AB"],
    #         matcher_output["confidence_AB"],
    #     )
    #     # refine warp
    #     B, C, H, W = img_A.shape
    #     scale_factor = torch.tensor(
    #         (W / self.matcher.anchor_width, H / self.matcher.anchor_height), device=img_A.device
    #     )
    #     refiner_features_A, refiner_features_B = {k: v[ii] for k, v in refiner_features.items()}, {k: v[jj] for k, v in refiner_features.items()}
    #     for patch_size_str, refiner in self.matcher.refiners.items():
    #         patch_size = int(patch_size_str)
    #         warp_AB, confidence_AB = _interpolate_warp_and_confidence(
    #             warp=warp_AB,
    #             confidence=confidence_AB,
    #             H=H,
    #             W=W,
    #             patch_size=patch_size,
    #             zero_out_precision=False,
    #         )

    #         f_patch_A = refiner_features_A[patch_size]
    #         f_patch_B = refiner_features_B[patch_size]
    #         refiner_output_AB = refiner(
    #             f_A=f_patch_A,
    #             f_B=f_patch_B,
    #             prev_warp=warp_AB,
    #             prev_confidence=confidence_AB,
    #             scale_factor=scale_factor,
    #         )
                
    #         warp_AB, confidence_AB = (
    #             refiner_output_AB["warp"],
    #             refiner_output_AB["confidence"],
    #         )
    #     warp_AB = to_pixel(warp_AB, H=H, W=W)
    #     warp_AB = padder.unpad(warp_AB.permute(0, 3, 1, 2)).clone()
    #     confidence_AB = padder.unpad(confidence_AB.permute(0, 3, 1, 2)).clone()
    #     overlap_AB = confidence_AB[:, :1].sigmoid()
    #     preds = {
    #         "warp_AB": warp_AB,
    #         "overlap_AB": overlap_AB,
    #         "precision_AB": confidence_AB[:, 1:4],
    #     }
    #     return preds

    @torch.no_grad()
    def extract_matches(self, images, ii, jj):
        B, S, _, H, W = images.shape
        images = images.reshape(B*S, *images.shape[2:])
        images = 2 * images - 1.0

        # padding
        padder = InputPadder(images.shape)
        images = padder.pad(images)[0]
        fmaps = self.raft.fnet(images)

        img1, img2 = images[ii], images[jj]  
        fmap1, fmap2 = fmaps[ii], fmaps[jj]
        output = self.raft.infer_with_fmap(img1, img2, fmap1, fmap2, padder)

        flow_final = output['flow']
        info_final = output['info']
        weight = torch.softmax(info_final[:, :2], dim=1)
        raw_b = info_final[:, 2:]
        log_b = torch.zeros_like(raw_b)
        var_max, var_min = self.raft.args.var_max, self.raft.args.var_min
        # Large b Component
        log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=var_max)
        # Small b Component
        log_b[:, 1] = torch.clamp(raw_b[:, 1], min=var_min, max=0)
        info = (torch.exp(-log_b) * weight).sum(dim=1)

        return flow_final, info

    def forward(self, images, intrinsics, graph, gt_depths, gt_depths_valid, gt_pose, use_fp16=False):

        ii, jj, kk = graph_to_edge_list(graph)

        ii = ii.to(device=images.device, dtype=torch.long)
        jj = jj.to(device=images.device, dtype=torch.long)

        pose_graph = torch.zeros((ii.shape[0], 7), device=images.device, dtype=torch.float32)
        L = self.model.num_out_layers if self.training else 1
        depths = torch.zeros((images.shape[1], L, images.shape[3], images.shape[4]), device=images.device, dtype=torch.float32)

        # predict optical flow between graph edges
        images = images[:, :, [2,1,0]] / 255.0  # from BGR to RGB, in range [0, 1]
        B, S, _, H, W = images.shape
        assert B == 1
        # flow_final_est, info_est = self.extract_matches(images, ii, jj)

        # ====
        gt_pose = SE3(gt_pose).inv()  # convert poses w2c -> c2w
        coords0, val0 = projective_transform(gt_pose, 1.0 / gt_depths, intrinsics, ii, jj)
        H, W = coords0.shape[2:4]
        v, u = torch.meshgrid(
            torch.arange(H, device=coords0.device, dtype=torch.float),
            torch.arange(W, device=coords0.device, dtype=torch.float),
            indexing="ij",
        )
        flow_final = (coords0 - torch.stack((u, v), dim=-1))[0].permute(0, 3, 1, 2)
        info = (val0.squeeze(-1) * gt_depths_valid[:, ii].float())[0]

        # error = torch.norm(flow_final_est - flow_final, dim=1, keepdim=False)
        # flow_norm = torch.norm(flow_final, dim=1, keepdim=False)
        # info_ = info * info_est
        # print("error", error[1, 200:210, 200:210], flow_norm[1, 200:210, 200:210], flow_final_est[1, 0, 200:210, 200:210])
        # print("info", info_[1, 200:210, 200:210])

        # romav2 match
        # match_preds = self.extract_matches(images, ii, jj)
        # coords0_est = match_preds["warp_AB"]
        # error = torch.norm(coords0_est.permute(0, 2, 3, 1) - coords0, dim=-1, keepdim=False)
        # print("error", error[0, 0, 300:310, 300:310])
        # print("warp_AB", coords0_est[0, :, 200:210, 200:210])
        # ====

        # Monocular depth prior
        fx = intrinsics[0, :, 0]
        fov_x = torch.rad2deg(2 * torch.atan(W / (2 * fx)))
        depth_predictions = self.mono.infer(images[0], fov_x=fov_x)
        mono_depths, valid = depth_predictions["depth"], depth_predictions["mask"]
        scaled_depth, scale = self.normalize_depth(mono_depths.clone(), valid)

        iu = torch.unique(ii)
        for fi in iu:
            torch.cuda.empty_cache()
            # collect edges connected to the keyframe
            mask = (ii == fi)
            flow_predictions = {
                "final": flow_final[mask],
                "info": info[mask],
            }
            depth_predictions = {
                "depth": scaled_depth[fi],
                "mask": valid[fi],
            }

            output_geo = self.model(
                flow_predictions,
                depth_predictions,
                intrinsics[0, fi],
                export_feat_layers=[],
                use_fp16=use_fp16,
            )

            depths[fi] = output_geo["depth"]
            pose_graph[mask] = output_geo["pose_enc"]

        predictions = {
            "pose_graph": pose_graph[None],
            "depth": depths[None].squeeze(1),  # 1,S,H,W in inference stage, 1,S,L,H,W in training stage
            "valid": valid[None],
            # "flow": flow_final,
            # "info": info,
            "scale": scale[None],  # (B,S)
        }

        return predictions