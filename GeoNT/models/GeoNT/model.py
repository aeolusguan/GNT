import torch
import torch.nn as nn

from .dinov2.dinov2 import DinoV2
from .dinov2.layers import PatchEmbed
from .heads.dpt_head import DPTHead
from .cam_dec import CameraDec

from GeoNT.geom.graph_utils import graph_to_edge_list, keyframe_indices
from ..external import load_moge, load_raft
from GeoNT.geom.projective_ops import projective_transform
from lietorch import SE3


class GeoNT(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = DinoV2(
            name='vitb',
            out_layers=[5, 7, 9, 11],
            alt_start=4,
            qknorm_start=4,
            cat_token=True,
        )
        self.embed_dim = self.backbone.pretrained.embed_dim
        self.patch_size = self.backbone.pretrained.patch_size
        self.depth_head = DPTHead(
            dim_in=2*self.embed_dim,
            patch_size=self.patch_size,
            output_dim=2,
            activation="exp",
            conf_activation="sigmoid",
            out_channels=[96, 192, 384, 768],
            features=128,
        )
        self.depth_patch_embed = PatchEmbed(in_chans=2, patch_size=self.patch_size, embed_dim=self.embed_dim - self.embed_dim // 4 * 3)
        self.motion_patch_embed = PatchEmbed(in_chans=5, patch_size=self.patch_size, embed_dim=self.embed_dim // 4 * 3)
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

        x = x.expand(dx.shape[0], -1, -1)
        y = y.expand(dy.shape[0], -1, -1)

        # motion field encoder
        motion_field = torch.stack((x, y, dx * 50, dy * 50, flow_info), dim=1)
        motion_token = self.motion_patch_embed(motion_field)

        # ---- depth tokenization ---- #
        depth = depth_predictions['depth']
        mask = depth_predictions['mask']
        assert depth.ndim == 2
        depthmap = torch.stack([depth, mask.to(depth.dtype)], dim=0)[None]
        depth_token = self.depth_patch_embed(depthmap)

        # expand the edge size
        depth_token = depth_token.expand(motion_token.shape[0], -1, -1)

        patch_token = torch.cat((depth_token, motion_token), dim=-1)[None]  # [1,E,N,C]

        # multi-view transformer aggregation
        with torch.autocast(device_type=patch_token.device.type, enabled=use_fp16):
            feats, aux_feats = self.backbone(patch_token, export_feat_layers=export_feat_layers)

        # process features through depth head
        with torch.autocast(device_type=patch_token.device.type, enabled=False):
            depth, depth_conf = self.depth_head(
                feats, H=ht, W=wd, patch_start_idx=0
            )
            depth = depth.squeeze(0)
            depth_conf = depth_conf.squeeze(0)
            pose_enc = self.cam_dec(feats[-1][1]).squeeze(0)

        output = {
            "depth": depth,  # H,W
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
    

class GeoNTWrapper(nn.Module):
    def __init__(self):
        super().__init__()

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
    
    def forward(self, images, intrinsics, graph, gt_depths, gt_depths_valid, gt_pose, use_fp16=False):

        ii, jj, kk = graph_to_edge_list(graph)

        ii = ii.to(device=images.device, dtype=torch.long)
        jj = jj.to(device=images.device, dtype=torch.long)

        pose_graph = torch.zeros((ii.shape[0], 7), device=images.device, dtype=torch.float32)
        depths = torch.zeros((images.shape[1], images.shape[3], images.shape[4]), device=images.device, dtype=torch.float32)

        # predict optical flow between graph edges
        images = images[:, :, [2,1,0]]  # from BGR to RGB
        B, S, _, H, W = images.shape
        assert B == 1
        # fmaps, images_, padder = self.raft.extract_features(images.reshape(B*S, *images.shape[2:]))
        # images_ = images_.reshape(B, S, *images_.shape[1:])
        # fmaps = fmaps.reshape(B, S, *fmaps.shape[1:])
        # image1 = images_[:, ii]  
        # image2 = images_[:, jj]
        # fmap1 = fmaps[:, ii]
        # fmap2 = fmaps[:, jj]
        # output = self.raft.continue_infer(image1.flatten(0, 1), image2.flatten(0, 1), fmap1.flatten(0, 1), fmap2.flatten(0, 1), padder)

        # flow_final = output['flow']
        # info_final = output['info']
        # weight = torch.softmax(info_final[:, :2], dim=1)
        # raw_b = info_final[:, 2:]
        # log_b = torch.zeros_like(raw_b)
        # var_max, var_min = self.raft.args.var_max, self.raft.args.var_min
        # # Large b Component
        # log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=var_max)
        # # Small b Component
        # log_b[:, 1] = torch.clamp(raw_b[:, 1], min=var_min, max=0)
        # info = (torch.exp(-log_b) * weight).sum(dim=1)

        # del weight, raw_b, log_b, output
        # del fmaps, images_, fmap1, fmap2, image1, image2

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
        # ====

        # Monocular depth prior
        fx = intrinsics[0, :, 0]
        fov_x = torch.rad2deg(2 * torch.atan(W / (2 * fx)))
        depth_predictions = self.mono.infer(images[0] / 255.0, fov_x=fov_x)
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
            "depth": depths[None],
            "valid": valid[None],
            "flow": flow_final,
            "info": info,
            "scale": scale[None],  # (B,S)
        }

        return predictions