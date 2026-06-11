from typing import Union, IO
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dinov2.dinov2 import DinoV2
from .dinov2.layers import PatchEmbed
from .cam_dec import CameraDec
from .heads.transformer_head import TransformerDecoder
from geont.geometry.graph_utils import graph_to_edge_list, keyframe_indices
from .external import load_moge
from .flow.core.utils import InputPadder
from .flow import load_flow
from geont.geometry.projective_ops import projective_transform
from lietorch import SE3


class MotionPatchEmbed(nn.Module):
    def __init__(
        self,
        embed_dim,
        patch_size,
    ):
        super().__init__()

        self.motion_embed = PatchEmbed(in_chans=5, patch_size=patch_size, embed_dim=embed_dim, flatten_embedding=False)

    def forward(self, motion_field, info, intrinsics, var_min, var_max):
        # compute gate
        weight = torch.softmax(info[:, :2], dim=1)
        raw_b = info[:, 2:]
        log_b = torch.zeros_like(raw_b)
        # Large b Component
        log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=var_max)
        # Small b Component
        log_b[:, 1] = torch.clamp(raw_b[:, 1], min=var_min, max=0)
        info_final = (torch.exp(-log_b) * weight).sum(dim=1, keepdim=True)

        # embed motion field
        ht, wd = motion_field.shape[-2:]
        fx, fy, cx, cy = intrinsics[..., None, None, :].unbind(dim=-1)
        v, u = torch.meshgrid(
            torch.arange(ht, device=motion_field.device, dtype=torch.float),
            torch.arange(wd, device=motion_field.device, dtype=torch.float),
            indexing="ij",
        )
        x, y = (u - cx) / fx, (v - cy) / fy

        # make flow intrinsic-invariant
        dx, dy = motion_field[:, 0] / fx, motion_field[:, 1] / fy

        x = x.expand(dx.shape[0], -1, -1)
        y = y.expand(dy.shape[0], -1, -1)

        # motion field encoder
        motion_token = self.motion_embed(torch.stack((x, y, dx * 50, dy * 50, info_final.squeeze(1)), dim=1))

        return motion_token

        
class GeoNT(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = DinoV2(
            name='vitb',
            out_layers=[5, 7, 9, 11],
            alt_start=4,
            qknorm_start=4,
            rope_start=4,
            cat_token=True,
        )
        self.embed_dim = self.backbone.pretrained.embed_dim
        self.patch_size = self.backbone.pretrained.patch_size

        self.depth_head = TransformerDecoder(
            in_dim=2*self.embed_dim,
            patch_size=self.patch_size,
            dec_embed_dim=512,
            dec_num_heads=8,
            output_dim=2,
            depth=2,
            activation="exp",
            conf_activation="expp1",
            rope=self.backbone.pretrained.rope,
        )
        self.depth_patch_embed = PatchEmbed(in_chans=2, patch_size=self.patch_size, embed_dim=self.embed_dim - self.embed_dim // 4 * 3, flatten_embedding=False)
        self.motion_patch_embed = MotionPatchEmbed(patch_size=self.patch_size, embed_dim=self.embed_dim // 4 * 3)
        self.res_depth_embed = PatchEmbed(in_chans=2, patch_size=self.patch_size//2, embed_dim=self.embed_dim, flatten_embedding=True)
        
        self.cam_dec = CameraDec(dim_in=1536)

    def forward(
        self, 
        flow_predictions, 
        depth_predictions,
        intrinsics: torch.Tensor,
        export_feat_layers: list[int] | None = None,
        use_fp16: bool = False,
    ):
        if export_feat_layers is None:
            export_feat_layers = []

        # ---- motion tokenization ---- #
        flow = flow_predictions['final']
        flow_info = flow_predictions['info']
        var_min, var_max = flow_predictions['var_min'], flow_predictions['var_max']

        motion_token = self.motion_patch_embed(flow, flow_info, intrinsics, var_min=var_min, var_max=var_max)

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
        with torch.autocast(device_type=patch_token.device.type, enabled=use_fp16 and patch_token.is_cuda):
            feats, aux_feats = self.backbone(patch_token, export_feat_layers=export_feat_layers)

        res_feat = self.res_depth_embed(depthmap)
        ht, wd = depth.shape[-2:]
        with torch.autocast(device_type=patch_token.device.type, enabled=False):
            depth, depth_conf = self.depth_head(feats, res_feat, img_shape=(ht, wd))  # 1,H,W
            pose_enc, pose_log_variance = self.cam_dec(feats[-1][1])  # 1,E,7, 1,E,2

        output = {
            "depth": depth.squeeze(0),  # H,W
            "depth_conf": depth_conf.squeeze(0),  # H,W
            "pose_enc": pose_enc.squeeze(0),  # E,7
            "pose_confidence": pose_log_variance.squeeze(0),  # E,2
            "pose_log_variance": pose_log_variance.squeeze(0),  # E,2
            "aux": self._extract_auxiliary_features(aux_feats, export_feat_layers, ht, wd),
        }
        
        return output

    def forward_from_motion_tokens(
        self,
        motion_token: torch.Tensor,
        depth_predictions,
        intrinsics: torch.Tensor,
        export_feat_layers: list[int] | None = None,
        use_fp16: bool = False,
        decode_depth: bool = True,
    ):
        """Run GeoNT from precomputed per-edge motion tokens.

        Streaming inference stores motion tokens in the factor graph so
        marginalization can finalize edges without re-running the flow frontend.
        """
        if export_feat_layers is None:
            export_feat_layers = []

        depth = depth_predictions["depth"]
        mask = depth_predictions["mask"]
        assert depth.ndim == 2
        depthmap = torch.stack([depth, mask.to(depth.dtype)], dim=0)[None]
        depth_token = self.depth_patch_embed(depthmap)
        depth_token = depth_token.expand(motion_token.shape[0], -1, -1, -1)

        patch_token = torch.cat((depth_token, motion_token), dim=-1)[None]

        with torch.autocast(device_type=patch_token.device.type, enabled=use_fp16 and patch_token.is_cuda):
            feats, aux_feats = self.backbone(patch_token, export_feat_layers=export_feat_layers)

        ht, wd = depth.shape[-2:]
        with torch.autocast(device_type=patch_token.device.type, enabled=False):
            pose_enc, pose_log_variance = self.cam_dec(feats[-1][1])

        output = {
            "pose_enc": pose_enc.squeeze(0),
            "pose_confidence": pose_log_variance.squeeze(0),
            "aux": self._extract_auxiliary_features(aux_feats, export_feat_layers, ht, wd),
        }
        if decode_depth:
            res_feat = self.res_depth_embed(depthmap)
            with torch.autocast(device_type=patch_token.device.type, enabled=False):
                depth, depth_conf = self.depth_head(feats, res_feat, img_shape=(ht, wd))
            output["depth"] = depth.squeeze(0)
            output["depth_conf"] = depth_conf.squeeze(0)
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

        self.flow = load_flow()
        self.mono = load_moge('v2')
        self.gnt = GeoNT()

        self.freeze_model()

    def freeze_model(self):
        def _freeze_model(model):
            model = model.eval()
            for p in model.parameters():
                p.requires_grad = False
            for p in model.buffers():
                p.requires_grad = False
            return model
        _freeze_model(self.mono)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, Path, IO[bytes]], **hf_kwargs) -> 'GeoNTWrapper':
        """
        Load a model from a checkpoint file.

        Args:
            pretrained_model_name_or_path: path to the checkpoint file or repo id.
            hf_kwargs: additional keyword arguments to pass to the huf_hub_download function. Ignored if pretrained_model_name_or_path is a local path.

        Returns:
            a new instance of 'GeoNTWrapper' with the parameters loaded from the checkpoint.
        """
        ckpt = torch.load(pretrained_model_name_or_path, map_location="cpu", weights_only=False)
        model = cls()
        model.load_state_dict(ckpt['model'], strict=True)
        return model

    def normalize_depth(self, depth, mask, eps=1e-8):
        """
        depth: [B,H,W]
        mask:  [B,H,W]
        """
        assert depth.shape == mask.shape, "mask and depth must have the same dimensions"

        depth = depth.masked_fill(~mask, 0)
        scaled_depth = torch.zeros_like(depth)
        scale = depth.new_zeros((depth.shape[0],))
        for b in range(depth.shape[0]):
            valid = depth[b][mask[b]]
            if valid.numel() == 0:
                continue

            mean = valid.mean() + eps
            depth_b = depth[b] / mean
            scaled_depth[b] = depth_b
            scale[b] = mean

        valid_scale = scale > 0
        if valid_scale.any():
            scale[~valid_scale] = scale[valid_scale].mean()
        else:
            scale[:] = 1.0

        return scaled_depth, scale

    def _frontend_forward(self, images, intrinsics, graph):

        ii, jj, kk = graph_to_edge_list(graph)

        ii = ii.to(device=images.device, dtype=torch.long)
        jj = jj.to(device=images.device, dtype=torch.long)

        B, S, _, H, W = images.shape
        assert B == 1

        # Monocular depth prior
        fx = intrinsics[0, :, 0]
        fov_x = torch.rad2deg(2 * torch.atan(W / (2 * fx)))
        depth_predictions = self.mono.infer(images[0], fov_x=fov_x)
        mono_depths, valid = depth_predictions["depth"], depth_predictions["mask"]

        # Predict optical flow between graph edges
        mono_depths = mono_depths.clamp_min(0.01)
        disps = torch.zeros_like(mono_depths)
        disps[valid] = 1.0 / mono_depths[valid]
        bases = self.flow.create_bases(disps.unsqueeze(1))
        mono = depth_predictions["feature"]
        
        images = images.reshape(B*S, *images.shape[2:])
        images = 2 * images - 1.0
        
        # padding
        padder = InputPadder(images.shape)
        images, mono, bases = padder.pad(images, mono, bases)
        fmap_8x = self.flow.fnet(images)
        mono_8x = self.flow.merge_head(mono)
        fmap_8x = torch.cat((fmap_8x, mono_8x), dim=1)
        flow_predictions = self.flow.forward_with_fmap(fmap_8x[ii], fmap_8x[jj], bases[ii])

        flow_predictions = {
            "flow": [padder.unpad(x) for x in flow_predictions["flow"]],
            "info": [padder.unpad(x) for x in flow_predictions["info"]],
            "prob_up": padder.unpad(flow_predictions["prob_up"]),
            "init": padder.unpad(flow_predictions["init"]),
        }

        return flow_predictions, depth_predictions

    def forward(self, images, intrinsics, graph, gt_depths, gt_depths_valid, gt_pose, use_fp16=False):

        ii, jj, kk = graph_to_edge_list(graph)

        ii = ii.to(device=images.device, dtype=torch.long)
        jj = jj.to(device=images.device, dtype=torch.long)

        images = images[:, :, [2,1,0]] / 255.0  # from BGR to RGB, in range [0, 1]

        L = self.gnt.num_out_layers if self.training else 1
        pose_graph = torch.zeros((ii.shape[0], 7), device=images.device, dtype=torch.float32)
        pose_graph_log_variance = torch.zeros((ii.shape[0], 2), device=images.device, dtype=torch.float32)
        depths = torch.zeros((images.shape[1], images.shape[3], images.shape[4]), device=images.device, dtype=torch.float32)
        depths_conf = torch.zeros((images.shape[1], images.shape[3], images.shape[4]), device=images.device, dtype=torch.float32)

        # ====
        # gt_pose = SE3(gt_pose).inv()  # convert poses w2c -> c2w
        # coords0, val0 = projective_transform(gt_pose, 1.0 / gt_depths, intrinsics, ii, jj)
        # H, W = coords0.shape[2:4]
        # v, u = torch.meshgrid(
        #     torch.arange(H, device=coords0.device, dtype=torch.float),
        #     torch.arange(W, device=coords0.device, dtype=torch.float),
        #     indexing="ij",
        # )
        # flow_final = (coords0 - torch.stack((u, v), dim=-1))[0].permute(0, 3, 1, 2)
        # info = (val0.squeeze(-1) * gt_depths_valid[:, ii].float())[0]

        # error = torch.norm(flow_final_est - flow_final, dim=1, keepdim=False)
        # flow_norm = torch.norm(flow_final, dim=1, keepdim=False)
        # info_ = info * info_est
        # print("error", error[1, 200:210, 200:210], flow_norm[1, 200:210, 200:210], flow_final_est[1, 0, 200:210, 200:210])
        # print("info", info_[1, 200:210, 200:210])

        # Front end
        flow_predictions, depth_predictions = self._frontend_forward(images, intrinsics, graph)
        mono_depths, valid = depth_predictions["depth"], depth_predictions["mask"]
        scaled_depth, scale = self.normalize_depth(mono_depths, valid)

        flow_final, info_final = flow_predictions["flow"][-1], flow_predictions["info"][-1]

        iu = torch.unique(ii)
        for fi in iu:
            torch.cuda.empty_cache()
            # collect edges connected to the keyframe
            mask = (ii == fi)
            flow_input = {
                "final": flow_final[mask],
                "info": info_final[mask],
                "var_min": self.flow.args.var_min,
                "var_max": self.flow.args.var_max,
            }
            depth_input = {
                "depth": scaled_depth[fi],
                "mask": valid[fi],
            }

            output_geo = self.gnt(
                flow_input,
                depth_input,
                intrinsics[0, fi],
                export_feat_layers=[],
                use_fp16=use_fp16,
            )

            depths[fi] = output_geo["depth"]
            depths_conf[fi] = output_geo["depth_conf"]
            pose_graph[mask] = output_geo["pose_enc"]
            pose_graph_log_variance[mask] = output_geo["pose_log_variance"]

        predictions = {
            "pose_graph": pose_graph[None],  # 1,E,7
            "pose_graph_log_variance": pose_graph_log_variance[None],  # 1,E,2
            "depth": depths[None],  # 1,S,H,W
            "depth_conf": depths_conf[None],  # 1,S,H,W
            "valid": valid[None],
            "scale": scale[None],  # (B,S)
            "flow_predictions": flow_predictions,
            "mono_depth": mono_depths[None],  # 1,S,H,W
        }

        return predictions

    def encode_features(self, images: torch.Tensor, intrinsics: torch.Tensor):
        """
        image (torch.Tensor): BCHW image RGB 0-1
        intrinsics (torch.Tensor): B4
        """
        B, _, H, W = images.shape
        # Monocular depth prior
        fx = intrinsics[:, 0]
        fov_x = torch.rad2deg(2 * torch.atan(W / (2 * fx)))
        depth_predictions = self.mono.infer(images, fov_x=fov_x)
        mono_depths, mono_feats, valid = depth_predictions["depth"], depth_predictions["feature"], depth_predictions["mask"]
        
        images = 2 * images - 1.0
        fmap_8x = self.flow.fnet(images)
        mono_8x = self.flow.merge_head(mono_feats)
        fmap_8x = torch.cat((fmap_8x, mono_8x), dim=1)

        return fmap_8x, mono_depths, valid
    
    def encode_bases(self, depths: torch.Tensor, valid: torch.Tensor):
        """
        depths (torch.Tensor): BHW depth estimation
        valid (torch.Tensor): BHW valid mask, boolean
        """
        depths = depths.clamp_min(0.01)
        disps = torch.zeros_like(depths)
        disps[valid] = 1.0 / depths[valid]
        bases = self.flow.create_bases(disps.unsqueeze(1))
        return self.flow.init_decoder.patch_embed_base(bases)

    def flow_init(self, fmap1_8x, fmap2_8x, bases):
        idx_bins = torch.linspace(-16, 16, self.flow.n_bins, device=fmap1_8x.device, dtype=fmap1_8x.dtype).view(1, self.flow.n_bins, 1, 1)

        x = self.flow.init_proj(torch.cat([fmap1_8x, fmap2_8x], dim=1))
        x, net = self.flow.init_decoder.forward_with_bases(x, bases)
        init_bins = self.flow.init_bin_head(x)
        init_mask = .25 * self.flow.init_mask_head(x)

        flow_8x = self.flow.init_pred(init_bins, idx_bins)
        init_flow = self.flow.upsample_flow(flow_8x, init_mask) / 8.0
        net = self.flow.net_init(net)

        return init_flow, net
    
    def encode_flow(self, fmap1_8x, fmap2_8x, bases, intrinsics):
        flow, info = self.flow.infer(fmap1_8x, fmap2_8x, bases)
        motion_token = self.gnt.motion_patch_embed(flow, info, intrinsics, var_min=self.flow.args.var_min, var_max=self.flow.args.var_max)
        return motion_token

    def refine_from_motion_tokens(
        self,
        motion_token: torch.Tensor,
        depth: torch.Tensor,
        mask: torch.Tensor,
        intrinsics: torch.Tensor,
        use_fp16: bool = False,
        decode_depth: bool = True,
    ):
        return self.gnt.forward_from_motion_tokens(
            motion_token,
            {"depth": depth, "mask": mask},
            intrinsics,
            export_feat_layers=[],
            use_fp16=use_fp16,
            decode_depth=decode_depth,
        )
