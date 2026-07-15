from pathlib import Path

import torch
import torch.nn as nn
from hydra.utils import instantiate

from .dinov2.layers import PatchEmbed
from .cam_dec import CameraDec
from .cam_enc import CameraEnc
from .heads.transformer_head import TransformerDecoder
from geont.geometry.graph_utils import graph_to_edge_list
from .external import load_moge
from .flow.core.utils import InputPadder
from .flow import load_flow


class MotionPatchEmbed(nn.Module):
    def __init__(
        self,
        embed_dim,
        patch_size,
    ):
        super().__init__()

        self.motion_embed = PatchEmbed(in_chans=5, patch_size=patch_size, embed_dim=embed_dim, flatten_embedding=False)

    def forward(self, motion_field, info, source_intrinsics, target_intrinsics, var_min, var_max):
        # compute gate
        weight = torch.softmax(info[:, :2], dim=1)
        raw_b = info[:, 2:]
        log_b = torch.zeros_like(raw_b)
        # Large b Component
        log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=var_max)
        # Small b Component
        log_b[:, 1] = torch.clamp(raw_b[:, 1], min=var_min, max=0)
        info_final = (torch.exp(-log_b) * weight).sum(dim=1, keepdim=True)

        ht, wd = motion_field.shape[-2:]
        assert source_intrinsics.shape == target_intrinsics.shape == (motion_field.shape[0], 4)
        source_fx, source_fy, source_cx, source_cy = source_intrinsics[..., None, None, :].unbind(dim=-1)
        target_fx, target_fy, target_cx, target_cy = target_intrinsics[..., None, None, :].unbind(dim=-1)
        v, u = torch.meshgrid(
            torch.arange(ht, device=motion_field.device, dtype=motion_field.dtype),
            torch.arange(wd, device=motion_field.device, dtype=motion_field.dtype),
            indexing="ij",
        )
        source_x = (u + motion_field[:, 0] - source_cx) / source_fx
        source_y = (v + motion_field[:, 1] - source_cy) / source_fy
        target_x = (u - target_cx) / target_fx
        target_y = (v - target_cy) / target_fy

        motion_token = self.motion_embed(
            torch.stack((source_x, source_y, target_x, target_y, info_final.squeeze(1)), dim=1)
        )

        return motion_token

        
class GeoNT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.backbone = instantiate(config["backbone"])
        self.embed_dim = self.backbone.pretrained.embed_dim
        self.patch_size = self.backbone.pretrained.patch_size

        depth_head_dim_in = (2 if self.backbone.cat_token else 1) * self.embed_dim
        self.depth_head = TransformerDecoder(
            in_dim=depth_head_dim_in,
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
        self.res_depth_embed = PatchEmbed(in_chans=2, patch_size=self.patch_size // 2, embed_dim=self.embed_dim, flatten_embedding=True)
        
        cam_dec_dim_in = (2 if self.backbone.cat_token else 1) * self.embed_dim
        self.cam_dec = CameraDec(dim_in=cam_dec_dim_in)
        self.cam_enc = CameraEnc(embed_dim=self.embed_dim)

    def encode_relative_pose_prior(
        self,
        relative_pose_prior: torch.Tensor,
        relative_pose_prior_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert relative_pose_prior.ndim == 2 and relative_pose_prior.shape[-1] == 7
        base_token = self.backbone.pretrained.camera_token[:, 1:, :]
        prior_token = self.cam_enc(relative_pose_prior, base_token)
        if relative_pose_prior_mask is None:
            return prior_token

        assert relative_pose_prior_mask.shape == (relative_pose_prior.shape[0],)
        base = base_token.to(device=prior_token.device, dtype=prior_token.dtype).expand_as(prior_token)
        mask = relative_pose_prior_mask.to(device=prior_token.device, dtype=torch.bool).view(-1, 1, 1)
        return torch.where(mask, prior_token, base)

    def forward(
        self, 
        flow_predictions, 
        depth_predictions,
        source_intrinsics: torch.Tensor,
        target_intrinsics: torch.Tensor,
        target_depth_predictions,
        use_fp16: bool = False,
        decode_depth: bool = True,
        relative_pose_prior: torch.Tensor | None = None,
        relative_pose_prior_mask: torch.Tensor | None = None,
    ):
        # ---- motion tokenization ---- #
        flow = flow_predictions['final']
        flow_info = flow_predictions['info']
        var_min, var_max = flow_predictions['var_min'], flow_predictions['var_max']

        zero_flow = torch.zeros(1, *flow.shape[1:], device=flow.device, dtype=flow.dtype)
        reference_info = torch.zeros(1, *flow_info.shape[1:], device=flow_info.device, dtype=flow_info.dtype)
        all_flow = torch.cat((zero_flow, flow), dim=0)
        all_info = torch.cat((reference_info, flow_info), dim=0)
        assert source_intrinsics.shape == (4,)
        assert target_intrinsics.shape == (flow.shape[0], 4)
        reference_intrinsics = source_intrinsics[None]
        all_source_intrinsics = torch.cat(
            (reference_intrinsics, reference_intrinsics.expand(flow.shape[0], -1)),
            dim=0,
        )
        all_target_intrinsics = torch.cat((reference_intrinsics, target_intrinsics), dim=0)
        motion_token = self.motion_patch_embed(
            all_flow,
            all_info,
            all_source_intrinsics,
            all_target_intrinsics,
            var_min=var_min,
            var_max=var_max,
        )

        # ---- depth tokenization ---- #
        depth = depth_predictions['depth']
        mask = depth_predictions['mask']
        target_depth = target_depth_predictions['depth']
        target_mask = target_depth_predictions['mask']
        assert depth.ndim == 2
        assert target_depth.ndim == 3
        assert target_depth.shape[0] == flow.shape[0]
        depthmap = torch.stack([depth, mask.to(depth.dtype)], dim=0)[None]
        target_depthmap = torch.stack([target_depth, target_mask.to(target_depth.dtype)], dim=1)
        all_depthmap = torch.cat((depthmap, target_depthmap), dim=0)
        depth_token = self.depth_patch_embed(all_depthmap)

        patch_token = torch.cat((depth_token, motion_token), dim=-1)[None]  # [1,1+E,H,W,C]

        # multi-view transformer aggregation
        backbone_kwargs = {}
        if relative_pose_prior is not None:
            prior_token = self.encode_relative_pose_prior(relative_pose_prior, relative_pose_prior_mask)
            assert prior_token.shape == (flow.shape[0], 1, self.embed_dim)
            reference_pose_prior = relative_pose_prior.new_zeros((1, 7))
            reference_pose_prior[:, 6] = 1.0
            reference_token = self.cam_enc(reference_pose_prior, self.backbone.pretrained.camera_token[:, :1, :])
            backbone_kwargs["cam_token"] = torch.cat((reference_token, prior_token), dim=0)
        with torch.autocast(device_type=patch_token.device.type, enabled=use_fp16 and patch_token.is_cuda):
            feats, _ = self.backbone(
                patch_token,
                export_feat_layers=[],
                **backbone_kwargs,
            )

        ht, wd = depth.shape[-2:]
        with torch.autocast(device_type=patch_token.device.type, enabled=False):
            pose_enc, pose_log_variance, relative_log_scale, scale_confidence = self.cam_dec(feats[-1][1][:, 1:])  # 1,E,7, 1,E,2

        output = {
            "pose_enc": pose_enc.squeeze(0),  # E,7
            "pose_log_variance": pose_log_variance.squeeze(0),  # E,2
            "relative_log_scale": relative_log_scale.squeeze(0),  # E
            "scale_confidence": scale_confidence.squeeze(0),  # E
        }
        if decode_depth:
            res_feat = self.res_depth_embed(depthmap)
            with torch.autocast(device_type=patch_token.device.type, enabled=False):
                depth, depth_conf = self.depth_head(feats, res_feat, img_shape=(ht, wd))  # 1,H,W
            output["depth"] = depth.squeeze(0)  # H,W
            output["depth_conf"] = depth_conf.squeeze(0)  # H,W
        
        return output

class GeoNTWrapper(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        self.flow = load_flow()
        self.mono = load_moge('v2')
        self.gnt = GeoNT(model_config)

        self.mono.eval()
        for p in self.mono.parameters():
            p.requires_grad = False

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path) -> "GeoNTWrapper":
        """
        Load a model from a checkpoint file.

        Args:
            pretrained_model_name_or_path: path to the checkpoint file.

        Returns:
            a new instance of 'GeoNTWrapper' with the parameters loaded from the checkpoint.
        """
        ckpt = torch.load(pretrained_model_name_or_path, map_location="cpu", weights_only=False)
        model = cls(ckpt["model_config"])
        model.load_state_dict(ckpt["model"])
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
        if valid_scale.any().item():
            scale[~valid_scale] = scale[valid_scale].mean()
        else:
            scale[:] = 1.0

        return scaled_depth, scale

    def _frontend_forward(self, images, intrinsics, graph):

        ii, jj, _ = graph_to_edge_list(graph)

        ii = ii.to(device=images.device, dtype=torch.long)
        jj = jj.to(device=images.device, dtype=torch.long)

        B, S, _, H, W = images.shape
        assert B == 1

        # Monocular depth prior
        fx = intrinsics[0, :, 0]
        fov_x = torch.rad2deg(2 * torch.atan(W / (2 * fx)))
        depth_predictions = self.mono.infer(images[0], fov_x=fov_x)
        mono_depths, valid = depth_predictions["depth"], depth_predictions["mask"]

        # Predict target-to-reference correspondence flow for each graph edge i -> j.
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
        flow_predictions = self.flow.forward_with_fmap(fmap_8x[jj], fmap_8x[ii], bases[jj])

        flow_predictions = {
            "flow": [padder.unpad(x) for x in flow_predictions["flow"]],
            "info": [padder.unpad(x) for x in flow_predictions["info"]],
            "prob_up": padder.unpad(flow_predictions["prob_up"]),
            "init": padder.unpad(flow_predictions["init"]),
        }

        return flow_predictions, depth_predictions

    def forward(
        self,
        images,
        intrinsics,
        graph,
        gt_depth_valid,
        pose_prior_args,
        use_fp16=False,
    ):
        ii, jj, _ = graph_to_edge_list(graph)
        iu_list = torch.unique(ii).tolist()

        ii = ii.to(device=images.device, dtype=torch.long)
        jj = jj.to(device=images.device, dtype=torch.long)

        images = images[:, :, [2,1,0]] / 255.0  # from BGR to RGB, in range [0, 1]

        # Edge measurement layout: 0:3 translation, 3:7 quaternion,
        # 7:9 pose confidence/log variance, 9 relative log scale, 10 scale confidence.
        edge_measurements = torch.zeros((ii.shape[0], 11), device=images.device, dtype=torch.float32)
        edge_measurements[:, 10] = 1.0
        depths = torch.zeros((images.shape[1], images.shape[3], images.shape[4]), device=images.device, dtype=torch.float32)
        depths_conf = torch.zeros((images.shape[1], images.shape[3], images.shape[4]), device=images.device, dtype=torch.float32)

        # Front end
        flow_predictions, depth_predictions = self._frontend_forward(images, intrinsics, graph)
        mono_depths, valid = depth_predictions["depth"], depth_predictions["mask"]
        # mono_depths/valid: [S,H,W]; gt_depth_valid: [1,S,H,W].
        # train_depth_valid is the shared MoGe/GT mask used for the input depth gauge.
        train_depth_valid = valid & gt_depth_valid[0]
        # scaled_depth: [S,H,W] with valid-mask mean 1 per frame; scale: [S].
        scaled_depth, scale = self.normalize_depth(mono_depths, train_depth_valid)

        flow_final, info_final = flow_predictions["flow"][-1], flow_predictions["info"][-1]

        train_prob = pose_prior_args.pose_prior_train_prob
        pose_prior_selected_list = (torch.rand(len(iu_list)) < train_prob).tolist()
        if train_prob > 0.0 and not any(pose_prior_selected_list):
            pose_prior_selected_list[int(torch.randint(len(iu_list), (1,)).item())] = True

        for group_idx, fi in enumerate(iu_list):
            torch.cuda.empty_cache()
            # mask: [E], selecting E_i outgoing edges for source frame fi.
            mask = (ii == fi)
            group_flow_predictions = {
                "final": flow_final[mask],  # [E_i,2,H,W]
                "info": info_final[mask],
                "var_min": self.flow.args.var_min,
                "var_max": self.flow.args.var_max,
            }
            depth_input = {
                "depth": scaled_depth[fi],  # [H,W]
                "mask": train_depth_valid[fi],  # [H,W]
            }
            target_scale_ratio = scale[jj[mask]] / scale[fi]
            target_depth = scaled_depth[jj[mask]] * target_scale_ratio[:, None, None]
            target_depth_input = {
                "depth": target_depth,  # [E_i,H,W], normalized in source/reference gauge.
                "mask": train_depth_valid[jj[mask]],  # [E_i,H,W]
            }

            if pose_prior_selected_list[group_idx]:
                with torch.no_grad():
                    first_output = self.gnt(
                        group_flow_predictions,
                        depth_input,
                        source_intrinsics=intrinsics[0, fi],
                        target_intrinsics=intrinsics[0, jj[mask]],
                        target_depth_predictions=target_depth_input,
                        use_fp16=use_fp16,
                        decode_depth=False,
                    )
                relative_pose_prior = first_output["pose_enc"].detach()
                output_geo = self.gnt(
                    group_flow_predictions,
                    depth_input,
                    source_intrinsics=intrinsics[0, fi],
                    target_intrinsics=intrinsics[0, jj[mask]],
                    target_depth_predictions=target_depth_input,
                    use_fp16=use_fp16,
                    decode_depth=True,
                    relative_pose_prior=relative_pose_prior,
                )
            else:
                output_geo = self.gnt(
                    group_flow_predictions,
                    depth_input,
                    source_intrinsics=intrinsics[0, fi],
                    target_intrinsics=intrinsics[0, jj[mask]],
                    target_depth_predictions=target_depth_input,
                    use_fp16=use_fp16,
                    decode_depth=True,
                )

            depths[fi] = output_geo["depth"]
            depths_conf[fi] = output_geo["depth_conf"]
            edge_measurements[mask, :7] = output_geo["pose_enc"]
            edge_measurements[mask, 7:9] = output_geo["pose_log_variance"]
            edge_measurements[mask, 9] = output_geo["relative_log_scale"]
            edge_measurements[mask, 10] = output_geo["scale_confidence"]

        predictions = {
            "edge_measurements": edge_measurements[None],  # 1,E,11
            "depth": depths[None],  # 1,S,H,W
            "depth_conf": depths_conf[None],  # 1,S,H,W
            "valid": train_depth_valid[None],  # 1,S,H,W
            "scale": scale[None],  # (B,S)
            "flow_predictions": flow_predictions,
            "mono_depth": mono_depths[None],  # 1,S,H,W
        }

        return predictions
