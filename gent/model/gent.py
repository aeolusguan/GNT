from pathlib import Path

import torch
import torch.nn as nn
from hydra.utils import instantiate

from .dinov2.layers import PatchEmbed
from .cam_dec import CameraDec
from .cam_enc import CameraEnc
from .heads.transformer_head import TransformerDecoder
from gent.geometry.graph_utils import graph_to_edge_list
from .external import load_moge
from .flow.core.utils import InputPadder
from .flow import load_flow


def build_depth_normalization_mask(
    depth: torch.Tensor,  # [S,H,W]
    non_sky_mask: torch.Tensor,  # [S,H,W]
    *,
    min_cutoff: float,
    quantile: float,
    valid_mask: torch.Tensor | None = None,  # [S,H,W]
) -> torch.Tensor:
    """Select the per-view support used to normalize a MoGe depth prior."""
    cutoffs = []
    for frame_depth, frame_non_sky in zip(depth, non_sky_mask, strict=True):
        non_sky_depth = frame_depth[frame_non_sky]
        if non_sky_depth.numel() == 0:
            cutoff = frame_depth.new_tensor(min_cutoff)
        else:
            cutoff = torch.quantile(non_sky_depth, quantile).clamp_min(min_cutoff)
        cutoffs.append(cutoff)

    cutoff = torch.stack(cutoffs)[:, None, None]
    normalization_mask = non_sky_mask & (depth <= cutoff)
    if valid_mask is not None:
        normalization_mask = normalization_mask & valid_mask
    return normalization_mask


class MotionPatchEmbed(nn.Module):
    def __init__(
        self,
        embed_dim,
        patch_size,
    ):
        super().__init__()

        self.motion_embed = PatchEmbed(in_chans=5, patch_size=patch_size, embed_dim=embed_dim, flatten_embedding=False)

    def forward(
        self,
        motion_field: torch.Tensor,  # [N,2,H,W]
        info: torch.Tensor,  # [N,C,H,W]
        source_intrinsics: torch.Tensor,  # [N,4]
        target_intrinsics: torch.Tensor,  # [N,4]
        var_min,
        var_max,
    ):
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


class GeNT(nn.Module):
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
        self.depth_embed_dim = self.embed_dim // 4
        self.depth_patch_embed = PatchEmbed(
            in_chans=2,
            patch_size=self.patch_size,
            embed_dim=self.depth_embed_dim,
            flatten_embedding=False,
        )
        self.motion_patch_embed = MotionPatchEmbed(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim - self.depth_embed_dim,
        )
        self.res_depth_embed = PatchEmbed(
            in_chans=2,
            patch_size=self.patch_size // 2,
            embed_dim=self.embed_dim,
            flatten_embedding=True,
        )

        cam_dec_dim_in = (2 if self.backbone.cat_token else 1) * self.embed_dim
        self.cam_dec = CameraDec(dim_in=cam_dec_dim_in)

        camera_encoder = config.get("camera_encoder")
        self.cam_enc = (
            None
            if camera_encoder is None
            else CameraEnc(dim_out=self.embed_dim, **camera_encoder)
        )

    def tokenize(
        self,
        flow_predictions,  # final: [E,2,H,W]; info: [E,C,H,W]
        depth_predictions,  # depth/mask: [H,W], [H,W]
        source_intrinsics: torch.Tensor,  # [4]
        target_intrinsics: torch.Tensor,  # [E,4]
        target_depth_predictions,  # depth/mask: [E,H,W], [E,H,W]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flow = flow_predictions["final"]
        flow_info = flow_predictions["info"]
        var_min = flow_predictions["var_min"]
        var_max = flow_predictions["var_max"]

        zero_flow = torch.zeros(1, *flow.shape[1:], device=flow.device, dtype=flow.dtype)
        reference_info = torch.zeros(1, *flow_info.shape[1:], device=flow_info.device, dtype=flow_info.dtype)
        all_flow = torch.cat((zero_flow, flow), dim=0)
        all_info = torch.cat((reference_info, flow_info), dim=0)
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

        depth = depth_predictions["depth"]
        mask = depth_predictions["mask"]
        target_depth = target_depth_predictions["depth"]
        target_mask = target_depth_predictions["mask"]
        source_depthmap = torch.stack([depth, mask.to(depth.dtype)], dim=0)[None]
        target_depthmap = torch.stack(
            (target_depth, target_mask.to(target_depth.dtype)),
            dim=1,
        )
        depth_token = self.depth_patch_embed(
            torch.cat((source_depthmap, target_depthmap), dim=0)
        )
        patch_token = torch.cat((depth_token, motion_token), dim=-1)[None]
        return patch_token, source_depthmap

    def camera_tokens(
        self,
        relative_pose_prior: torch.Tensor,  # [E,7], source normalized-depth gauge
        source_intrinsics: torch.Tensor,  # [4]
        target_intrinsics: torch.Tensor,  # [E,4]
        image_shape: tuple[int, int],
    ) -> torch.Tensor:
        identity = relative_pose_prior.new_zeros((1, 1, 7))
        identity[..., 6] = 1.0
        poses = torch.cat((identity, relative_pose_prior[None]), dim=1)
        intrinsics = torch.cat((source_intrinsics[None], target_intrinsics), dim=0)[None]
        with torch.autocast(device_type=relative_pose_prior.device.type, enabled=False):
            tokens = self.cam_enc(poses.float(), intrinsics.float(), image_shape)
        return tokens.reshape(-1, 1, self.embed_dim)

    def solve(
        self,
        patch_token: torch.Tensor,  # [B,S,H/P,W/P,C]
        source_depthmap: torch.Tensor | None,  # [B,2,H,W]
        source_intrinsics: torch.Tensor,  # [4]
        target_intrinsics: torch.Tensor,  # [E,4]
        image_shape: tuple[int, int],
        use_fp16: bool = False,
        decode_depth: bool = True,
        relative_pose_prior: torch.Tensor | None = None,  # [E,7]
    ):
        backbone_args = {}
        if relative_pose_prior is not None:
            backbone_args["cam_token"] = self.camera_tokens(
                relative_pose_prior,
                source_intrinsics,
                target_intrinsics,
                image_shape,
            )

        with torch.autocast(
            device_type=patch_token.device.type,
            enabled=use_fp16 and patch_token.is_cuda,
        ):
            feats, _ = self.backbone(
                patch_token,
                export_feat_layers=[],
                **backbone_args,
            )

        with torch.autocast(device_type=patch_token.device.type, enabled=False):
            pose_enc, pose_log_variance, relative_log_scale, scale_confidence = (
                self.cam_dec(feats[-1][1][:, 1:])
            )

        output = {
            "pose_enc": pose_enc,
            "pose_log_variance": pose_log_variance,
            "relative_log_scale": relative_log_scale,
            "scale_confidence": scale_confidence,
        }
        if decode_depth:
            res_feat = self.res_depth_embed(source_depthmap)
            with torch.autocast(device_type=patch_token.device.type, enabled=False):
                refined_depth, depth_confidence = self.depth_head(
                    feats,
                    res_feat,
                    img_shape=image_shape,
                )
            output["depth"] = refined_depth
            output["depth_conf"] = depth_confidence
        return output

    def forward(
        self,
        flow_predictions,  # final: [E,2,H,W]; info: [E,C,H,W]
        depth_predictions,  # depth/mask: [H,W], [H,W]
        source_intrinsics: torch.Tensor,  # [4]
        target_intrinsics: torch.Tensor,  # [E,4]
        target_depth_predictions,  # depth/mask: [E,H,W], [E,H,W]
        use_fp16: bool = False,
        decode_depth: bool = True,
        relative_pose_prior: torch.Tensor | None = None,  # [E,7]
    ):
        patch_token, source_depthmap = self.tokenize(
            flow_predictions,
            depth_predictions,
            source_intrinsics,
            target_intrinsics,
            target_depth_predictions,
        )
        output = self.solve(
            patch_token,
            source_depthmap,
            source_intrinsics,
            target_intrinsics,
            tuple(source_depthmap.shape[-2:]),
            use_fp16=use_fp16,
            decode_depth=decode_depth,
            relative_pose_prior=relative_pose_prior,
        )
        return {key: value.squeeze(0) for key, value in output.items()}


class GeNTWrapper(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        self.flow = load_flow()
        self.mono = load_moge('v2')
        self.gnt = GeNT(model_config)

        self.mono.eval()
        for p in self.mono.parameters():
            p.requires_grad = False

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path) -> "GeNTWrapper":
        """
        Load a model from a checkpoint file.

        Args:
            pretrained_model_name_or_path: path to the checkpoint file.

        Returns:
            a new instance of 'GeNTWrapper' with the parameters loaded from the checkpoint.
        """
        ckpt = torch.load(pretrained_model_name_or_path, map_location="cpu", weights_only=False)
        model_config = ckpt["model_config"]
        # Checkpoints store architecture settings; Python package paths belong to the runtime.
        model_config["backbone"]["_target_"] = "gent.model.dinov2.dinov2.DinoV2"
        model = cls(model_config)
        model.load_state_dict(ckpt["model"])
        return model

    def normalize_depth(
        self,
        depth: torch.Tensor,  # [B,H,W]
        mask: torch.Tensor,  # [B,H,W]
        eps=1e-8,
    ):
        masked_depth = depth.masked_fill(~mask, 0)
        valid_count = mask.flatten(1).sum(dim=1)
        valid_scale = valid_count > 0
        scale = masked_depth.flatten(1).sum(dim=1) / valid_count.clamp_min(1)
        scale = torch.where(valid_scale, scale + eps, torch.zeros_like(scale))
        fallback_scale = torch.where(
            valid_scale.any(),
            scale.sum() / valid_scale.sum().clamp_min(1),
            scale.new_tensor(1.0),
        )
        scale = torch.where(valid_scale, scale, fallback_scale)
        return depth / scale[:, None, None], scale

    def _frontend_forward(
        self,
        images: torch.Tensor,  # [1,S,3,H,W]
        intrinsics: torch.Tensor,  # [1,S,4]
        graph,
    ):

        ii, jj, _ = graph_to_edge_list(graph)

        ii = ii.to(device=images.device, dtype=torch.long)
        jj = jj.to(device=images.device, dtype=torch.long)

        B, S, _, H, W = images.shape

        # Monocular depth prior
        fx = intrinsics[0, :, 0]
        fov_x = torch.rad2deg(2 * torch.atan(W / (2 * fx)))
        depth_predictions = self.mono.infer(images[0], fov_x=fov_x, apply_mask=False)
        mono_depths = depth_predictions["depth"].clamp_min(0.01)
        non_sky_mask = depth_predictions["mask"]

        # Predict target-to-reference correspondence flow for each graph edge i -> j.
        disps = mono_depths.reciprocal()
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

        del depth_predictions
        return flow_predictions, mono_depths, non_sky_mask

    def forward(
        self,
        images: torch.Tensor,  # [1,S,3,H,W]
        intrinsics: torch.Tensor,  # [1,S,4]
        graph,
        gt_depth_valid: torch.Tensor,  # [1,S,H,W]
        use_fp16=False,
        depth_normalization_min_cutoff: float = 80.0,
        depth_normalization_quantile: float = 0.8,
    ):
        assert intrinsics.shape == (*images.shape[:2], 4), (
            "intrinsics must provide [fx, fy, cx, cy] for every training view"
        )
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
        flow_predictions, mono_depths, non_sky_mask = self._frontend_forward(
            images,
            intrinsics,
            graph,
        )

        # Keep the normalization support consistent with runtime: use all
        # non-sky pixels up to max(min_cutoff, quantile(non-sky depth)), then
        # intersect it with the dataset-valid mask available during training.
        normalization_mask = build_depth_normalization_mask(
            mono_depths,
            non_sky_mask,
            min_cutoff=depth_normalization_min_cutoff,
            quantile=depth_normalization_quantile,
            valid_mask=gt_depth_valid[0],
        )
        scaled_depth, scale = self.normalize_depth(mono_depths, normalization_mask)

        flow_final, info_final = flow_predictions["flow"][-1], flow_predictions["info"][-1]

        conditioned_groups = set()
        if self.gnt.cam_enc is not None and len(iu_list) >= 2:
            permutation = torch.randperm(len(iu_list))
            conditioned_groups = set(permutation[: len(iu_list) // 2].tolist())

        for group_index, fi in enumerate(iu_list):
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
                "mask": normalization_mask[fi],  # [H,W]
            }
            target_depth = scaled_depth[jj[mask]]
            target_depth_input = {
                "depth": target_depth,  # [E_i,H,W], independently normalized per target view.
                "mask": normalization_mask[jj[mask]],  # [E_i,H,W]
            }

            source_intrinsics = intrinsics[0, fi]
            target_intrinsics = intrinsics[0, jj[mask]]
            patch_token, source_depthmap = self.gnt.tokenize(
                group_flow_predictions,
                depth_input,
                source_intrinsics,
                target_intrinsics,
                target_depth_input,
            )
            image_shape = tuple(source_depthmap.shape[-2:])

            if group_index in conditioned_groups:
                with torch.no_grad():
                    first_output = self.gnt.solve(
                        patch_token,
                        source_depthmap,
                        source_intrinsics,
                        target_intrinsics,
                        image_shape,
                        use_fp16=use_fp16,
                        decode_depth=False,
                    )
                output_geo = self.gnt.solve(
                    patch_token,
                    source_depthmap,
                    source_intrinsics,
                    target_intrinsics,
                    image_shape,
                    use_fp16=use_fp16,
                    decode_depth=True,
                    relative_pose_prior=first_output["pose_enc"][0].detach(),
                )
            else:
                output_geo = self.gnt.solve(
                    patch_token,
                    source_depthmap,
                    source_intrinsics,
                    target_intrinsics,
                    image_shape,
                    use_fp16=use_fp16,
                    decode_depth=True,
                )

            depths[fi] = output_geo["depth"][0]
            depths_conf[fi] = output_geo["depth_conf"][0]
            edge_measurements[mask, :7] = output_geo["pose_enc"][0]
            edge_measurements[mask, 7:9] = output_geo["pose_log_variance"][0]
            edge_measurements[mask, 9] = output_geo["relative_log_scale"][0]
            edge_measurements[mask, 10] = output_geo["scale_confidence"][0]

        predictions = {
            "edge_measurements": edge_measurements[None],  # 1,E,11
            "depth": depths[None],  # 1,S,H,W
            "depth_conf": depths_conf[None],  # 1,S,H,W
            "normalization_mask": normalization_mask[None],  # 1,S,H,W
            "scale": scale[None],  # (B,S)
            "flow_predictions": flow_predictions,
            "mono_depth": mono_depths[None],  # 1,S,H,W
        }

        return predictions
