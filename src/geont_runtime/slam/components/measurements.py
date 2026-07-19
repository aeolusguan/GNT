from dataclasses import dataclass

import torch

from geont.models import GeoNTWrapper
from .buffer import GraphBuffer, KeyframeCandidate


@dataclass
class PoseMeasurement:
    ii: torch.Tensor
    jj: torch.Tensor
    relative_pose: torch.Tensor
    relative_log_scale: torch.Tensor
    confidence: torch.Tensor


@dataclass
class PoseDepthMeasurement(PoseMeasurement):
    refined_depth: torch.Tensor
    depth_confidence: torch.Tensor


class GeoNTMeasurements:
    """Inference-time GeoNT measurement helper."""

    def __init__(
        self,
        net: GeoNTWrapper,
        buffer: GraphBuffer | None,
        device: torch.device,
        *,
        use_fp16: bool,
    ):
        self.net = net
        self.buffer = buffer
        self.device = device
        self.use_fp16 = bool(use_fp16)

    @torch.no_grad()
    def moge_prior(self, images: torch.Tensor, intrinsics: torch.Tensor):
        _, _, _height, width = images.shape
        fx = intrinsics[:, 0]
        fov_x = torch.rad2deg(2 * torch.atan(width / (2 * fx)))
        depth_predictions = self.net.mono.infer(images, fov_x=fov_x)
        mono_depths = depth_predictions["depth"]
        mono_feats = depth_predictions["feature"]
        valid = depth_predictions["mask"]

        images = 2 * images - 1.0
        fmap_8x = self.net.flow.fnet(images)
        mono_8x = self.net.flow.merge_head(mono_feats)
        return torch.cat((fmap_8x, mono_8x), dim=1), mono_depths, valid

    @torch.no_grad()
    def encode_bases(self, depths: torch.Tensor, valid: torch.Tensor):
        depths = depths.clamp_min(0.01)
        disps = torch.zeros_like(depths)
        disps[valid] = 1.0 / depths[valid]
        bases = self.net.flow.create_bases(disps.unsqueeze(1))
        return self.net.flow.init_decoder.patch_embed_base(bases)

    def geont_patch_embed(
        self,
        depthmap: torch.Tensor,
        flow: torch.Tensor,
        info: torch.Tensor,
        source_intrinsics: torch.Tensor,
        target_intrinsics: torch.Tensor,
    ) -> torch.Tensor:
        assert depthmap.shape[0] == flow.shape[0]
        assert info.shape[0] == flow.shape[0]
        assert source_intrinsics.shape == target_intrinsics.shape == (flow.shape[0], 4)
        depth_token = self.net.gnt.depth_patch_embed(depthmap)
        flow_token = self.net.gnt.motion_patch_embed(
            flow,
            info,
            source_intrinsics,
            target_intrinsics,
            var_min=self.net.flow.args.var_min,
            var_max=self.net.flow.args.var_max,
        )
        return torch.cat((depth_token, flow_token), dim=-1)

    @torch.no_grad()
    def forward_geont(
        self,
        patch_token: torch.Tensor,
        *,
        depth_head_input: torch.Tensor | None = None,
        image_shape: tuple[int, int] | None = None,
        decode_depth: bool = True,
        relative_pose_prior: torch.Tensor | None = None,
        relative_pose_prior_mask: torch.Tensor | None = None,
    ):
        backbone_kwargs = {}
        if relative_pose_prior is not None:
            prior_token = self.net.gnt.encode_relative_pose_prior(relative_pose_prior, relative_pose_prior_mask)
            reference_pose_prior = relative_pose_prior.new_zeros((1, 7))
            reference_pose_prior[:, 6] = 1.0
            reference_token = self.net.gnt.cam_enc(reference_pose_prior, self.net.gnt.backbone.pretrained.camera_token[:, :1, :])
            backbone_kwargs["cam_token"] = torch.cat((reference_token, prior_token), dim=0)
        with torch.autocast(device_type=patch_token.device.type, enabled=self.use_fp16 and patch_token.is_cuda):
            feats, _ = self.net.gnt.backbone(
                patch_token,
                export_feat_layers=[],
                **backbone_kwargs,
            )

        with torch.autocast(device_type=patch_token.device.type, enabled=False):
            pose_enc, pose_confidence, relative_log_scale, scale_confidence = self.net.gnt.cam_dec(feats[-1][1][:, 1:])

        output = {
            "pose_enc": pose_enc,
            "pose_confidence": pose_confidence,
            "relative_log_scale": relative_log_scale,
            "scale_confidence": scale_confidence,
        }
        if decode_depth:
            assert depth_head_input is not None
            assert image_shape is not None
            height, width = image_shape
            res_feat = self.net.gnt.res_depth_embed(depth_head_input)
            with torch.autocast(device_type=patch_token.device.type, enabled=False):
                refined_depth, depth_confidence = self.net.gnt.depth_head(feats, res_feat, img_shape=(height, width))
            output["depth"] = refined_depth
            output["depth_conf"] = depth_confidence
        return output

    @torch.no_grad()
    def _buffer_edge_flow(
        self,
        ii: torch.Tensor,
        jj: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_fmap = self.buffer.fmaps[ii, 0]
        target_fmap = self.buffer.fmaps[jj, 0]
        target_bases = self.buffer.bases[jj, 0]
        with torch.amp.autocast(device_type=self.device.type, enabled=False):
            flow, info = self.net.flow.infer(
                target_fmap.float().contiguous(),
                source_fmap.float().contiguous(),
                target_bases.float().contiguous(),
            )
        return flow, info

    @torch.no_grad()
    def _candidate_edge_flow(
        self,
        keyframe_candidate: KeyframeCandidate,
        jj: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_fmap = keyframe_candidate.fmap.expand(jj.numel(), -1, -1, -1)
        target_fmap = self.buffer.fmaps[jj, 0]
        target_bases = self.buffer.bases[jj, 0]
        with torch.amp.autocast(device_type=self.device.type, enabled=False):
            flow, info = self.net.flow.infer(
                target_fmap.float().contiguous(),
                source_fmap.float().contiguous(),
                target_bases.float().contiguous(),
            )
        return flow, info

    def make_pose_measurement(
        self,
        ii: torch.Tensor,
        jj: torch.Tensor,
        pose: torch.Tensor,
        pose_confidence: torch.Tensor,
        relative_log_scale: torch.Tensor,
        scale_confidence: torch.Tensor,
    ) -> PoseMeasurement:
        pose = pose.float()
        pose[:, 3:7] = pose[:, 3:7] / pose[:, 3:7].norm(dim=-1, keepdim=True).clamp_min(1e-6)
        assert pose_confidence.shape[0] == int(ii.numel())
        assert relative_log_scale.shape == ii.shape
        assert scale_confidence.shape == ii.shape
        conf = torch.zeros((ii.numel(), 3), device=pose.device, dtype=torch.float32)
        conf[:, :2] = 1 - 1 / pose_confidence.float()
        conf[:, 1] = 1.0
        conf[:, 2] = 1.0
        return PoseMeasurement(
            ii=ii,
            jj=jj,
            relative_pose=pose,
            relative_log_scale=relative_log_scale.float(),
            confidence=conf,
        )

    def predict_pose_depth_impl(
        self,
        ii: torch.Tensor,
        jj: torch.Tensor,
        flow: torch.Tensor,
        info: torch.Tensor,
        source_intrinsics: torch.Tensor,
        target_intrinsics: torch.Tensor,
        source_depth: torch.Tensor,
        source_mask: torch.Tensor,
        target_depth: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> PoseDepthMeasurement:
        zero_flow = torch.zeros(1, *flow.shape[1:], device=flow.device, dtype=flow.dtype)
        zero_info = torch.zeros(1, *info.shape[1:], device=info.device, dtype=info.dtype)
        source_depthmap = torch.stack([source_depth, source_mask.to(source_depth.dtype)], dim=0)[None]
        target_depthmap = torch.stack([target_depth, target_mask.to(target_depth.dtype)], dim=1)
        reference_intrinsics = source_intrinsics[None]
        patch_token = self.geont_patch_embed(
            torch.cat((source_depthmap, target_depthmap), dim=0),
            torch.cat((zero_flow, flow), dim=0),
            torch.cat((zero_info, info), dim=0),
            torch.cat((reference_intrinsics, reference_intrinsics.expand(flow.shape[0], -1)), dim=0),
            torch.cat((reference_intrinsics, target_intrinsics), dim=0),
        )[None]
        output = self.forward_geont(
            patch_token,
            depth_head_input=source_depthmap,
            image_shape=tuple(source_depth.shape[-2:]),
        )
        measurement = self.make_pose_measurement(
            ii,
            jj,
            output["pose_enc"].squeeze(0),
            output["pose_confidence"].squeeze(0),
            output["relative_log_scale"].squeeze(0),
            output["scale_confidence"].squeeze(0),
        )
        return PoseDepthMeasurement(
            ii=measurement.ii,
            jj=measurement.jj,
            relative_pose=measurement.relative_pose,
            relative_log_scale=measurement.relative_log_scale,
            confidence=measurement.confidence,
            refined_depth=output["depth"].squeeze(0),
            depth_confidence=output["depth_conf"].squeeze(0),
        )

    def predict_pose_edges(
        self,
        ii: torch.Tensor,
        jj: torch.Tensor,
    ) -> PoseMeasurement:
        flow, info = self._buffer_edge_flow(ii, jj)
        intrinsics = self.buffer.intrinsics[0:1].expand(ii.numel(), -1).contiguous()
        source_depth = self.buffer.depths_sens_normed[ii, 0].float()
        source_mask = self.buffer.non_sky_masks[ii, 0]
        # target_scale_ratio = self.buffer.depths_sens_scale[jj, 0].float() / self.buffer.depths_sens_scale[ii, 0].float()
        target_depth = self.buffer.depths_sens_normed[jj, 0].float()
        target_mask = self.buffer.non_sky_masks[jj, 0]
        zero_flow = torch.zeros_like(flow)
        zero_info = torch.zeros_like(info)
        source_depthmap = torch.stack([source_depth, source_mask.to(source_depth.dtype)], dim=1)
        target_depthmap = torch.stack([target_depth, target_mask.to(target_depth.dtype)], dim=1)
        flat_tokens = self.geont_patch_embed(
            torch.stack((source_depthmap, target_depthmap), dim=1).reshape(-1, 2, *source_depth.shape[-2:]),
            torch.stack((zero_flow, flow), dim=1).reshape(-1, *flow.shape[1:]),
            torch.stack((zero_info, info), dim=1).reshape(-1, *info.shape[1:]),
            torch.stack((intrinsics, intrinsics), dim=1).reshape(-1, 4),
            torch.stack((intrinsics, intrinsics), dim=1).reshape(-1, 4),
        )
        output = self.forward_geont(flat_tokens.reshape(flow.shape[0], 2, *flat_tokens.shape[1:]), decode_depth=False)
        return self.make_pose_measurement(
            ii,
            jj,
            output["pose_enc"].squeeze(1),
            output["pose_confidence"].squeeze(1),
            output["relative_log_scale"].squeeze(1),
            output["scale_confidence"].squeeze(1),
        )

    def predict_candidate_pose_depth(
        self,
        source: int,
        keyframe_candidate: KeyframeCandidate,
        neighbors: torch.Tensor,
    ) -> tuple[PoseDepthMeasurement, torch.Tensor]:
        assert neighbors.numel() > 0
        ii = torch.full_like(neighbors, int(source))
        flow, info = self._candidate_edge_flow(keyframe_candidate, neighbors)
        source_intrinsics = keyframe_candidate.intrinsics[0]
        target_intrinsics = self.buffer.intrinsics[0:1].expand(neighbors.numel(), -1).contiguous()
        source_depth = keyframe_candidate.depth_sens_normed[0].float()
        source_mask = keyframe_candidate.mask[0]
        # target_scale_ratio = self.buffer.depths_sens_scale[neighbors, 0].float() / keyframe_candidate.scale[0].float()
        target_depth = self.buffer.depths_sens_normed[neighbors, 0].float()
        target_mask = self.buffer.non_sky_masks[neighbors, 0]
        measurement = self.predict_pose_depth_impl(
            ii,
            neighbors,
            flow,
            info,
            source_intrinsics,
            target_intrinsics,
            source_depth,
            source_mask,
            target_depth,
            target_mask,
        )
        return (
            measurement,
            flow.norm(dim=1).mean(dim=(1, 2)),
        )

    def predict_pose_depth(
        self,
        source: int,
        neighbors: torch.Tensor,
    ) -> PoseDepthMeasurement:
        assert neighbors.numel() > 0
        ii = torch.full_like(neighbors, int(source))
        flow, info = self._buffer_edge_flow(ii, neighbors)
        source_intrinsics = self.buffer.intrinsics[0]
        target_intrinsics = self.buffer.intrinsics[0:1].expand(neighbors.numel(), -1).contiguous()
        source_depth = self.buffer.depths_sens_normed[source, 0].float()
        source_mask = self.buffer.non_sky_masks[source, 0]
        # target_scale_ratio = self.buffer.depths_sens_scale[neighbors, 0].float() / self.buffer.depths_sens_scale[source, 0].float()
        target_depth = self.buffer.depths_sens_normed[neighbors, 0].float()
        target_mask = self.buffer.non_sky_masks[neighbors, 0]
        return self.predict_pose_depth_impl(
            ii,
            neighbors,
            flow,
            info,
            source_intrinsics,
            target_intrinsics,
            source_depth,
            source_mask,
            target_depth,
            target_mask,
        )
