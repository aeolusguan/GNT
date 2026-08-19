from dataclasses import dataclass
from pathlib import Path

import torch

from gent.model import GeNTWrapper


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


class GeNTMeasurements:
    """Inference-time GeNT measurement helper."""

    @classmethod
    def build(
        cls,
        checkpoint_path: str | Path,
        *,
        device: torch.device,
        use_fp16: bool,
    ) -> "GeNTMeasurements":
        net = GeNTWrapper.from_pretrained(checkpoint_path).eval().to(device)
        return cls(net, device, use_fp16=use_fp16)

    def __init__(
        self,
        net: GeNTWrapper,
        device: torch.device,
        *,
        use_fp16: bool,
    ):
        self.net = net
        self.device = device
        self.use_fp16 = bool(use_fp16)
        self._unit_flow_bases: torch.Tensor | None = None

    @property
    def depth_token_dim(self) -> int:
        return int(self.net.gnt.depth_embed_dim)

    @torch.no_grad()
    def encode_frame(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Encode a frame into fmap, depth prior, depth token, scale, mask, and bases."""
        fmap, mono_depth, non_sky_mask = self._moge_prior(images, intrinsics)
        normalization_mask = non_sky_mask & (mono_depth < 80)
        normed_depth, scale = self.net.normalize_depth(mono_depth, normalization_mask)
        depth_token = self.net.gnt.depth_patch_embed(
            torch.stack(
                (normed_depth, normalization_mask.to(normed_depth.dtype)),
                dim=1,
            )
        )
        bases = self._encode_bases(mono_depth)
        return (
            fmap,
            normed_depth,
            depth_token,
            scale,
            normalization_mask,
            bases,
        )

    @torch.no_grad()
    def _moge_prior(
        self,
        images: torch.Tensor,  # [S,3,H,W]
        intrinsics: torch.Tensor,  # [S,4]
    ):
        _, _, _height, width = images.shape
        fx = intrinsics[:, 0]
        fov_x = torch.rad2deg(2 * torch.atan(width / (2 * fx)))
        depth_predictions = self.net.mono.infer(images, fov_x=fov_x, apply_mask=False)
        mono_depths = depth_predictions["depth"]
        mono_feats = depth_predictions["feature"]
        valid = depth_predictions["mask"]

        images = 2 * images - 1.0
        fmap_8x = self.net.flow.fnet(images)
        mono_8x = self.net.flow.merge_head(mono_feats)
        return torch.cat((fmap_8x, mono_8x), dim=1), mono_depths, valid

    @torch.no_grad()
    def _encode_bases(
        self,
        depths: torch.Tensor,  # [1,H,W]
    ):
        depths = depths.clamp_min(0.01)
        disps = depths.reciprocal().unsqueeze(1)
        if self._unit_flow_bases is None:
            self._unit_flow_bases = self.net.flow.create_bases(torch.ones_like(disps))

        # Tx, Ty, Tz occupy the first six channels and scale with disparity.
        # Rotation channels depend only on the fixed image geometry.
        bases = torch.cat(
            (
                self._unit_flow_bases[:, :6] * disps,
                self._unit_flow_bases[:, 6:],
            ),
            dim=1,
        )
        return self.net.flow.init_decoder.patch_embed_base(bases)

    def gent_patch_embed(
        self,
        depth_token: torch.Tensor,  # [N,H/P,W/P,D/4]
        flow: torch.Tensor,  # [N,2,H,W]
        info: torch.Tensor,  # [N,C,H,W]
        source_intrinsics: torch.Tensor,  # [N,4]
        target_intrinsics: torch.Tensor,  # [N,4]
    ) -> torch.Tensor:
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
    def forward_gent(
        self,
        patch_token: torch.Tensor,  # [B,S,H/P,W/P,C]
        source_intrinsics: torch.Tensor,  # [4] or [B,4]
        target_intrinsics: torch.Tensor,  # [S-1,4] or [B,4]
        *,
        depth_head_input: torch.Tensor | None = None,  # [B,2,H,W]
        image_shape: tuple[int, int],
        decode_depth: bool = True,
        relative_pose_prior: torch.Tensor | None = None,  # [S-1,7]
    ):
        prediction = self.net.gnt.solve(
            patch_token,
            depth_head_input,
            source_intrinsics,
            target_intrinsics,
            image_shape,
            use_fp16=self.use_fp16,
            decode_depth=decode_depth,
            relative_pose_prior=relative_pose_prior,
        )
        output = {
            "pose_enc": prediction["pose_enc"],
            "pose_confidence": prediction["pose_log_variance"],
            "relative_log_scale": prediction["relative_log_scale"],
            "scale_confidence": prediction["scale_confidence"],
        }
        if decode_depth:
            output["depth"] = prediction["depth"]
            output["depth_conf"] = prediction["depth_conf"]
        return output

    @torch.no_grad()
    def predict_flow(
        self,
        source_fmap: torch.Tensor,  # [1 or E,C,H/8,W/8]
        target_fmap: torch.Tensor,  # [E,C,H/8,W/8]
        target_bases: torch.Tensor,  # [E,C,H/P,W/P]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_fmap = source_fmap.expand(target_fmap.shape[0], -1, -1, -1)
        with torch.amp.autocast(device_type=self.device.type, enabled=False):
            flow, info = self.net.flow.infer(
                target_fmap.float().contiguous(),
                source_fmap.float().contiguous(),
                target_bases.float().contiguous(),
            )
        return flow, info

    def make_pose_measurement(
        self,
        ii: torch.Tensor,  # [E]
        jj: torch.Tensor,  # [E]
        pose: torch.Tensor,  # [E,7]
        pose_confidence: torch.Tensor,  # [E,2]
        relative_log_scale: torch.Tensor,  # [E]
        scale_confidence: torch.Tensor,  # [E]
    ) -> PoseMeasurement:
        pose = pose.float()
        pose[:, 3:7] = pose[:, 3:7] / pose[:, 3:7].norm(dim=-1, keepdim=True).clamp_min(1e-6)
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

    def _pose_depth_tokens(
        self,
        flow: torch.Tensor,  # [E,2,H,W]
        info: torch.Tensor,  # [E,C,H,W]
        source_intrinsics: torch.Tensor,  # [4]
        target_intrinsics: torch.Tensor,  # [E,4]
        source_depth_token: torch.Tensor,  # [H/P,W/P,D/4]
        target_depth_token: torch.Tensor,  # [E,H/P,W/P,D/4]
        source_depth: torch.Tensor,  # [H,W]
        source_mask: torch.Tensor,  # [H,W]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zero_flow = torch.zeros(1, *flow.shape[1:], device=flow.device, dtype=flow.dtype)
        zero_info = torch.zeros(1, *info.shape[1:], device=info.device, dtype=info.dtype)
        source_depthmap = torch.stack([source_depth, source_mask.to(source_depth.dtype)], dim=0)[None]
        reference_intrinsics = source_intrinsics[None]
        patch_token = self.gent_patch_embed(
            torch.cat((source_depth_token[None], target_depth_token), dim=0),
            torch.cat((zero_flow, flow), dim=0),
            torch.cat((zero_info, info), dim=0),
            torch.cat((reference_intrinsics, reference_intrinsics.expand(flow.shape[0], -1)), dim=0),
            torch.cat((reference_intrinsics, target_intrinsics), dim=0),
        )[None]
        return patch_token, source_depthmap

    def _make_pose_depth_measurement(
        self,
        ii: torch.Tensor,  # [E]
        jj: torch.Tensor,  # [E]
        output: dict[str, torch.Tensor],
    ) -> PoseDepthMeasurement:
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

    def predict_multiview(
        self,
        ii: torch.Tensor,
        jj: torch.Tensor,
        *,
        flow: torch.Tensor,
        info: torch.Tensor,
        source_depth: torch.Tensor,
        source_mask: torch.Tensor,
        source_depth_token: torch.Tensor,
        target_depth_token: torch.Tensor,
        source_intrinsics: torch.Tensor,
        target_intrinsics: torch.Tensor,
        decode_depth: bool,
        relative_pose_prior: torch.Tensor | None = None,
    ) -> PoseMeasurement | PoseDepthMeasurement:
        """Predict one source against multiple targets from explicit geometry inputs."""
        assert ii.shape == jj.shape
        assert ii.numel() > 0
        patch_token, source_depthmap = self._pose_depth_tokens(
            flow,
            info,
            source_intrinsics,
            target_intrinsics,
            source_depth_token,
            target_depth_token,
            source_depth,
            source_mask,
        )
        output = self.forward_gent(
            patch_token,
            source_intrinsics,
            target_intrinsics,
            depth_head_input=source_depthmap,
            image_shape=tuple(source_depth.shape[-2:]),
            decode_depth=decode_depth,
            relative_pose_prior=relative_pose_prior,
        )
        if decode_depth:
            return self._make_pose_depth_measurement(ii, jj, output)
        return self.make_pose_measurement(
            ii,
            jj,
            output["pose_enc"].squeeze(0),
            output["pose_confidence"].squeeze(0),
            output["relative_log_scale"].squeeze(0),
            output["scale_confidence"].squeeze(0),
        )

    def predict_pairwise_pose(
        self,
        ii: torch.Tensor,
        jj: torch.Tensor,
        *,
        flow: torch.Tensor,
        info: torch.Tensor,
        source_depth_token: torch.Tensor,
        target_depth_token: torch.Tensor,
        source_intrinsics: torch.Tensor,
        target_intrinsics: torch.Tensor,
        image_shape: tuple[int, int],
    ) -> PoseMeasurement:
        zero_flow = torch.zeros_like(flow)
        zero_info = torch.zeros_like(info)
        flat_tokens = self.gent_patch_embed(
            torch.stack((source_depth_token, target_depth_token), dim=1).reshape(
                -1, *source_depth_token.shape[1:]
            ),
            torch.stack((zero_flow, flow), dim=1).reshape(-1, *flow.shape[1:]),
            torch.stack((zero_info, info), dim=1).reshape(-1, *info.shape[1:]),
            torch.stack((source_intrinsics, source_intrinsics), dim=1).reshape(-1, 4),
            torch.stack((source_intrinsics, target_intrinsics), dim=1).reshape(-1, 4),
        )
        output = self.forward_gent(
            flat_tokens.reshape(flow.shape[0], 2, *flat_tokens.shape[1:]),
            source_intrinsics,
            target_intrinsics,
            image_shape=image_shape,
            decode_depth=False,
        )
        return self.make_pose_measurement(
            ii,
            jj,
            output["pose_enc"].squeeze(1),
            output["pose_confidence"].squeeze(1),
            output["relative_log_scale"].squeeze(1),
            output["scale_confidence"].squeeze(1),
        )
