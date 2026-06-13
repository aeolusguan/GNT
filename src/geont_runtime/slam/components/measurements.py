import torch

from geont.models import GeoNTWrapper
from .buffer import GraphBuffer, KeyframeCandidate


class GeoNTMeasurements:
    """Inference-time GeoNT measurement helper."""

    def __init__(
        self,
        net: GeoNTWrapper,
        buffer: GraphBuffer,
        device: torch.device,
        *,
        use_fp16: bool,
    ):
        self.net = net
        self.buffer = buffer
        self.device = device
        self.use_fp16 = bool(use_fp16)

    def _backbone_kwargs(self, camera_prior: torch.Tensor | None, n_views: int) -> dict:
        if camera_prior is None:
            return {}
        assert camera_prior.shape == (n_views, 1, self.net.gnt.embed_dim)
        return {"cam_token": camera_prior}

    @torch.no_grad()
    def patch_embed(self, ii: torch.Tensor, jj: torch.Tensor) -> torch.Tensor:
        fmap1 = self.buffer.fmaps[ii, 0].float().contiguous()
        fmap2 = self.buffer.fmaps[jj, 0].float().contiguous()
        bases = self.buffer.bases[ii, 0].float().contiguous()
        intrinsics = self.buffer.intrinsics[0:1].expand(fmap1.shape[0], -1).contiguous()
        with torch.amp.autocast(device_type=self.device.type, enabled=False):
            return self.net.encode_flow(fmap1, fmap2, bases, intrinsics)

    @torch.no_grad()
    def patch_embed_candidate(self, keyframe_candidate: KeyframeCandidate, jj: torch.Tensor) -> torch.Tensor:
        fmap1 = keyframe_candidate.fmap.expand(jj.numel(), -1, -1, -1).float().contiguous()
        fmap2 = self.buffer.fmaps[jj, 0].float().contiguous()
        bases = keyframe_candidate.bases.expand(jj.numel(), -1, -1, -1).float().contiguous()
        intrinsics = keyframe_candidate.intrinsics.expand(jj.numel(), -1).contiguous()
        with torch.amp.autocast(device_type=self.device.type, enabled=False):
            return self.net.encode_flow(fmap1, fmap2, bases, intrinsics)

    @torch.no_grad()
    def coarse_flow_motion_score(self, source: int, keyframe_candidate: KeyframeCandidate) -> float:
        fmap1 = self.buffer.fmaps[source : source + 1, 0].float().contiguous()
        fmap2 = keyframe_candidate.fmap.float().contiguous()
        bases = self.buffer.bases[source : source + 1, 0].float().contiguous()
        with torch.amp.autocast(device_type=self.device.type, enabled=False):
            delta, _ = self.net.flow_init(fmap1, fmap2, bases)
        dense_flow = delta.norm(dim=1)
        return float(dense_flow.mean(dim=(1, 2)).max().item())

    @torch.no_grad()
    def _run_one_view_pose_from_motion_tokens(
        self,
        motion_tokens: torch.Tensor,
        depth: torch.Tensor,
        mask: torch.Tensor,
        *,
        camera_prior: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert depth.ndim == 3
        assert mask.shape == depth.shape
        assert motion_tokens.shape[0] == depth.shape[0]

        depthmap = torch.stack([depth, mask.to(depth.dtype)], dim=1)
        depth_token = self.net.gnt.depth_patch_embed(depthmap)
        patch_token = torch.cat((depth_token, motion_tokens), dim=-1)[:, None]
        backbone_kwargs = self._backbone_kwargs(camera_prior, int(motion_tokens.shape[0]))
        with torch.autocast(device_type=patch_token.device.type, enabled=self.use_fp16 and patch_token.is_cuda):
            feats, _ = self.net.gnt.backbone(
                patch_token,
                export_feat_layers=[],
                **backbone_kwargs,
            )
        with torch.autocast(device_type=patch_token.device.type, enabled=False):
            pose_enc, pose_confidence = self.net.gnt.cam_dec(feats[-1][1])
        return pose_enc.squeeze(1), pose_confidence.squeeze(1)

    @torch.no_grad()
    def run_multiview_pose_depth_from_motion_tokens(
        self,
        motion_tokens: torch.Tensor,
        depth: torch.Tensor,
        mask: torch.Tensor,
        *,
        camera_prior: torch.Tensor | None = None,
    ) -> dict:
        assert depth.ndim == 2
        assert mask.shape == depth.shape

        depthmap = torch.stack([depth, mask.to(depth.dtype)], dim=0)[None]
        depth_token = self.net.gnt.depth_patch_embed(depthmap)
        depth_token = depth_token.expand(motion_tokens.shape[0], -1, -1, -1)
        patch_token = torch.cat((depth_token, motion_tokens), dim=-1)[None]
        backbone_kwargs = self._backbone_kwargs(camera_prior, int(motion_tokens.shape[0]))
        with torch.autocast(device_type=patch_token.device.type, enabled=self.use_fp16 and patch_token.is_cuda):
            feats, _ = self.net.gnt.backbone(
                patch_token,
                export_feat_layers=[],
                **backbone_kwargs,
            )

        with torch.autocast(device_type=patch_token.device.type, enabled=False):
            pose_enc, pose_confidence = self.net.gnt.cam_dec(feats[-1][1])

        res_feat = self.net.gnt.res_depth_embed(depthmap)
        height, width = depth.shape[-2:]
        with torch.autocast(device_type=patch_token.device.type, enabled=False):
            refined_depth, depth_confidence = self.net.gnt.depth_head(
                feats,
                res_feat,
                img_shape=(height, width),
            )
        return {
            "pose": pose_enc.squeeze(0),
            "pose_confidence": pose_confidence.squeeze(0),
            "refined_depth": refined_depth.squeeze(0),
            "depth_confidence": depth_confidence.squeeze(0),
        }

    def make_pose_measurement(
        self,
        ii: torch.Tensor,
        jj: torch.Tensor,
        pose: torch.Tensor,
        confidence: torch.Tensor,
    ) -> dict:
        pose = pose.float()
        pose[:, 3:7] = pose[:, 3:7] / pose[:, 3:7].norm(dim=-1, keepdim=True).clamp_min(1e-6)
        assert confidence.shape[0] == int(ii.numel())
        conf = 1 - 1 / confidence.float()
        conf[:, 1] = 1.0
        return {
            "ii": ii,
            "jj": jj,
            "relative_pose": pose,
            "confidence": conf,
        }

    def predict_pose_edges(
        self,
        ii: torch.Tensor,
        jj: torch.Tensor,
        *,
        camera_prior: torch.Tensor | None = None,
    ) -> dict:
        motion_tokens = self.patch_embed(ii, jj)
        depth = self.buffer.depths_sens_normed[ii, 0].float()
        mask = self.buffer.non_sky_masks[ii, 0]
        pose, confidence = self._run_one_view_pose_from_motion_tokens(
            motion_tokens,
            depth,
            mask,
            camera_prior=camera_prior,
        )
        return self.make_pose_measurement(ii, jj, pose, confidence)

    def measure_multiview_pose_depth(
        self,
        source: int,
        neighbors: torch.Tensor,
    ) -> dict:
        assert neighbors.numel() > 0
        ii = torch.full_like(neighbors, int(source))
        motion_tokens = self.patch_embed(ii, neighbors)
        depth = self.buffer.depths_sens_normed[source, 0].float()
        mask = self.buffer.non_sky_masks[source, 0]
        output = self.run_multiview_pose_depth_from_motion_tokens(
            motion_tokens,
            depth,
            mask,
        )
        result = self.make_pose_measurement(ii, neighbors, output["pose"], output["pose_confidence"])
        result.update(
            {
                "refined_depth": output["refined_depth"],
                "depth_confidence": output["depth_confidence"],
            }
        )
        return result
