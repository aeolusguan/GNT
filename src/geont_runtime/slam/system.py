# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import uuid
from pathlib import Path

import numpy as np
import rerun as rr
import torch
from einops import rearrange
from omegaconf import DictConfig, OmegaConf

from lietorch import SE3
from geont_runtime.streams.base import FrameAttribute, ProcessedVideoStream, StreamProcessor, VideoFrame, VideoStream
from geont_runtime.utils.logging import pbar
from geont.models import GeoNTWrapper

from .components.buffer import GraphBuffer
from .components.frontend import SLAMFrontend
from .components.initializer import OnePassInitializer
from .components.motion_filter import MotionFilter
from .frontend_snapshot import save_initializer_snapshot
from .interface import SLAMOutput


class StandardResizeStreamProcessor(StreamProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.fac_x, self.fac_y = 1.0, 1.0

    def _compute_frame_size_crop(self, previous_frame_size: tuple[int, int]):
        h0, w0 = previous_frame_size
        scale_factor = np.sqrt((384 * 512) / (h0 * w0))
        h1 = int(h0 * scale_factor)
        w1 = int(w0 * scale_factor)

        crop_h, crop_w = h1 % 16, w1 % 16
        crop_top, crop_bottom = crop_h // 2, crop_h - crop_h // 2
        crop_left, crop_right = crop_w // 2, crop_w - crop_w // 2

        self.fac_x, self.fac_y = w0 / w1, h0 / h1
        self.scx, self.scy = crop_left, crop_top
        return (h1, w1), (crop_top, crop_bottom, crop_left, crop_right)
    
    def update_frame_size(self, previous_frame_size: tuple[int, int]):
        (h1, w1), (crop_top, crop_bottom, crop_left, crop_right) = self._compute_frame_size_crop(previous_frame_size)
        return h1 - (crop_top + crop_bottom), w1 - (crop_left + crop_right)
    
    def __call__(self, frame_idx: int, frame_data: VideoFrame) -> VideoFrame:
        (h1, w1), (crop_top, crop_bottom, crop_left, crop_right) = self._compute_frame_size_crop(frame_data.size())
        frame_data = frame_data.resize((h1, w1))
        frame_data = frame_data.crop(top=crop_top, bottom=crop_bottom, left=crop_left, right=crop_right)
        return frame_data
    
    def recover_intrinsics(self, after_intrinsics: torch.Tensor) -> torch.Tensor:
        fx, fy, cx, cy = after_intrinsics
        return torch.stack(
            [
                fx * self.fac_x,
                fy * self.fac_y,
                (cx + self.scx) * self.fac_x,
                (cy + self.scy) * self.fac_y,
            ]
        )


class FixedIntrinsicsStreamProcessor(StreamProcessor):
    def __init__(self, intrinsics) -> None:
        super().__init__()
        self.intrinsics = torch.as_tensor(intrinsics, dtype=torch.float)

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes.union({FrameAttribute.INTRINSICS})

    def __call__(self, frame_idx: int, frame_data: VideoFrame) -> VideoFrame:
        frame_data.intrinsics = self.intrinsics.to(device=frame_data.rgb.device)
        return frame_data
    

class SLAMSystem:

    def __init__(self, device: torch.device, config: DictConfig) -> None:
        self.device = device
        self.visualize = config.visualize
        self.config = config.copy()
        OmegaConf.set_struct(self.config, False)

    def _build_components(self):
        self.gnt = GeoNTWrapper.from_pretrained(self.config.ckpt_path).eval().to(self.device)
        self.buffer = GraphBuffer(
            height=self.config.height,
            width=self.config.width,
            buffer_size=self.config.buffer,
            device=self.device
        )
        self.motion_filter = MotionFilter(
            self.gnt,
            thresh=self.config.filter_thresh,
            device=self.device,
        )
        self.initializer = OnePassInitializer(
            self.gnt,
            self.buffer,
            self.config,
            device=self.device,
        )
        self.frontend = SLAMFrontend(
            self.gnt,
            self.buffer,
            self.config,
            device=self.device,
        )

    def _add_keyframe(
        self,
        frame_idx: int,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        phase: int,
        features: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        bases: torch.Tensor | None = None,
    ):
        assert phase in [1, 2]
        kf_idx = self.buffer.n_frames
        if kf_idx >= self.buffer.poses.shape[0]:
            raise RuntimeError(
                f"SLAM buffer is full at {kf_idx} keyframes; increase slam.buffer in the config."
            )
        self.buffer.tstamp[kf_idx] = frame_idx
        if features is None:
            gmap, mono_depth, non_sky_mask = self.gnt.encode_features(images, intrinsics)
        else:
            gmap, mono_depth, non_sky_mask = features
        self.buffer.fmaps[kf_idx] = gmap
        normed_depth, scale = self.gnt.normalize_depth(mono_depth, non_sky_mask)
        if bases is None:
            bases = self.gnt.encode_bases(mono_depth, non_sky_mask)
        depth_prior = normed_depth.float()
        self.buffer.depths[kf_idx] = depth_prior.to(dtype=self.buffer.depths.dtype)
        self.buffer.depths_sens_normed[kf_idx] = normed_depth
        self.buffer.depths_sens_scale[kf_idx] = scale
        self.buffer.non_sky_masks[kf_idx] = non_sky_mask
        self.buffer.bases[kf_idx] = bases

        if kf_idx == 0:
            self.buffer.intrinsics[0] = intrinsics[0]
            
        self.buffer.n_frames += 1
        
    def _precompute_features(self, frame_data: VideoFrame):
        assert frame_data.intrinsics is not None
        images = rearrange(frame_data.rgb[None], "n h w c -> n c h w")
        intrinsics = frame_data.intrinsics[None]
        return images.to(self.device), intrinsics.to(self.device)

    @torch.no_grad()
    def run(
        self,
        video_stream: VideoStream,
    ) -> SLAMOutput:
        intrinsics_processor = FixedIntrinsicsStreamProcessor(self.config.intrinsics)
        resizer = StandardResizeStreamProcessor()
        video_stream = ProcessedVideoStream(video_stream, [intrinsics_processor, resizer])

        frame_size = video_stream.frame_size()
        total_n_frames = len(video_stream)
        if total_n_frames <= 0:
            raise ValueError("SLAMSystem.run requires at least one frame")

        self.config.update(
            {
                "height": frame_size[0],
                "width": frame_size[1],
                "n_views": 1,
                "has_init_pose": FrameAttribute.POSE in video_stream.attributes(),
            }
        )

        self._build_components()

        if self.visualize:
            rr.init("ViPE Visualization", spawn=True, recording_id=uuid.uuid4())
            rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

        frame_data: VideoFrame
        for frame_idx, frame_data in pbar(
            enumerate(video_stream), desc="SLAM Initialization", total=total_n_frames
        ):
            images, intrinsics = self._precompute_features(frame_data)

            should_add_keyframe = self.motion_filter.check(
                images,
                intrinsics,
                force=frame_idx == total_n_frames - 1,
            )
            if should_add_keyframe:
                self._add_keyframe(
                    frame_idx,
                    images,
                    intrinsics,
                    phase=1,
                    features=(
                        self.motion_filter.f_fmap,
                        self.motion_filter.f_mono,
                        self.motion_filter.f_non_sky_mask,
                    ),
                    bases=self.motion_filter.f_bases,
                )
                self.initializer.run()

        self.initializer.finalize()
        initializer_snapshot_path = str(self.config.initializer_snapshot_path)
        if initializer_snapshot_path:
            save_initializer_snapshot(
                Path(initializer_snapshot_path),
                self.buffer,
                self.initializer.edges,
                metadata={
                    "ckpt_path": str(self.config.ckpt_path),
                    "pgo_mode": str(self.config.pgo_mode),
                    "pgo_backend": str(self.config.pgo_backend),
                },
            )
        edges = self.initializer.edges
        if self.config.enable_frontend:
            self.frontend.run(self.initializer.edges)
            edges = self.frontend.graph.edges

        original_intrinsics = resizer.recover_intrinsics(self.buffer.intrinsics[0])
        return SLAMOutput(
            trajectory=SE3(self.buffer.poses[: self.buffer.n_frames]),
            intrinsics=original_intrinsics,
            log_scales=torch.log(self.buffer.depths_sens_scale[: self.buffer.n_frames, 0].clamp_min(1e-6)),
            depths=self.buffer.depths[: self.buffer.n_frames],
            depth_masks=self.buffer.non_sky_masks[: self.buffer.n_frames],
            finalized_edges={
                "ii": edges.ii,
                "jj": edges.jj,
                "relative_pose": edges.relative_pose,
                "relative_scale": edges.relative_scale,
                "confidence": edges.confidence,
            },
            pgo_info=dict(edges.pgo_info),
            pgo_replay=dict(edges.pgo_replay),
            slam_map=None,
            timestamps=self.buffer.tstamp[: self.buffer.n_frames].cpu().numpy(),
        )
