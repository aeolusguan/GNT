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

import numpy as np
import rerun as rr
import torch
from einops import rearrange
from omegaconf import DictConfig, OmegaConf

from lietorch import SE3
from geont_runtime.streams.base import FrameAttribute, ProcessedVideoStream, StreamProcessor, VideoFrame, VideoStream
from geont_runtime.utils.logging import pbar
from geont.models import GeoNTWrapper

from .components.buffer import GraphBuffer, KeyframeCandidate
from .components.backend import SLAMBackend
from .components.factor_graph import FactorGraph
from .components.frontend import SLAMFrontend
from .components.measurements import GeoNTMeasurements
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
        self.measurements = GeoNTMeasurements(
            self.gnt,
            self.buffer,
            self.device,
            use_fp16=self.config.use_fp16,
        )
        self.graph = FactorGraph(self.buffer, self.device)
        self.frontend = SLAMFrontend(
            self.measurements,
            self.buffer,
            self.config,
            device=self.device,
        )
        self.backend = SLAMBackend(
            self.measurements,
            self.buffer,
            self.graph,
            self.config,
            device=self.device,
        )

    def _make_keyframe_candidate(
        self,
        frame_idx: int,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> KeyframeCandidate:
        gmap, mono_depth, non_sky_mask = self.measurements.moge_prior(images, intrinsics)
        normalization_mask = non_sky_mask & (mono_depth < 80)
        normed_depth, scale = self.gnt.normalize_depth(mono_depth, normalization_mask)
        bases = self.measurements.encode_bases(mono_depth, non_sky_mask)
        return KeyframeCandidate(
            frame_idx=int(frame_idx),
            fmap=gmap,
            depth=normed_depth,
            depth_sens_normed=normed_depth,
            scale=scale,
            mask=normalization_mask,
            bases=bases,
            intrinsics=intrinsics,
        )

    def _frame_to_model_inputs(self, frame_data: VideoFrame):
        assert frame_data.intrinsics is not None
        images = rearrange(frame_data.rgb[None], "n h w c -> n c h w")
        intrinsics = frame_data.intrinsics[None]
        return images.to(self.device), intrinsics.to(self.device)

    def _make_output(self, resizer: StandardResizeStreamProcessor, edges) -> SLAMOutput:
        n_frames = self.buffer.n_frames
        original_intrinsics = resizer.recover_intrinsics(self.buffer.intrinsics[0])
        return SLAMOutput(
            trajectory=SE3(self.buffer.poses[:n_frames]),
            intrinsics=original_intrinsics,
            log_scales=torch.log(self.buffer.scale[:n_frames, 0].clamp_min(1e-6)),
            moge_log_scales=torch.log(self.buffer.depths_sens_scale[:n_frames, 0].clamp_min(1e-6)),
            depths=self.buffer.depths[:n_frames],
            depth_masks=self.buffer.non_sky_masks[:n_frames],
            depth_status=self.buffer.depth_status[:n_frames].clone(),
            depth_dirty=self.buffer.depth_dirty[:n_frames].clone(),
            pose_edges={
                "ii": edges.ii,
                "jj": edges.jj,
                "relative_pose": edges.relative_pose,
                "relative_log_scale": edges.relative_log_scale,
                "confidence": edges.confidence,
            },
            pgo_info=dict(edges.pgo_info),
            slam_map=None,
            timestamps=self.buffer.tstamp[:n_frames].cpu().numpy(),
            frame_trajectory=self.frontend.make_frame_trajectory(),
            frame_timestamps=self.frontend.frame_timestamps(),
        )

    def _run_streaming_local_mapping(
        self,
        video_stream: VideoStream,
        resizer: StandardResizeStreamProcessor,
        total_n_frames: int,
    ) -> SLAMOutput:
        frame_data: VideoFrame
        for frame_idx, frame_data in pbar(
            enumerate(video_stream), desc="GeoNT local mapping", total=total_n_frames
        ):
            images, intrinsics = self._frame_to_model_inputs(frame_data)
            keyframe_candidate = self._make_keyframe_candidate(frame_idx, images, intrinsics)
            accepted = self.frontend.track(
                keyframe_candidate,
                force=frame_idx == total_n_frames - 1,
            )
            if not accepted:
                continue

            current_keyframe = self.buffer.n_frames - 1
            if current_keyframe == 0:
                continue

            self.backend.update_local_graph(current_keyframe)
            finalized_end = current_keyframe - self.backend.local_mapping_radius + 1
            if finalized_end > 1:
                self.backend.optimize_local_pgo(finalized_end)

        self.backend.finalize_pending_keyframes()
        self.backend.optimize_full_graph()

        edges = self.graph.edges
        return self._make_output(resizer, edges)

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
            }
        )

        self._build_components()

        if self.visualize:
            rr.init("ViPE Visualization", spawn=True, recording_id=uuid.uuid4())
            rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

        return self._run_streaming_local_mapping(
            video_stream,
            resizer,
            total_n_frames,
        )
