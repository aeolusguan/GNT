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
from vipe.streams.base import FrameAttribute, ProcessedVideoStream, StreamProcessor, VideoFrame, VideoStream
from vipe.utils.logging import pbar
from GeoNT.models import GeoNTWrapper

from .components.buffer import GraphBuffer
from .components.motion_filter import MotionFilter
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
        new_intrinsics = after_intrinsics.clone()
        new_intrinsics[2] += self.scx
        new_intrinsics[3] += self.scy
        new_intrinsics[0:4:2] *= self.fac_x
        new_intrinsics[1:4:2] *= self.fac_y
        return new_intrinsics
    

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
            n_views=self.config.n_views,
            buffer_size=self.config.buffer,
            cross_view_idx=self.config.get("cross_view_idx", None),
            device=self.device
        )
        self.motion_filter = MotionFilter(
            self.gnt,
            thresh=self.config.filter_thresh,
            device=self.device,
        )

    def _add_keyframe(
        self,
        frame_idx: int,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        frame_data_list: list[VideoFrame],
        phase: int,
    ):
        assert phase in [1, 2]
        kf_idx = self.buffer.n_frames
        self.buffer.tstamp[kf_idx] = frame_idx
        gmap, mono_depth, non_sky_mask = self.gnt.encode_features(images, intrinsics)
        self.buffer.fmaps[kf_idx] = gmap
        normed_depth, scale = self.gnt.normalize_depth(mono_depth, non_sky_mask)
        bases = self.gnt.encode_bases(mono_depth, non_sky_mask)
        self.buffer.depths_sens_normed[kf_idx] = normed_depth
        self.buffer.depths_sens_scale[kf_idx] = scale
        self.buffer.non_sky_masks[kf_idx] = non_sky_mask
        self.buffer.bases[kf_idx] = bases

        for view_idx, frame_data in enumerate(frame_data_list):
            if kf_idx == 0:
                self.buffer.intrinsics[view_idx] = frame_data.intrinsics
            
        self.buffer.n_frames += 1
        
    def _precompute_features(self, frame_data_list: list[VideoFrame]):
        images_list = []
        intrinsics_list = []
        for frame_data in frame_data_list:
            images_list.append(frame_data.rgb)
            intrinsics_list.append(frame_data.intrinsics)
        images = rearrange(torch.stack(images_list), "n h w c -> n c h w")
        intrinsics = torch.stack(intrinsics_list)
        return images, intrinsics

    @torch.no_grad()
    def run(
        self,
        video_streams: list[VideoStream],
        rig: SE3 | None = None,
    ) -> SLAMOutput:
        assert len(video_streams) > 0
        resizers = [StandardResizeStreamProcessor() for _ in video_streams]
        video_streams = [
            ProcessedVideoStream(video_stream, [resizer]) for video_stream, resizer in zip(video_streams, resizers)
        ]

        frame_size = video_streams[0].frame_size()
        total_n_frames = len(video_streams[0])
        for vs in video_streams:
            assert vs.frame_size() == frame_size
            assert len(vs) == total_n_frames

        if rig is None:
            assert len(video_streams) == 1, "Need rig for multiple views"
            rig = SE3.Identity(1)
        self.rig = rig

        self.config.update(
            {
                "height": frame_size[0],
                "width": frame_size[1],
                "n_views": len(video_streams),
                "has_init_pose": FrameAttribute.POSE in video_streams[0].attributes(),
            }
        )

        self._build_components()

        if self.visualize:
            rr.init("ViPE Visualization", spawn=True, recording_id=uuid.uuid4())
            rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

        # Run frontend to get attributes initialization. This will also populate attribute buffers.
        frame_data_list: list[VideoFrame]
        frame_idx: int = 0
        for frame_idx, frame_data_list in pbar(
            enumerate(zip(*video_streams)), desc="SLAM Pass (1/2)", total=total_n_frames
        ):
            images, intrinsics = self._precompute_features(frame_data_list)

            if self.motion_filter.check(images, intrinsics) or frame_idx == total_n_frames - 1:
                is_keyframe = True
                self._add_keyframe(frame_idx, images, intrinsics, frame_data_list, phase=1)
            else:
                is_keyframe = False

            self.frontend.run()
            

        