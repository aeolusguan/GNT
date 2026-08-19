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
from bisect import bisect_left
from collections.abc import Callable

import numpy as np
import rerun as rr
import torch
from einops import rearrange
from omegaconf import DictConfig, OmegaConf

from lietorch import SE3
from gent.runtime.streams.base import FrameAttribute, ProcessedVideoStream, StreamProcessor, VideoFrame, VideoStream
from gent.runtime.utils.logging import pbar
from .components.buffer import GraphBuffer, KeyframeCandidate
from .components.backend import SLAMBackend
from .components.factor_graph import FactorGraph
from .components.frontend import NonKeyframePoseRecord, SLAMFrontend
from .components.measurements import GeNTMeasurements
from .components.pose_utils import multiply_scaled_se3
from .interface import SLAMOutput
from .pgo import optimize_fixed_frame_pose_scale


FrameDepth = tuple[torch.Tensor, torch.Tensor]


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
        intrinsics = after_intrinsics.clone()
        intrinsics[..., 0] *= self.fac_x
        intrinsics[..., 1] *= self.fac_y
        intrinsics[..., 2] = (intrinsics[..., 2] + self.scx) * self.fac_x
        intrinsics[..., 3] = (intrinsics[..., 3] + self.scy) * self.fac_y
        return intrinsics


class FrameIntrinsicsStreamProcessor(StreamProcessor):
    """Assign one pinhole calibration row to each selected video frame."""

    def __init__(self, intrinsics, n_frames: int) -> None:
        super().__init__()
        intrinsics = torch.as_tensor(intrinsics, dtype=torch.float)
        if intrinsics.shape == (4,):
            intrinsics = intrinsics[None].expand(n_frames, -1).clone()
        if intrinsics.shape != (n_frames, 4):
            raise ValueError(
                f"intrinsics must have shape ({n_frames}, 4), got {tuple(intrinsics.shape)}"
            )
        self.intrinsics = intrinsics

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes.union({FrameAttribute.INTRINSICS})

    def __call__(self, frame_idx: int, frame_data: VideoFrame) -> VideoFrame:
        frame_data.intrinsics = self.intrinsics[frame_idx].to(device=frame_data.rgb.device)
        return frame_data


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
        self.measurements = GeNTMeasurements.build(
            self.config.ckpt_path,
            device=self.device,
            use_fp16=self.config.use_fp16,
        )
        self.buffer = GraphBuffer(
            height=self.config.height,
            width=self.config.width,
            buffer_size=self.config.buffer,
            depth_token_dim=self.measurements.depth_token_dim,
            device=self.device,
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
            self.frontend.tracking_groups,
            self.config,
            device=self.device,
        )

    def _frame_to_model_inputs(self, frame_data: VideoFrame):
        assert frame_data.intrinsics is not None
        images = rearrange(frame_data.rgb[None], "n h w c -> n c h w")
        intrinsics = frame_data.intrinsics[None]
        return images.to(self.device), intrinsics.to(self.device)

    def _make_output(self, resizer: StandardResizeStreamProcessor, edges) -> SLAMOutput:
        n_frames = self.buffer.n_frames
        original_intrinsics = resizer.recover_intrinsics(self.buffer.intrinsics[:n_frames, 0])
        return SLAMOutput(
            trajectory=SE3(self.buffer.poses[:n_frames]),
            intrinsics=original_intrinsics,
            frame_intrinsics=self.frame_intrinsics.clone(),
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
            timestamps=self.buffer.tstamp[:n_frames].numpy(),
            frame_trajectory=self.frontend.make_frame_trajectory(),
            frame_timestamps=self.frontend.frame_timestamps(),
        )

    def _complete_benchmark_video_depths(
        self,
        nonkeyframe_candidates: list[KeyframeCandidate],
        total_n_frames: int,
    ) -> list[FrameDepth]:
        frame_depths: list[FrameDepth | None] = [None] * total_n_frames
        n_keyframes = self.buffer.n_frames
        keyframe_timestamps = self.buffer.tstamp[:n_keyframes].tolist()
        nonkeyframe_pose_records = self.frontend.nonkeyframe_pose_records
        assert len(nonkeyframe_candidates) == len(nonkeyframe_pose_records)

        def temporal_keyframe_neighbors(frame_timestamp: int) -> torch.Tensor:
            insertion = bisect_left(keyframe_timestamps, frame_timestamp)
            radius = self.backend.local_mapping_radius
            start = max(0, insertion - radius)
            end = min(len(keyframe_timestamps), insertion + radius)
            return torch.arange(start, end, device=self.device, dtype=torch.long)

        def initial_nonkeyframe_state(
            record: NonKeyframePoseRecord,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            reference = record.reference_keyframe
            reference_scale = self.buffer.scale[reference, 0].float()
            source_scale = reference_scale * torch.exp(-record.relative_log_scale.float())
            source_pose = multiply_scaled_se3(
                record.relative_pose[None],
                self.buffer.poses[reference].float()[None],
                source_scale.view(1),
            )[0]
            return source_pose, torch.log(source_scale.clamp_min(1e-6))

        for keyframe, timestamp in enumerate(keyframe_timestamps):
            metric_depth = self.buffer.depths[keyframe, 0].float() * self.buffer.scale[keyframe, 0].float()
            valid_mask = self.buffer.non_sky_masks[keyframe, 0]
            frame_depths[timestamp] = (metric_depth.cpu(), valid_mask.bool().cpu())

        for candidate, pose_record in zip(
            nonkeyframe_candidates,
            nonkeyframe_pose_records,
            strict=True,
        ):
            assert candidate.frame_idx == pose_record.timestamp
            neighbors = temporal_keyframe_neighbors(candidate.frame_idx)
            pose_measurement, tracking_group = self.frontend.track_candidate(
                candidate.frame_idx,
                candidate,
                neighbors,
                decode_depth=False,
            )
            ii = torch.full_like(neighbors, candidate.frame_idx)
            target_intrinsics = self.buffer.intrinsics[neighbors, 0]
            depth_measurement = self.measurements.predict_multiview(
                ii,
                neighbors,
                flow=tracking_group.flow,
                info=tracking_group.info,
                source_depth=candidate.depth_sens_normed[0].float(),
                source_mask=candidate.mask[0],
                source_depth_token=candidate.depth_token[0].float(),
                target_depth_token=self.buffer.depth_tokens[neighbors, 0].float(),
                source_intrinsics=candidate.intrinsics[0],
                target_intrinsics=target_intrinsics,
                decode_depth=True,
                relative_pose_prior=pose_measurement.relative_pose,
            )
            source_pose, source_log_scale = initial_nonkeyframe_state(pose_record)
            target_poses = self.buffer.poses[neighbors].float()
            target_log_scales = torch.log(self.buffer.scale[neighbors, 0].float().clamp_min(1e-6))
            result = optimize_fixed_frame_pose_scale(
                source_pose,
                source_log_scale,
                target_poses,
                target_log_scales,
                depth_measurement.relative_pose,
                depth_measurement.relative_log_scale,
                depth_measurement.confidence,
                n_iters=self.backend.pgo_iters,
                damping=self.backend.pgo_damping,
                lm_max_attempts=self.backend.pgo_lm_max_attempts,
                huber_delta=self.backend.pgo_huber_delta,
            )
            if not result.info["success"]:
                raise RuntimeError(f"fixed-frame PGO failed for video frame {candidate.frame_idx}")

            optimized_log_scale = result.log_scales[0]
            refined_depth = depth_measurement.refined_depth
            metric_depth = refined_depth.float() * torch.exp(optimized_log_scale)
            frame_depths[candidate.frame_idx] = (
                metric_depth.cpu(),
                candidate.mask[0].bool().cpu(),
            )

        missing = [frame for frame, result in enumerate(frame_depths) if result is None]
        if missing:
            raise RuntimeError(f"video-depth completion missed frames: {missing}")
        return [result for result in frame_depths if result is not None]

    def _run_dense_slam(
        self,
        video_stream: VideoStream,
        resizer: StandardResizeStreamProcessor,
        total_n_frames: int,
        *,
        on_nonkeyframe: Callable[[KeyframeCandidate], None] | None = None,
    ) -> SLAMOutput:
        frame_data: VideoFrame
        for frame_idx, frame_data in pbar(
            enumerate(video_stream), desc="GeNT dense SLAM", total=total_n_frames
        ):
            images, intrinsics = self._frame_to_model_inputs(frame_data)
            keyframe_candidate = self.frontend.make_candidate(
                frame_idx,
                images,
                intrinsics,
            )
            accepted = self.frontend.track(
                keyframe_candidate,
                force=frame_idx == total_n_frames - 1,
            )
            if not accepted:
                if on_nonkeyframe is not None:
                    on_nonkeyframe(keyframe_candidate)
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

    def _initialize_dense_slam(
        self,
        video_stream: VideoStream,
    ) -> tuple[ProcessedVideoStream, StandardResizeStreamProcessor, int]:
        total_n_frames = len(video_stream)
        if total_n_frames <= 0:
            raise ValueError("SLAMSystem.run requires at least one frame")

        intrinsics_processor = FrameIntrinsicsStreamProcessor(
            self.config.intrinsics,
            total_n_frames,
        )
        self.frame_intrinsics = intrinsics_processor.intrinsics
        resizer = StandardResizeStreamProcessor()
        video_stream = ProcessedVideoStream(video_stream, [intrinsics_processor, resizer])

        frame_size = video_stream.frame_size()

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

        return video_stream, resizer, total_n_frames

    @torch.no_grad()
    def run(
        self,
        video_stream: VideoStream,
    ) -> SLAMOutput:
        video_stream, resizer, total_n_frames = self._initialize_dense_slam(video_stream)
        return self._run_dense_slam(
            video_stream,
            resizer,
            total_n_frames,
        )

    @torch.no_grad()
    def run_video_depth_benchmark(
        self,
        video_stream: VideoStream,
    ) -> tuple[SLAMOutput, list[FrameDepth]]:
        video_stream, resizer, total_n_frames = self._initialize_dense_slam(video_stream)
        nonkeyframe_candidates: list[KeyframeCandidate] = []
        output = self._run_dense_slam(
            video_stream,
            resizer,
            total_n_frames,
            on_nonkeyframe=nonkeyframe_candidates.append,
        )
        frame_depths = self._complete_benchmark_video_depths(
            nonkeyframe_candidates,
            total_n_frames,
        )
        return output, frame_depths
