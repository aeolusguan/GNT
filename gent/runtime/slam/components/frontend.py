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

from dataclasses import dataclass

import numpy as np
import torch
from lietorch import SE3

from .buffer import DEPTH_STATUS_REFINED, GraphBuffer, KeyframeCandidate
from .measurements import GeNTMeasurements, PoseMeasurement
from .pose_utils import multiply_scaled_se3


@dataclass
class NonKeyframePoseRecord:
    timestamp: int
    reference_keyframe: int
    relative_pose: torch.Tensor
    relative_log_scale: torch.Tensor


@dataclass
class TrackingGroup:
    neighbors: torch.Tensor  # [E], newest neighbor first
    flow: torch.Tensor  # [E,2,H,W], newest neighbor first
    info: torch.Tensor  # [E,C,H,W], newest neighbor first
    relative_pose: torch.Tensor  # [E,7], source normalized-depth gauge
    relative_log_scale: torch.Tensor  # [E], log(target scale) - log(source scale)


class SLAMFrontend:
    """Streaming keyframe tracking and admission."""

    def __init__(
        self,
        measurements: GeNTMeasurements,
        buffer: GraphBuffer,
        args,
        device: torch.device,
    ):
        self.measurements = measurements
        self.buffer = buffer
        self.device = device
        self.keyframe_motion_thresh = float(args.keyframe_motion_thresh)
        self.local_mapping_radius = int(args.local_mapping_radius)
        assert self.local_mapping_radius > 0
        self.last_keyframe: int | None = None
        self.nonkeyframe_pose_records: list[NonKeyframePoseRecord] = []
        self.tracking_groups: dict[int, TrackingGroup] = {}

    def make_candidate(
        self,
        frame_idx: int,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> KeyframeCandidate:
        fmap, depth, depth_token, scale, mask, bases = self.measurements.encode_frame(
            images,
            intrinsics,
        )
        return KeyframeCandidate(
            frame_idx=int(frame_idx),
            fmap=fmap,
            depth=depth,
            depth_sens_normed=depth,
            depth_token=depth_token,
            scale=scale,
            mask=mask,
            bases=bases,
            intrinsics=intrinsics,
        )

    def _commit_keyframe(self, keyframe_candidate: KeyframeCandidate) -> None:
        keyframe = self.buffer.n_frames
        if keyframe >= self.buffer.poses.shape[0]:
            raise RuntimeError(
                f"SLAM buffer is full at {keyframe} keyframes; increase slam.buffer in the config."
            )

        self.buffer.tstamp[keyframe] = int(keyframe_candidate.frame_idx)
        self.buffer.fmaps[keyframe] = keyframe_candidate.fmap
        self.buffer.depths[keyframe] = keyframe_candidate.depth.float().to(dtype=self.buffer.depths.dtype)
        self.buffer.depths_sens_normed[keyframe] = keyframe_candidate.depth_sens_normed
        self.buffer.depth_tokens[keyframe] = keyframe_candidate.depth_token
        self.buffer.depths_sens_scale[keyframe] = keyframe_candidate.scale
        self.buffer.scale[keyframe] = keyframe_candidate.scale
        self.buffer.non_sky_masks[keyframe] = keyframe_candidate.mask
        self.buffer.bases[keyframe] = keyframe_candidate.bases
        self.buffer.depth_status[keyframe] = int(keyframe_candidate.depth_status)
        self.buffer.depth_dirty[keyframe] = True

        self.buffer.intrinsics[keyframe] = keyframe_candidate.intrinsics

        self.buffer.n_frames += 1
        self.last_keyframe = keyframe

    def track_candidate(
        self,
        source: int,
        keyframe_candidate: KeyframeCandidate,
        neighbors: torch.Tensor,
        *,
        decode_depth: bool,
    ) -> tuple[PoseMeasurement, TrackingGroup]:
        flow, info = self.measurements.predict_flow(
            keyframe_candidate.fmap,
            self.buffer.fmaps[neighbors, 0],
            self.buffer.bases[neighbors, 0],
        )
        ii = torch.full_like(neighbors, int(source))
        target_intrinsics = self.buffer.intrinsics[neighbors, 0]
        result = self.measurements.predict_multiview(
            ii,
            neighbors,
            flow=flow,
            info=info,
            source_depth=keyframe_candidate.depth_sens_normed[0].float(),
            source_mask=keyframe_candidate.mask[0],
            source_depth_token=keyframe_candidate.depth_token[0].float(),
            target_depth_token=self.buffer.depth_tokens[neighbors, 0].float(),
            source_intrinsics=keyframe_candidate.intrinsics[0],
            target_intrinsics=target_intrinsics,
            decode_depth=decode_depth,
        )
        return result, TrackingGroup(
            neighbors=neighbors,
            flow=flow,
            info=info,
            relative_pose=result.relative_pose,
            relative_log_scale=result.relative_log_scale,
        )

    def track(self, keyframe_candidate: KeyframeCandidate, *, force: bool = False) -> bool:
        if self.last_keyframe is None:
            self._commit_keyframe(keyframe_candidate)
            return True

        last_keyframe = self.last_keyframe
        current_keyframe = self.buffer.n_frames

        start = max(0, current_keyframe - self.local_mapping_radius)
        neighbors = torch.arange(current_keyframe - 1, start - 1, -1, device=self.device, dtype=torch.long)
        tracking_result, tracking_group = self.track_candidate(
            current_keyframe,
            keyframe_candidate,
            neighbors,
            decode_depth=True,
        )
        keyframe_candidate.depth = tracking_result.refined_depth.float()[None].to(dtype=keyframe_candidate.depth.dtype)
        keyframe_candidate.depth_status = DEPTH_STATUS_REFINED

        # Neighbors are newest-first, so edge 0 is current -> last keyframe.
        # Admission branches on this scalar immediately; `.item()` is the required GPU sync.
        flow_magnitude = tracking_group.flow[0].norm(dim=0).mean().item()

        scale = keyframe_candidate.scale
        relative_pose = tracking_result.relative_pose[0].float()
        relative_log_scale = tracking_result.relative_log_scale[0].float()

        if not force and flow_magnitude <= self.keyframe_motion_thresh:
            self.nonkeyframe_pose_records.append(
                NonKeyframePoseRecord(
                    timestamp=int(keyframe_candidate.frame_idx),
                    reference_keyframe=int(last_keyframe),
                    relative_pose=relative_pose.float().clone(),
                    relative_log_scale=relative_log_scale.clone(),
                )
            )
            return False

        self._commit_keyframe(keyframe_candidate)
        self.tracking_groups[current_keyframe] = tracking_group
        current_pose = multiply_scaled_se3(
            relative_pose[None],
            self.buffer.poses[last_keyframe].float()[None],
            scale.view(1),
        )[0]
        self.buffer.poses[current_keyframe] = current_pose.to(dtype=self.buffer.poses.dtype)

        return True

    def frame_timestamps(self) -> np.ndarray:
        n_keyframes = self.buffer.n_frames
        timestamps = self.buffer.tstamp[:n_keyframes].tolist()
        timestamps.extend(record.timestamp for record in self.nonkeyframe_pose_records)
        return np.asarray(sorted(timestamps), dtype=np.int64)

    def make_frame_trajectory(self) -> SE3:
        frame_poses = []
        keyframe_timestamps = self.buffer.tstamp[: self.buffer.n_frames].tolist()
        for keyframe, timestamp in enumerate(keyframe_timestamps):
            frame_poses.append((timestamp, self.buffer.poses[keyframe].float()))
        for record in self.nonkeyframe_pose_records:
            reference_scale = self.buffer.scale[record.reference_keyframe, 0].float()
            source_scale = reference_scale * torch.exp(-record.relative_log_scale.float())
            pose = multiply_scaled_se3(
                record.relative_pose[None],
                self.buffer.poses[record.reference_keyframe].float()[None],
                source_scale.view(1),
            )[0]
            frame_poses.append((record.timestamp, pose))

        if not frame_poses:
            return SE3(torch.empty(0, 7, device=self.device, dtype=torch.float))
        poses = [pose.to(device=self.device, dtype=torch.float) for _, pose in sorted(frame_poses, key=lambda item: item[0])]
        return SE3(torch.stack(poses, dim=0))
