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
from .measurements import GeoNTMeasurements
from .pose_utils import multiply_scaled_se3


@dataclass
class NonKeyframePoseRecord:
    timestamp: int
    reference_keyframe: int
    relative_pose: torch.Tensor
    source_scale: torch.Tensor


class SLAMFrontend:
    """Streaming keyframe tracking and admission."""

    def __init__(
        self,
        measurements: GeoNTMeasurements,
        buffer: GraphBuffer,
        args,
        device: torch.device,
    ):
        self.measurements = measurements
        self.buffer = buffer
        self.device = device
        self.keyframe_motion_thresh = float(args.keyframe_motion_thresh)
        self.tracking_multiview_neighbors = int(args.tracking_multiview_neighbors)
        assert self.tracking_multiview_neighbors > 0
        self.last_keyframe: int | None = None
        self.nonkeyframe_pose_records: list[NonKeyframePoseRecord] = []

    def _commit_keyframe(self, keyframe_candidate: KeyframeCandidate) -> int:
        keyframe = self.buffer.n_frames
        if keyframe >= self.buffer.poses.shape[0]:
            raise RuntimeError(
                f"SLAM buffer is full at {keyframe} keyframes; increase slam.buffer in the config."
            )

        self.buffer.tstamp[keyframe] = int(keyframe_candidate.frame_idx)
        self.buffer.fmaps[keyframe] = keyframe_candidate.fmap
        self.buffer.depths[keyframe] = keyframe_candidate.depth.float().to(dtype=self.buffer.depths.dtype)
        self.buffer.depths_sens_normed[keyframe] = keyframe_candidate.depth_sens_normed
        self.buffer.depths_sens_scale[keyframe] = keyframe_candidate.scale
        self.buffer.scale[keyframe] = keyframe_candidate.scale
        self.buffer.non_sky_masks[keyframe] = keyframe_candidate.mask
        self.buffer.bases[keyframe] = keyframe_candidate.bases
        self.buffer.depth_status[keyframe] = int(keyframe_candidate.depth_status)
        self.buffer.depth_dirty[keyframe] = True

        if keyframe == 0:
            self.buffer.intrinsics[0] = keyframe_candidate.intrinsics[0]

        self.buffer.n_frames += 1
        self.last_keyframe = keyframe
        return keyframe

    def track(self, keyframe_candidate: KeyframeCandidate, *, force: bool = False) -> bool:
        if self.last_keyframe is None:
            self._commit_keyframe(keyframe_candidate)
            return True

        last_keyframe = self.last_keyframe
        target_keyframe = self.buffer.n_frames

        start = max(0, target_keyframe - self.tracking_multiview_neighbors)
        neighbors = torch.arange(target_keyframe - 1, start - 1, -1, device=self.device, dtype=torch.long)
        tracking_result, flow_magnitudes = self.measurements.predict_candidate_pose_depth(
            target_keyframe,
            keyframe_candidate,
            neighbors,
        )
        keyframe_candidate.depth = tracking_result.refined_depth.float()[None].to(dtype=keyframe_candidate.depth.dtype)
        keyframe_candidate.depth_status = DEPTH_STATUS_REFINED

        last_edge = tracking_result.jj == int(last_keyframe)
        assert last_edge.any()
        edge_idx = int(torch.nonzero(last_edge, as_tuple=False)[0].item())
        flow_magnitude = float(flow_magnitudes[edge_idx].item())

        scale = keyframe_candidate.scale
        relative_pose = tracking_result.relative_pose[edge_idx].float()

        if not force and flow_magnitude <= self.keyframe_motion_thresh:
            self.nonkeyframe_pose_records.append(
                NonKeyframePoseRecord(
                    timestamp=int(keyframe_candidate.frame_idx),
                    reference_keyframe=int(last_keyframe),
                    relative_pose=relative_pose.float().clone(),
                    source_scale=scale.float().clone(),
                )
            )
            return False

        current_keyframe = self._commit_keyframe(keyframe_candidate)
        current_pose = multiply_scaled_se3(
            relative_pose[None],
            self.buffer.poses[last_keyframe].float()[None],
            scale.view(1),
        )[0]
        self.buffer.poses[current_keyframe] = current_pose.to(dtype=self.buffer.poses.dtype)

        return True

    def frame_timestamps(self) -> np.ndarray:
        n_keyframes = self.buffer.n_frames
        timestamps = self.buffer.tstamp[:n_keyframes].cpu().numpy().astype(np.int64).tolist()
        timestamps.extend(record.timestamp for record in self.nonkeyframe_pose_records)
        return np.asarray(sorted(timestamps), dtype=np.int64)

    def make_frame_trajectory(self) -> SE3:
        frame_poses = []
        for keyframe in range(self.buffer.n_frames):
            frame_poses.append((int(self.buffer.tstamp[keyframe].item()), self.buffer.poses[keyframe].float()))
        for record in self.nonkeyframe_pose_records:
            raw_ref_scale = self.buffer.depths_sens_scale[record.reference_keyframe, 0].float().clamp_min(1e-6)
            current_ref_scale = self.buffer.scale[record.reference_keyframe, 0].float()
            source_scale = record.source_scale.float() * (current_ref_scale / raw_ref_scale)
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
