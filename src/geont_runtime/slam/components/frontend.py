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


@dataclass
class FramePoseRecord:
    timestamp: int
    is_keyframe: bool
    keyframe: int
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
        self.keyframe_motion_thresh = float(args.get("keyframe_motion_thresh", 2.5))
        self.tracking_multiview_neighbors = int(args.get("tracking_multiview_neighbors", 2))
        assert self.tracking_multiview_neighbors > 0
        self.last_keyframe: int | None = None
        self.frame_pose_records: list[FramePoseRecord] = []
        self.tracking_info = {
            "tracked_frames": 0,
            "accepted_keyframes": 0,
            "non_keyframe_frames": 0,
            "keyframe_motion_rejections": 0,
            "keyframe_motion_thresh": self.keyframe_motion_thresh,
            "tracking_multiview_neighbors": self.tracking_multiview_neighbors,
            "tracking_multiview_runs": 0,
        }

    def _record_keyframe_accepted(self) -> None:
        self.tracking_info["accepted_keyframes"] += 1

    def _commit_keyframe(self, keyframe_candidate: KeyframeCandidate, pgo_base_scale: torch.Tensor) -> int:
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
        self.buffer.pgo_base_scale[keyframe] = pgo_base_scale
        self.buffer.non_sky_masks[keyframe] = keyframe_candidate.mask
        self.buffer.bases[keyframe] = keyframe_candidate.bases
        self.buffer.depth_status[keyframe] = int(keyframe_candidate.depth_status)
        self.buffer.depth_dirty[keyframe] = True

        if keyframe == 0:
            self.buffer.intrinsics[0] = keyframe_candidate.intrinsics[0]

        self.buffer.n_frames += 1
        self._record_keyframe_accepted()
        self.last_keyframe = keyframe
        return keyframe

    def _update_candidate_depth(self, keyframe_candidate: KeyframeCandidate, refined_depth: torch.Tensor) -> None:
        refined_depth = refined_depth.float()
        keyframe_candidate.depth = refined_depth[None].to(dtype=keyframe_candidate.depth.dtype)
        keyframe_candidate.depth_status = DEPTH_STATUS_REFINED

    def _run_tracking_multiview(self, target_keyframe: int, keyframe_candidate: KeyframeCandidate) -> dict:
        assert target_keyframe > 0
        start = max(0, int(target_keyframe) - self.tracking_multiview_neighbors)
        neighbors = torch.arange(int(target_keyframe) - 1, start - 1, -1, device=self.device, dtype=torch.long)
        motion_tokens = self.measurements.patch_embed_candidate(keyframe_candidate, neighbors)
        ii = torch.full_like(neighbors, int(target_keyframe))
        output = self.measurements.run_multiview_pose_depth_from_motion_tokens(
            motion_tokens,
            keyframe_candidate.depth[0].float(),
            keyframe_candidate.mask[0],
        )
        result = self.measurements.make_pose_measurement(ii, neighbors, output["pose"], output["pose_confidence"])
        result.update(
            {
                "refined_depth": output["refined_depth"],
                "depth_confidence": output["depth_confidence"],
            }
        )
        self._update_candidate_depth(keyframe_candidate, result["refined_depth"])
        self.tracking_info["tracking_multiview_runs"] += 1
        return result

    def _measurement_subset(self, result: dict, mask: torch.Tensor) -> dict:
        n_edges = int(result["ii"].numel())
        subset = {}
        for key, value in result.items():
            if isinstance(value, torch.Tensor) and value.shape[:1] == (n_edges,):
                subset[key] = value[mask]
            else:
                subset[key] = value
        return subset

    def _source_pose_from_relative(self, relative_pose: torch.Tensor, target_pose: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        relative_pose = relative_pose.float().clone()
        relative_pose[:3] = relative_pose[:3] * scale
        source_pose = SE3(relative_pose.view(1, 7)).inv() * SE3(target_pose.float().view(1, 7))
        return source_pose.data[0]

    def _relative_pose_to_target(self, result: dict, source: int, target: int) -> torch.Tensor:
        edge = (result["ii"] == int(source)) & (result["jj"] == int(target))
        assert edge.any()
        edge_idx = int(torch.nonzero(edge, as_tuple=False)[0].item())
        return result["relative_pose"][edge_idx].float()

    def _append_keyframe_record(self, timestamp: int, keyframe: int) -> None:
        identity = torch.as_tensor([0, 0, 0, 0, 0, 0, 1], device=self.device, dtype=torch.float)
        scale = self.buffer.depths_sens_scale[keyframe, 0].float().clone()
        self.frame_pose_records.append(
            FramePoseRecord(
                timestamp=int(timestamp),
                is_keyframe=True,
                keyframe=int(keyframe),
                reference_keyframe=int(keyframe),
                relative_pose=identity,
                source_scale=scale,
            )
        )

    def _append_non_keyframe_record(
        self,
        timestamp: int,
        reference_keyframe: int,
        relative_pose: torch.Tensor,
        source_scale: torch.Tensor,
    ) -> None:
        self.frame_pose_records.append(
            FramePoseRecord(
                timestamp=int(timestamp),
                is_keyframe=False,
                keyframe=-1,
                reference_keyframe=int(reference_keyframe),
                relative_pose=relative_pose.float().clone(),
                source_scale=source_scale.float().clone(),
            )
        )

    def track(self, keyframe_candidate: KeyframeCandidate, *, force: bool = False) -> dict:
        self.tracking_info["tracked_frames"] += 1
        if self.last_keyframe is None:
            keyframe = self._commit_keyframe(keyframe_candidate, keyframe_candidate.scale)
            self._append_keyframe_record(
                keyframe_candidate.frame_idx,
                keyframe,
            )
            return {
                "accepted": True,
                "reason": "first_frame",
                "keyframe": keyframe,
                "tracked": True,
            }

        motion_score = self.measurements.coarse_flow_motion_score(self.last_keyframe, keyframe_candidate)
        last_keyframe = self.last_keyframe
        target_keyframe = self.buffer.n_frames
        tracking_result = self._run_tracking_multiview(target_keyframe, keyframe_candidate)
        last_result = self._measurement_subset(tracking_result, tracking_result["jj"] == int(last_keyframe))

        mask = keyframe_candidate.mask[0]
        if mask.any():
            source_mean = keyframe_candidate.depth_sens_normed[0].float()[mask].mean().clamp_min(1e-6)
            committed_mean = keyframe_candidate.depth[0].float()[mask].mean().clamp_min(1e-6)
            tracking_scale_ratio = source_mean / committed_mean
        else:
            tracking_scale_ratio = torch.ones((), device=self.device, dtype=torch.float)
        scale = keyframe_candidate.scale * tracking_scale_ratio
        relative_pose = self._relative_pose_to_target(last_result, target_keyframe, last_keyframe)

        if not force and motion_score <= self.keyframe_motion_thresh:
            self.tracking_info["keyframe_motion_rejections"] += 1
            self.tracking_info["non_keyframe_frames"] += 1
            self._append_non_keyframe_record(
                keyframe_candidate.frame_idx,
                last_keyframe,
                relative_pose,
                scale,
            )
            return {
                "accepted": False,
                "reason": "low_motion",
                "coarse_flow_motion_score": motion_score,
                "tracked": True,
            }

        current_keyframe = self._commit_keyframe(keyframe_candidate, scale)
        immediate_pose = self._source_pose_from_relative(relative_pose, self.buffer.poses[last_keyframe].float(), scale)
        self.buffer.poses[current_keyframe] = immediate_pose.to(dtype=self.buffer.poses.dtype)
        self._append_keyframe_record(keyframe_candidate.frame_idx, current_keyframe)

        return {
            "accepted": True,
            "reason": "keyframe",
            "keyframe": current_keyframe,
            "coarse_flow_motion_score": motion_score,
            "tracked": True,
        }

    def frame_timestamps(self) -> np.ndarray:
        return np.asarray([record.timestamp for record in self.frame_pose_records], dtype=np.int64)

    def make_frame_trajectory(self) -> SE3:
        poses = []
        for record in self.frame_pose_records:
            if record.is_keyframe:
                pose = self.buffer.poses[record.keyframe].float()
            else:
                pgo_base_scale = self.buffer.pgo_base_scale[record.reference_keyframe, 0].float().clamp_min(1e-6)
                optimized_ref_scale = self.buffer.depths_sens_scale[record.reference_keyframe, 0].float()
                source_scale = record.source_scale.float() * (optimized_ref_scale / pgo_base_scale)
                pose = self._source_pose_from_relative(
                    record.relative_pose,
                    self.buffer.poses[record.reference_keyframe].float(),
                    source_scale,
                )
            poses.append(pose.to(device=self.device, dtype=torch.float))

        if not poses:
            return SE3(torch.empty(0, 7, device=self.device, dtype=torch.float))
        return SE3(torch.stack(poses, dim=0))
