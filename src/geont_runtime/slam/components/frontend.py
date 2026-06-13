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

import torch
from lietorch import SE3

from .buffer import DEPTH_STATUS_REFINED, GraphBuffer, KeyframeCandidate
from .measurements import GeoNTMeasurements


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
        self.tracking_info = {
            "accepted_keyframes": 0,
            "keyframe_motion_rejections": 0,
            "keyframe_motion_thresh": self.keyframe_motion_thresh,
            "tracking_multiview_neighbors": self.tracking_multiview_neighbors,
            "tracking_multiview_runs": 0,
        }

    def _record_keyframe_accepted(self) -> None:
        self.tracking_info["accepted_keyframes"] += 1

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

    def _seed_source_pose_from_measurement(self, result: dict, source: int, target: int, scale: float) -> None:
        edge = (result["ii"] == int(source)) & (result["jj"] == int(target))
        assert edge.any()
        edge_idx = int(torch.nonzero(edge, as_tuple=False)[0].item())
        relative_pose = result["relative_pose"][edge_idx].float().clone()
        relative_pose[:3] = relative_pose[:3] * scale
        source_pose = SE3(relative_pose.view(1, 7)).inv() * SE3(self.buffer.poses[target].float().view(1, 7))
        self.buffer.poses[source] = source_pose.data[0].to(dtype=self.buffer.poses.dtype)

    def track(self, keyframe_candidate: KeyframeCandidate, *, force: bool = False) -> dict:
        if self.last_keyframe is None:
            keyframe = self._commit_keyframe(keyframe_candidate)
            return {
                "accepted": True,
                "reason": "first_frame",
                "keyframe": keyframe,
            }

        motion_score = self.measurements.coarse_flow_motion_score(self.last_keyframe, keyframe_candidate)
        if not force and motion_score <= self.keyframe_motion_thresh:
            self.tracking_info["keyframe_motion_rejections"] += 1
            return {
                "accepted": False,
                "reason": "low_motion",
                "coarse_flow_motion_score": motion_score,
            }

        last_keyframe = self.last_keyframe
        target_keyframe = self.buffer.n_frames
        tracking_result = self._run_tracking_multiview(target_keyframe, keyframe_candidate)
        current_keyframe = self._commit_keyframe(keyframe_candidate)
        last_result = self._measurement_subset(tracking_result, tracking_result["jj"] == int(last_keyframe))
        
        # Seed the source keyframe pose from the tracking measurement
        mask = keyframe_candidate.mask[0]
        refined_depth = tracking_result["refined_depth"].float()
        if mask.any():
            source_mean = keyframe_candidate.depth_sens_normed[0].float()[mask].mean().clamp_min(1e-6)
            refined_mean = refined_depth[mask].mean().clamp_min(1e-6)
            scale = keyframe_candidate.scale * (source_mean / refined_mean)
        else:
            scale = keyframe_candidate.scale
        self._seed_source_pose_from_measurement(last_result, current_keyframe, last_keyframe, scale)

        return {
            "accepted": True,
            "reason": "keyframe",
            "keyframe": current_keyframe,
            "coarse_flow_motion_score": motion_score,
        }
