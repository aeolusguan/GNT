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
from .edge_filters import filter_local_edge_measurement
from .factor_graph import FactorGraph
from .measurements import GeoNTMeasurements


class SLAMFrontend:
    """Streaming keyframe tracking and admission."""

    def __init__(
        self,
        measurements: GeoNTMeasurements,
        buffer: GraphBuffer,
        graph: FactorGraph,
        args,
        device: torch.device,
    ):
        self.measurements = measurements
        self.buffer = buffer
        self.graph = graph
        self.device = device
        self.keyframe_motion_thresh = float(args.get("keyframe_motion_thresh", 2.5))
        self.depth_bootstrap_neighbors = int(args.get("depth_bootstrap_neighbors", 2))
        self.local_edge_outlier_trans_conf_thresh = float(args.local_edge_outlier_trans_conf_thresh)
        self.local_edge_max_rotation_deg = float(args.get("local_edge_max_rotation_deg", 30.0))
        self.local_edge_max_translation = float(args.get("local_edge_max_translation", 5.0))
        self.last_keyframe: int | None = None
        self.tracking_info = {
            "accepted_keyframes": 0,
            "keyframe_motion_rejections": 0,
            "keyframe_motion_thresh": self.keyframe_motion_thresh,
            "depth_bootstrap_neighbors": self.depth_bootstrap_neighbors,
            "depth_bootstrap_runs": 0,
            "depth_bootstrap_edges_added": 0,
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
        mask = keyframe_candidate.mask[0]
        if mask.any():
            source_mean = keyframe_candidate.depth_sens_normed[0].float()[mask].mean().clamp_min(1e-6)
            refined_mean = refined_depth[mask].mean().clamp_min(1e-6)
            keyframe_candidate.scale = keyframe_candidate.scale * (source_mean / refined_mean)
        keyframe_candidate.depth = refined_depth[None].to(dtype=keyframe_candidate.depth.dtype)
        keyframe_candidate.depth_status = DEPTH_STATUS_REFINED

    def _prepare_keyframe_bootstrap(self, target_keyframe: int, keyframe_candidate: KeyframeCandidate) -> dict | None:
        if self.depth_bootstrap_neighbors <= 0 or target_keyframe <= 0:
            return None

        start = max(0, int(target_keyframe) - self.depth_bootstrap_neighbors)
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
                "depth_observability_logits": output["observability_logits"],
                "depth_observability_rank": 2,
            }
        )
        self._update_candidate_depth(keyframe_candidate, result["refined_depth"])
        self.tracking_info["depth_bootstrap_runs"] += 1
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

    def _seed_source_pose_from_measurement(self, result: dict, source: int, target: int) -> None:
        edge = (result["ii"] == int(source)) & (result["jj"] == int(target))
        assert edge.any()
        edge_idx = int(torch.nonzero(edge, as_tuple=False)[0].item())
        relative_pose = result["relative_pose"][edge_idx].float().clone()
        relative_pose[:3] = relative_pose[:3] * self.buffer.depths_sens_scale[source, 0].float()
        source_pose = SE3(relative_pose.view(1, 7)).inv() * SE3(self.buffer.poses[target].float().view(1, 7))
        self.buffer.poses[source] = source_pose.data[0].to(dtype=self.buffer.poses.dtype)

    def _add_bootstrap_edges(
        self,
        current_keyframe: int,
        last_keyframe: int,
        bootstrap_result: dict | None,
    ) -> None:
        if bootstrap_result is None:
            return

        last_mask = bootstrap_result["jj"] == int(last_keyframe)
        if last_mask.any():
            last_result = self._measurement_subset(bootstrap_result, last_mask)
            added = self.graph.add_measurement_result(last_result)
            if int(added["n_added"]) == 0:
                raise RuntimeError(
                    f"bootstrap edge {current_keyframe}->{last_keyframe} was accepted but not inserted"
            )
            self._seed_source_pose_from_measurement(last_result, current_keyframe, last_keyframe)
            self.tracking_info["depth_bootstrap_edges_added"] += int(added["n_added"])

        other_mask = ~last_mask
        if other_mask.any():
            other_result = self._measurement_subset(bootstrap_result, other_mask)
            other_result = filter_local_edge_measurement(
                other_result,
                max_rotation_deg=self.local_edge_max_rotation_deg,
                max_translation=self.local_edge_max_translation,
                trans_conf_thresh=self.local_edge_outlier_trans_conf_thresh,
            )
            added = self.graph.add_measurement_result(other_result)
            self.tracking_info["depth_bootstrap_edges_added"] += int(added["n_added"])

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
        bootstrap_result = self._prepare_keyframe_bootstrap(target_keyframe, keyframe_candidate)
        current_keyframe = self._commit_keyframe(keyframe_candidate)
        self._add_bootstrap_edges(current_keyframe, last_keyframe, bootstrap_result)
        return {
            "accepted": True,
            "reason": "keyframe",
            "keyframe": current_keyframe,
            "coarse_flow_motion_score": motion_score,
        }
