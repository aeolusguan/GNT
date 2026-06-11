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
# -------------------------------------------------------------------------------------------------
# This file includes code originally from the DROID-SLAM repository:
# https://github.com/cvg/DROID-SLAM
# Licensed under the MIT License. See THIRD_PARTY_LICENSES.md for details.
# -------------------------------------------------------------------------------------------------

import torch
from geont.models import GeoNTWrapper
from geont_runtime.utils.logging import pbar
from .buffer import GraphBuffer
from .factor_graph import FactorGraph


class SLAMFrontend:
    """
    Offline frontend refinement over keyframes initialized by the one-pass initializer.
    """

    def __init__(self, net: GeoNTWrapper, video: GraphBuffer, args, device: torch.device):
        self.video = video
        self.max_factors = args.max_factors
        self.graph = FactorGraph(
            net,
            video,
            device,
            max_factors=self.max_factors,
            frontend_outlier_trans_conf_thresh=args.frontend_outlier_trans_conf_thresh,
        )

        self.t1 = 0
        self.initialized = True
        self.frontend_radius = args.frontend_radius
        self.proximity_window = args.proximity_window
        self.proximity_recent = args.proximity_recent
        self.proximity_nms = args.proximity_nms
        self.proximity_thresh = args.proximity_thresh
        self.frontend_min_edges_per_source = args.frontend_min_edges_per_source
        self.pgo_iters = args.pgo_iters
        self.pgo_damping = args.pgo_damping
        self.pgo_lm_max_attempts = args.pgo_lm_max_attempts
        self.pgo_huber_delta = args.pgo_huber_delta
        self.pgo_scale_conf = args.pgo_scale_conf
        self.pgo_mode = args.pgo_mode
        self.pgo_rotation_only = args.pgo_rotation_only
        self.pgo_backend = args.pgo_backend
        self.use_fp16 = args.use_fp16

    def _run_final_full_graph_pgo(self):
        return self.graph.optimize_finalized_pose_graph(
            anchor=0,
            n_iters=self.pgo_iters,
            damping=self.pgo_damping,
            lm_max_attempts=self.pgo_lm_max_attempts,
            huber_delta=self.pgo_huber_delta,
            scale_conf=self.pgo_scale_conf,
            mode="rotation_only" if self.pgo_rotation_only else self.pgo_mode,
            backend=self.pgo_backend,
        )

    def _ready_keyframes(self, min_edges: int | None = None) -> list[int]:
        if self.graph.ii.numel() == 0:
            return []

        keyframes, counts = torch.unique(self.graph.ii, return_counts=True)
        if min_edges is not None:
            keyframes = keyframes[counts >= int(min_edges)]
        return [int(keyframe) for keyframe in keyframes.cpu().tolist()]

    def _window_deadline_keyframes(self, end: int) -> list[int]:
        if end < self.proximity_window or self.graph.ii.numel() == 0:
            return []

        last_in_window = end - self.proximity_window
        keyframes = torch.unique(self.graph.ii)
        keyframes = keyframes[keyframes <= int(last_in_window)]
        return [int(keyframe) for keyframe in keyframes.cpu().tolist()]

    def _marginalize_sources(self, min_edges: int | None = None) -> bool:
        marginalized = False
        for keyframe in self._ready_keyframes(min_edges=min_edges):
            result = self.graph.marginalize_keyframe(keyframe, use_fp16=self.use_fp16)
            if result is None:
                continue
            marginalized = True
        return marginalized

    def _marginalize_window_deadline_sources(self, end: int) -> bool:
        marginalized = False
        for keyframe in self._window_deadline_keyframes(end):
            result = self.graph.marginalize_keyframe(keyframe, use_fp16=self.use_fp16)
            if result is None:
                raise RuntimeError(f"deadline source keyframe {keyframe} has no factors to marginalize")
            marginalized = True
        return marginalized

    def _marginalize_until_within_budget(self) -> bool:
        if self.max_factors <= 0:
            return False

        marginalized = False
        while self.graph.active_edge_count() > self.max_factors:
            keyframe = self.graph.oldest_active_keyframe()
            if keyframe is None:
                raise RuntimeError("active frontend edge budget exceeded but no active keyframe exists")
            result = self.graph.marginalize_keyframe(keyframe, use_fp16=self.use_fp16)
            if result is None:
                raise RuntimeError(f"active source keyframe {keyframe} has no factors to marginalize")
            marginalized = True
        return marginalized

    def run(self, initialized_edges):
        """Refine initialized keyframes with projection-proximity factors."""
        if self.video.n_frames <= 1:
            return

        assert self.initialized
        assert self.graph.ii.numel() == 0
        self.t1 = self.video.n_frames
        self.graph.edges.add_from(initialized_edges)

        for end in pbar(range(2, self.t1 + 1), desc="Offline frontend sweep"):
            source_start = max(end - self.proximity_recent, 0)
            window_start = max(end - self.proximity_window, 0)
            self.graph.add_proximity_factors(
                source_start,
                window_start,
                end,
                radius=self.frontend_radius,
                nms=self.proximity_nms,
                thresh=self.proximity_thresh,
            )

        self._marginalize_sources()
        self._run_final_full_graph_pgo()
