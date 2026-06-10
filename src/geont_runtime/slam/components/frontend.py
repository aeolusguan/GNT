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
from .buffer import GraphBuffer
from .factor_graph import FactorGraph


class SLAMFrontend:
    """
    Frontend is called given every new frame. Currently it is a no-op for non-keyframe frames.
    For keyframe, it handles the system initialization and partial update logic (i.e. use GeoNT to get pose for this kf).
    """

    def __init__(self, net: GeoNTWrapper, video: GraphBuffer, args, device: torch.device):
        self.video = video
        self.max_factors = args.max_factors
        self.graph = FactorGraph(
            net,
            video,
            device,
            max_factors=self.max_factors,
        )

        # Number of frames that the frontend has so far optimized.
        self.t1 = 0

        # frontend variables
        self.is_initialized = False

        # Number of frames to wait before initializing (default 8)
        self.warmup = args.warmup
        self.frontend_radius = args.frontend_radius
        self.frontend_thresh = args.frontend_thresh
        self.seq_init = args.seq_init
        self.pgo_iters = args.pgo_iters
        self.pgo_damping = args.pgo_damping
        self.pgo_lm_max_attempts = args.pgo_lm_max_attempts
        self.pgo_huber_delta = args.pgo_huber_delta
        self.pgo_scale_conf = args.pgo_scale_conf
        self.pgo_mode = args.pgo_mode
        self.pgo_rotation_only = args.pgo_rotation_only
        self.pgo_backend = args.pgo_backend
        self.use_fp16 = args.use_fp16

    def _run_pgo(self):
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

    def _marginalize_oldest(self):
        keyframe = self.graph.oldest_active_keyframe()
        if keyframe is None:
            return None
        result = self.graph.marginalize_keyframe(keyframe, use_fp16=self.use_fp16)
        return result

    def _marginalize_all(self):
        marginalized = False
        while self.graph.ii.numel() > 0:
            if self._marginalize_oldest() is None:
                break
            marginalized = True
        if marginalized:
            self._run_pgo()

    def finish(self):
        """Flush the active graph at the end of a finite stream."""
        if not self.is_initialized:
            if self.video.n_frames <= 1:
                return
            self.__initialize()

        self._marginalize_all()

    def run_second_pass(self):
        """Refine initialized keyframes with projection-proximity factors."""
        if self.video.n_frames <= 1:
            return

        self.t1 = self.video.n_frames
        self.is_initialized = True
        self.graph.add_proximity_factors(
            0,
            self.t1,
            thresh=self.frontend_thresh,
        )
        self._marginalize_all()

    def __initialize(self):
        """initialize the SLAM system with keyframe idx [t0, t1)"""

        self.t1 = self.video.n_frames

        self.graph.add_neighborhood_factors(
            0,
            self.t1,
            r=1 if self.seq_init else self.frontend_radius,
            thresh=self.frontend_thresh,
        )
        self.is_initialized = True

    def __update(self):
        """Add temporal neighborhood factors for newly added keyframes."""
        prev_t1 = self.t1
        self.t1 = self.video.n_frames
        if self.t1 <= prev_t1:
            return

        self.graph.increment_age()
        self.graph.add_neighborhood_factors(
            max(0, self.t1 - self.frontend_radius - 1),
            self.t1,
            r=self.frontend_radius,
            thresh=self.frontend_thresh,
        )
        
    def run(self):
        """main update"""

        # do initialization
        if not self.is_initialized and self.video.n_frames == self.warmup:
            self.__initialize()

        # do update if new keyframe is added.
        elif self.is_initialized and self.t1 < self.video.n_frames:
            self.__update()
