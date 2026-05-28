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
from omegaconf import DictConfig

from lietorch import SE3
from GeoNT.models import GeoNTWrapper
from .buffer import GraphBuffer
from .factor_graph import FactorGraph


class SLAMFrontend:
    """
    Frontend is called given every new frame. Currently it is a no-op for non-keyframe frames.
    For keyframe, it handles the system initialization and partial update logic (i.e. use GeoNT to get pose for this kf).
    """

    def __init__(self, net: GeoNTWrapper, video: GraphBuffer, args: DictConfig, device: torch.device):
        self.video = video
        self.graph = FactorGraph(
            net,
            video,
            device,
            max_factors=48,
            incremental=True,
            cross_view=args.cross_view,
        )

        # Number of frames that the frontend has so far optimized.
        self.t1 = 0

        # frontend variables
        self.is_initialized = False

        self.max_age = 25
        
        # Number of frames to wait before initializing (default 8)
        self.args = args
        self.warmup = args.warmup
        self.frontend_nms = args.frontend_nms
        self.keyframe_thresh = args.keyframe_thresh
        self.frontend_window = args.frontend_window
        self.frontend_thresh = args.frontend_thresh
        self.frontend_radius = args.frontend_radius

    def __initialize(self):
        """initialize the SLAM system with keyframe idx [t0, t1)"""

        self.t1 = self.video.n_frames

        self.graph.add_neighborhood_factors(0, self.t1, r=1 if self.args.seq_init else 3)
        
    def run(self):
        """main update"""

        # do initialization
        if not self.is_initialized and self.video.n_frames == self.warmup:
            self.__initialize()

        # do update if new keyframe is added.
        elif self.is_initialized and self.t1 < self.video.n_frames:
            self.__update()