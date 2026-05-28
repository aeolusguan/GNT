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

from GeoNT.models import GeoNTWrapper


class MotionFilter:
    """
    This class is used to filter incoming frames and extract features.
    This module re-uses GeoNT's network to detect scene changes without considering the mask.
    For multi-view input, spawn keyframes if any of the views exceed the threshold.
    """

    def __init__(
        self,
        gnt: GeoNTWrapper,
        thresh: float,
        device: torch.device = torch.device("cuda"),
    ):
        self.net = gnt
        self.thresh = thresh
        self.device = device
        self.initialized = False

    @staticmethod
    def coords_grid(ht, wd, **kwargs):
        y, x = torch.meshgrid(
            torch.arange(ht).to(**kwargs).float(),
            torch.arange(wd).to(**kwargs).float(),
            indexing="ij",
        )
        return torch.stack([x, y], dim=-1)
    
    @torch.no_grad()
    def check(self, images: torch.Tensor, intrinsics: torch.Tensor) -> bool:
        """
        main update operation - run on every frame in video

        Args:
            image (torch.Tensor): VCHW image RGB 0-1
            intrinsics (torch.Tensor): V4, each row: [fx, fy, cx, cy]
        """

        num_views = images.shape[0]
        ht = images.shape[-2] // 8
        wd = images.shape[-1] // 8

        # extract features
        gmap, mono_depth, non_sky_mask = self.net.encode_features(images, intrinsics)

        ### always add first frame to the depth video ###
        if not self.initialized:
            bases = self.net.encode_bases(mono_depth, non_sky_mask)
            # Store features of the last keyframe
            self.f_fmap, self.f_mono, self.f_non_sky_mask, self.f_bases = gmap, mono_depth, non_sky_mask, bases
            self.current_frame_idx = 0
            self.last_kf_frame_idx = 0
            self.initialized = True
            return True
        
        ### only add new frame if there is enough motion ###
        else:
            self.current_frame_idx += 1

            # approximate flow magnitude using initialization
            delta, net = self.net.flow_init(self.f_fmap, gmap, self.f_bases)
            # flow: (V, 2, ht//8, wd//8)

            dense_flow = delta.norm(dim=1)
            dense_motion_score = dense_flow.mean([1, 2])
            # Across views [max is the most conservative, while min don't add a lot of KFs]
            dense_motion_score = dense_motion_score.min().item()

            # check motion magnitude / add new frame to video
            if dense_motion_score > self.thresh:
                bases = self.net.encode_bases(mono_depth, non_sky_mask)
                self.f_fmap, self.f_mono, self.f_non_sky_mask, self.f_bases = gmap, mono_depth, non_sky_mask, bases
                self.last_kf_frame_idx = self.current_frame_idx
                return True
            
            else:
                return False

