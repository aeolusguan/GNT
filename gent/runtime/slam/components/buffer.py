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

from dataclasses import dataclass

import torch


DEPTH_STATUS_MONO = 1
DEPTH_STATUS_REFINED = 2


@dataclass(kw_only=True)
class KeyframeCandidate:
    frame_idx: int
    fmap: torch.Tensor
    depth: torch.Tensor
    depth_sens_normed: torch.Tensor
    depth_token: torch.Tensor
    scale: torch.Tensor
    mask: torch.Tensor
    bases: torch.Tensor
    intrinsics: torch.Tensor
    depth_status: int = DEPTH_STATUS_MONO


class GraphBuffer:
    def __init__(
        self,
        height: int,
        width: int,
        buffer_size: int,
        depth_token_dim: int,
        device: torch.device = torch.device("cuda"),
    ):
        self.n_frames: int = 0

        self.height = height
        self.width = width
        self.n_views = 1
        self.device = device
        
        assert self.height % 16 == 0 and self.width % 16 == 0

        # Frame indices are CPU timeline metadata; GPU kernels never consume them.
        self.tstamp = torch.zeros(buffer_size, dtype=torch.long)
        self.dirty = torch.zeros(buffer_size, device=device, dtype=torch.bool)
        self.depth_status = torch.zeros(buffer_size, device=device, dtype=torch.uint8)
        self.depth_dirty = torch.zeros(buffer_size, device=device, dtype=torch.bool)
        # Camera pose for each keyframe.
        self.poses = torch.zeros(buffer_size, 7, device=device, dtype=torch.float)
        self.poses[:] = torch.as_tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device=self.poses.device)
        # Resized/cropped pinhole intrinsics for every committed keyframe view.
        self.intrinsics = torch.zeros(
            buffer_size,
            self.n_views,
            4,
            device=device,
            dtype=torch.float,
        )

        # Inferred depth
        self.depths = torch.ones(
            buffer_size,
            self.n_views,
            self.height,
            self.width,
            device=device,
            dtype=torch.half,
        )

        # Sensor depth (normalized)
        self.depths_sens_normed = torch.zeros(
            buffer_size,
            self.n_views,
            self.height,
            self.width,
            device=device,
            dtype=torch.half,
        )
        # Raw MoGe depth scale for the normalized monocular prior. This is fixed
        # after keyframe commit; PGO updates `scale` instead.
        self.depths_sens_scale = torch.zeros(
            buffer_size,
            self.n_views,
            device=device,
            dtype=torch.float,
        )
        self.scale = torch.zeros_like(self.depths_sens_scale)
        # Non sky mask
        self.non_sky_masks = torch.zeros(
            buffer_size,
            self.n_views,
            self.height,
            self.width,
            device=device,
            dtype=torch.bool,
        )
        # Fixed normalized-depth and validity patch tokens.
        self.depth_tokens = torch.zeros(
            buffer_size,
            self.n_views,
            self.height // 16,
            self.width // 16,
            depth_token_dim,
            device=device,
            dtype=torch.half,
        )

        # Flow attributes
        # - feature maps
        self.fmaps = torch.zeros(
            buffer_size,
            self.n_views,
            256,
            self.height // 8,
            self.width // 8,
            device=device,
            dtype=torch.half,
        )
        # - bases
        self.bases = torch.zeros(
            buffer_size,
            self.n_views,
            96,
            self.height // 16,
            self.width // 16,
            device=device,
            dtype=torch.half,
        )
