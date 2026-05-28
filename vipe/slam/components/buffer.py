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

import logging

import numpy as np
import rerun as rr
import torch
from einops import rearrange
from omegaconf.dictconfig import DictConfig


class GraphBuffer:
    def __init__(
        self,
        height: int,
        width: int,
        n_views: int,
        buffer_size: int,
        cross_view_idx: list[int] | None,
        device: torch.device = torch.device("cuda"),
    ):
        if cross_view_idx is None:
            cross_view_idx = [(i + 1) % n_views for i in range(n_views)]

        self.n_frames: int = 0

        self.height = height
        self.width = width
        self.n_views = n_views
        self.device = device
        
        assert self.height % 16 == 0 and self.width % 16 == 0

        # timestamp (frame index)
        self.tstamp = torch.zeros(buffer_size, device=device, dtype=torch.int)
        self.dirty = torch.zeros(buffer_size, device=device, dtype=torch.bool)
        # Rig pose defined as the 0-th view of each frame.
        self.poses = torch.zeros(buffer_size, 7, device=device, dtype=torch.float)
        self.poses[:] = torch.as_tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device=self.poses.device)
        # This will be the original intrinsics
        self.intrinsics = torch.zeros(
            self.n_views,
            4,
            device=device,
            dtype=torch.float,
        )
        # rig pose in a multi-view setting.
        self.rig = torch.zeros(self.n_views, 7, device=device, dtype=torch.float)
        self.rig[:] = torch.as_tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device=self.rig.device)

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
        self.depths_sens_scale = torch.zeros(
            buffer_size,
            self.n_views,
            device=device,
            dtype=torch.float,
        )
        # Non sky mask
        self.non_sky_masks = torch.zeros(
            buffer_size,
            n_views,
            self.height,
            self.width,
            device=device,
            dtype=torch.bool,
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

        # [..., 0] is time, [..., 1] is view
        assert len(cross_view_idx) == self.n_views
        self.cross_view_idx = torch.zeros(buffer_size, self.n_views, 2, device=device, dtype=torch.long)
        self.cross_view_idx[..., 0] = torch.arange(buffer_size, device=device)[:, None]
        self.cross_view_idx[..., 1] = torch.tensor(cross_view_idx, device=device).long()[None]

    def expand_edge_multiview(
        self,
        ii: torch.Tensor,
        jj: torch.Tensor,
        cross: bool = True,
        view_offset: int = 0,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
            Expand from edge (ii, jj) to (pi, qi, di, pj, qj, dj) which refers
        to the actual indices in the flattened buffer.
            In terms of ii == jj and cross=True this means to create cross-view edges of a single frame,
        where the number of edges will also be n_views, e.g. (0, 1), (1, 2), (2, 0) for n_views=3.

        Args:
            ii (torch.Tensor): edge source (M, )
            jj (torch.Tensor): edge target (M, )
            cross (bool): whether to create cross-view edges for the same frame

        Returns:
            pi (torch.Tensor): source pose index (M * n_views, )
            qi (torch.Tensor): source rig index (M * n_views, )
            di (torch.Tensor): source dense_disp index (M * n_views, )
            pj (torch.Tensor): target pose index (M * n_views, )
            qj (torch.Tensor): target rig index (M * n_views, )
            dj (torch.Tensor): target dense_disp index (M * n_views, )
        """
        qi = torch.arange(self.n_views, device=self.device).reshape(1, -1).to(self.device)
        qi = qi.repeat(ii.shape[0], 1)
        pi = ii.reshape(-1, 1).repeat(1, self.n_views).to(self.device)
        qj = torch.arange(self.n_views, device=self.device).reshape(1, -1).to(self.device)
        qj = qj.repeat(jj.shape[0], 1)
        pj = jj.reshape(-1, 1).repeat(1, self.n_views).to(self.device)

        if cross:
            cross_mask = ii == jj
            if torch.any(cross_mask):
                t, v = self.cross_view_idx[pi[cross_mask], qi[cross_mask]].unbind(-1)
                pj[cross_mask], qj[cross_mask] = t, v

        qj = (qj + view_offset) % self.n_views

        di = pi * self.n_views + qi
        dj = pj * self.n_views + qj

        return (
            pi.reshape(-1),
            qi.reshape(-1),
            di.reshape(-1),
            pj.reshape(-1),
            qj.reshape(-1),
            dj.reshape(-1),
        )