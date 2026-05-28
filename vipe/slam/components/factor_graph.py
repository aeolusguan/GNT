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

import warnings

import torch

from GeoNT.models import GeoNTWrapper
from .buffer import GraphBuffer

# Disable all future warnings (mainly torch.cuda.amp related)
warnings.simplefilter(action="ignore", category=FutureWarning)


class FactorGraph:
    @staticmethod
    def coords_grid(ht, wd, **kwargs):
        y, x = torch.meshgrid(
            torch.arange(ht).to(**kwargs).float(),
            torch.arange(wd).to(**kwargs).float(),
            indexing="ij",
        )
        return torch.stack([x, y], dim=-1)
    
    def __init__(
        self,
        net: GeoNTWrapper,
        buffer: GraphBuffer,
        device: torch.device,
        max_factors: int,
        cross_view: bool,
    ):
        self.net = net
        self.buffer = buffer
        self.device = device
        self.max_factors = max_factors
        self.cross_view = cross_view and buffer.n_views > 1

        ht = buffer.height // 16
        wd = buffer.width // 16

        # edge connection are the same for all the views.
        self.ii = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj = torch.as_tensor([], dtype=torch.long, device=device)
        self.age = torch.as_tensor([], dtype=torch.long, device=device)

        # flow and info embed.
        self.embed = torch.zeros([0, ht, wd, 288], device=device, dtype=torch.half)

        # inactive and bad factors
        # - inactive factors are those who are removed by rm_factors(store=True)
        # They could be later revived in GeoNT if use_inactive=True)
        self.ii_inac = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj_inac = torch.as_tensor([], dtype=torch.long, device=device)
        self.embed_inac = torch.zeros([0, ht, wd, 288], device=device, dtype=torch.half)

    def patch_embed(self, ii, jj):
        fmap1 = self.buffer.fmaps[ii, 0]
        fmap2 = self.buffer.fmaps[jj, 0]
        bases = self.buffer.bases[ii, 0]
        intrinsics = self.buffer.intrinsics[0:1].expand(fmap1.shape[0], -1)
        return self.net.encode_flow(fmap1, fmap2, bases, intrinsics)

    def __filter_repeated_edges(self, ii, jj):
        """remove duplicate edges"""

        keep = torch.zeros(ii.shape[0], dtype=torch.bool, device=ii.device)
        eset = set(
            [(i.item(), j.item()) for i, j in zip(self.ii, self.jj)]
            + [(i.item(), j.item()) for i, j in zip(self.ii_inac, self.jj_inac)]
        )

        for k, (i, j) in enumerate(zip(ii, jj)):
            keep[k] = (i.item(), j.item()) not in eset

        return ii[keep], jj[keep]

    def add_factors(self, ii, jj, remove=False):
        """add edges to factor graph"""

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii, dtype=torch.long, device=self.device)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj, dtype=torch.long, device=self.device)

        # remove duplicate edges
        ii, jj = self.__filter_repeated_edges(ii, jj)

        if ii.shape[0] == 0:
            return

        # place limit on number of factors
        if (
            self.max_factors > 0
            and self.ii.shape[0] + ii.shape[0] > self.max_factors
            and remove
        ):
            ix = torch.arange(len(self.age))[torch.argsort(self.age).cpu()]
            self.rm_factors(ix >= self.max_factors - ii.shape[0], store=True)

        embed = self.patch_embed(ii, jj)

        self.ii = torch.cat([self.ii, ii], 0)
        self.jj = torch.cat([self.jj, jj], 0)
        self.age = torch.cat([self.age, torch.zeros_like(ii)], 0)

        self.embed = torch.cat([self.embed, embed], 0)


    def rm_factors(self, mask: torch.Tensor, store: bool = False):
        """drop edges from factor graph"""

        # store estimated factors
        if store:
            self.ii_inac = torch.cat([self.ii_inac, self.ii[mask]], 0)
            self.jj_inac = torch.cat([self.jj_inac, self.jj[mask]], 0)
            self.embed_inac = torch.cat([self.embed_inac, self.embed_inac[mask]], 0)

        self.ii = self.ii[~mask]
        self.jj = self.jj[~mask]
        self.age = self.age[~mask]
        self.embed = self.embed[~mask]

    def update(
        self,
        t0: int | None = None,  # will limit pose update to >= t0 if provided
        t1: int | None = None,  # will limit pose update to < t1 if provided
        use_inactive: bool = False,
        motion_only: bool = False,
        fixed_motion: bool = False,
    ):
        """run update operator on factor graph"""
        assert not (motion_only and fixed_motion)

        if t0 is None:
            t0 = int(max(1, self.ii.min().item() + 1))

        if t1 is None:
            t1 = int(max(self.ii.max().item(), self.jj.max().item()) + 1)

        if use_inactive:
            m = (self.ii_inac >= t0 - 3) & (self.jj_inac >= t0 - 3)
            ii = torch.cat([self.ii_inac[m], self.ii], 0)
            jj = torch.cat([self.jj_inac[m], self.jj], 0)
            embed = torch.cat([self.embed_inac[m], self.embed], 0)
        else:
            ii, jj, embed = self.ii, self.jj, self.embed

        


    def add_neighborhood_factors(self, t0, t1, r: int = 3):
        """
        add edges between neighboring frames within radius r
        (note that the edges are uni-directional, hence both 0,1 and 1,0 are added)
        """

        ii, jj = torch.meshgrid(torch.arange(t0, t1), torch.arange(t0, t1), indexing="ij")
        ii = ii.reshape(-1).to(dtype=torch.long, device=self.device)
        jj = jj.reshape(-1).to(dtype=torch.long, device=self.device)

        c = -1 if self.cross_view else 0
        keep = ((ii - jj).abs() > c) & ((ii - jj).abs() <= r)
        self.add_factors(ii[keep], jj[keep])