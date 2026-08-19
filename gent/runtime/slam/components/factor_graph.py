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
import time

import torch

from .buffer import GraphBuffer
from .measurements import PoseMeasurement
from gent.runtime.slam.pgo import DEFAULT_LM_MAX_ATTEMPTS, optimize_sim3_pose_graph
from gent.runtime.slam.pgo.marginalized_local import MarginalizedLocalPGO

warnings.simplefilter(action="ignore", category=FutureWarning)


class PoseGraphEdges:
    def __init__(self, device: torch.device):
        self.ii = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj = torch.as_tensor([], dtype=torch.long, device=device)
        self.edge_key = torch.as_tensor([], dtype=torch.long, device=device)
        self.relative_pose = torch.zeros([0, 7], device=device, dtype=torch.float)
        self.relative_log_scale = torch.zeros([0], device=device, dtype=torch.float)
        self.confidence = torch.zeros([0, 3], device=device, dtype=torch.float)
        self.pgo_info = {}
        self._edge_key_stride = 1_000_000

    def _make_edge_key(self, ii: torch.Tensor, jj: torch.Tensor) -> torch.Tensor:
        return ii.long() * self._edge_key_stride + jj.long()

    def _first_unique_mask(self, keys: torch.Tensor) -> torch.Tensor:
        if keys.numel() == 0:
            return torch.zeros_like(keys, dtype=torch.bool)
        _, inverse = torch.unique(keys, sorted=False, return_inverse=True)
        positions = torch.arange(keys.numel(), device=keys.device, dtype=torch.long)
        first = torch.full(
            (int(inverse.max().item()) + 1,),
            keys.numel(),
            device=keys.device,
            dtype=torch.long,
        )
        first.scatter_reduce_(0, inverse, positions, reduce="amin", include_self=True)
        keep = torch.zeros(keys.numel(), device=keys.device, dtype=torch.bool)
        keep[first] = True
        return keep

    def new_edge_mask(self, ii: torch.Tensor, jj: torch.Tensor) -> torch.Tensor:
        keys = self._make_edge_key(ii, jj)
        keep = self._first_unique_mask(keys)
        if self.edge_key.numel() > 0:
            keep &= ~torch.isin(keys, self.edge_key)
        return keep

    def add(
        self,
        ii: torch.Tensor,  # [E]
        jj: torch.Tensor,  # [E]
        pose: torch.Tensor,  # [E,7]
        relative_log_scale: torch.Tensor,  # [E]
        confidence: torch.Tensor,  # [E,3]
    ) -> torch.Tensor:
        keep_mask = self.new_edge_mask(ii, jj)
        if not keep_mask.any():
            return torch.zeros(ii.shape[0], device=ii.device, dtype=torch.bool)

        keep_idx = torch.nonzero(keep_mask, as_tuple=False).flatten()
        self.ii = torch.cat((self.ii, ii[keep_idx].long()), dim=0)
        self.jj = torch.cat((self.jj, jj[keep_idx].long()), dim=0)
        self.edge_key = torch.cat((self.edge_key, self._make_edge_key(ii[keep_idx], jj[keep_idx])), dim=0)
        self.relative_pose = torch.cat((self.relative_pose, pose[keep_idx].float()), dim=0)
        self.relative_log_scale = torch.cat((self.relative_log_scale, relative_log_scale[keep_idx].float()), dim=0)
        self.confidence = torch.cat((self.confidence, confidence[keep_idx].float()), dim=0)
        return keep_mask

class FactorGraph:
    def __init__(
        self,
        buffer: GraphBuffer,
        device: torch.device,
    ):
        self.buffer = buffer
        self.device = device
        self.edges = PoseGraphEdges(device)
        self.local_pgo: MarginalizedLocalPGO | None = None
        self.local_pgo_edge_cursor = 0

    def projection_distance(self, ii: torch.Tensor, jj: torch.Tensor, stride: int = 8) -> torch.Tensor:
        if ii.numel() == 0:
            return torch.empty_like(ii, dtype=torch.float)

        from . import geom_cuda

        n_frames = self.buffer.n_frames
        return geom_cuda.projection_distance(
            self.buffer.poses[:n_frames].float(),
            self.buffer.depths[:n_frames, 0].float(),
            self.buffer.scale[:n_frames, 0].float(),
            self.buffer.non_sky_masks[:n_frames, 0],
            self.buffer.intrinsics[:n_frames, 0].float() / float(stride),
            ii,
            jj,
            stride,
        )

    def _filter_new_edges(self, ii: torch.Tensor, jj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        keep = self.edges.new_edge_mask(ii, jj)
        if not keep.any():
            empty = torch.as_tensor([], dtype=torch.long, device=ii.device)
            return empty, empty
        return ii[keep], jj[keep]

    def add_measurement_result(self, result: PoseMeasurement) -> None:
        self.edges.add(
            result.ii,
            result.jj,
            result.relative_pose,
            result.relative_log_scale,
            result.confidence,
        )

    def _bidirectional_pairs(self, ii: torch.Tensor, jj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.cat((ii, jj), dim=0), torch.cat((jj, ii), dim=0)

    def _select_proximity_nms(
        self,
        ii: torch.Tensor,
        jj: torch.Tensor,
        distance: torch.Tensor,
        nms: int,
        max_pairs: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if ii.numel() == 0:
            return ii, jj
        if max_pairs == 0:
            empty = torch.as_tensor([], dtype=torch.long, device=self.device)
            return empty, empty

        ii_cpu = ii.cpu().tolist()
        jj_cpu = jj.cpu().tolist()
        order = torch.argsort(distance.cpu()).tolist()
        selected: list[tuple[int, int]] = []
        for idx in order:
            i = int(ii_cpu[idx])
            j = int(jj_cpu[idx])
            if nms > 0:
                suppressed = False
                for selected_i, selected_j in selected:
                    suppression_radius = max(min(abs(selected_i - selected_j) - 2, int(nms)), 0)
                    if abs(i - selected_i) + abs(j - selected_j) <= suppression_radius:
                        suppressed = True
                        break
                if suppressed:
                    continue
            selected.append((i, j))
            if max_pairs > 0 and len(selected) >= max_pairs:
                break

        if not selected:
            empty = torch.as_tensor([], dtype=torch.long, device=self.device)
            return empty, empty
        selected_ii = torch.as_tensor([pair[0] for pair in selected], dtype=torch.long, device=self.device)
        selected_jj = torch.as_tensor([pair[1] for pair in selected], dtype=torch.long, device=self.device)
        return selected_ii, selected_jj

    def select_nonlocal_proximity_edges(
        self,
        source_start: int,
        target_start: int,
        end: int,
        radius: int,
        nms: int,
        thresh: float = 16.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sources = torch.arange(source_start, end, device=self.device)
        targets = torch.arange(target_start, end, device=self.device)
        empty = torch.as_tensor([], dtype=torch.long, device=self.device)
        if sources.numel() == 0 or targets.numel() == 0:
            return empty, empty

        ii, jj = torch.meshgrid(sources, targets, indexing="ij")
        ii = ii.reshape(-1).long()
        jj = jj.reshape(-1).long()

        proximity = (ii - jj) > int(radius)
        ii, jj = self._filter_new_edges(ii[proximity], jj[proximity])
        if ii.numel() == 0:
            return empty, empty

        distance = self.projection_distance(ii, jj)
        keep = distance <= float(thresh)
        if not keep.any():
            return empty, empty

        ii, jj, distance = ii[keep], jj[keep], distance[keep]
        ii, jj = self._select_proximity_nms(
            ii,
            jj,
            distance,
            nms=int(nms),
            max_pairs=-1,
        )
        ii, jj = self._bidirectional_pairs(ii, jj)
        return self._filter_new_edges(ii, jj)

    def optimize_pose_graph(
        self,
        anchor: int = 0,
        n_iters: int = 12,
        damping: float = 1e-3,
        lm_max_attempts: int = DEFAULT_LM_MAX_ATTEMPTS,
        huber_delta: float = 1.0,
        mode: str = "staged",
        backend: str = "cuda_eigen",
        moge_mode_nis: bool = False,
        moge_mode_count: int = 8,
        moge_mode_nis_cutoff: float = 0.01,
    ):
        if self.edges.ii.numel() == 0:
            return None
        n_frames = self.buffer.n_frames
        initial_poses = self.buffer.poses[:n_frames].clone()
        initial_log_scales = torch.log(self.buffer.scale[:n_frames, 0].clamp_min(1e-6))
        moge_log_scales = torch.log(self.buffer.depths_sens_scale[:n_frames, 0].clamp_min(1e-6))
        result = optimize_sim3_pose_graph(
            n_nodes=n_frames,
            ii=self.edges.ii,
            jj=self.edges.jj,
            rel_poses=self.edges.relative_pose,
            rel_log_scales=self.edges.relative_log_scale,
            edge_conf=self.edges.confidence,
            initial_poses=initial_poses,
            initial_log_scales=initial_log_scales,
            moge_log_scales=moge_log_scales,
            anchor=anchor,
            n_iters=n_iters,
            damping=damping,
            lm_max_attempts=lm_max_attempts,
            huber_delta=huber_delta,
            mode=mode,
            backend=backend,
            moge_mode_nis=moge_mode_nis,
            moge_mode_count=moge_mode_count,
            moge_mode_nis_cutoff=moge_mode_nis_cutoff,
        )
        self.buffer.poses[: self.buffer.n_frames] = result.poses.to(dtype=self.buffer.poses.dtype)
        new_scales = torch.exp(result.log_scales).to(dtype=self.buffer.scale.dtype)
        self.buffer.scale[: self.buffer.n_frames, 0] = new_scales
        self.edges.pgo_info = result.info
        return result

    def optimize_local_pgo(
        self,
        finalized_end: int,
        window_size: int,
        n_iters: int = 12,
        damping: float = 1e-3,
        lm_max_attempts: int = DEFAULT_LM_MAX_ATTEMPTS,
        huber_delta: float = 1.0,
        moge_mode_nis_cutoff: float = 0.01,
    ):
        assert 1 < finalized_end <= self.buffer.n_frames
        if self.edges.ii.numel() == 0:
            return None
        if self.local_pgo is None:
            self.local_pgo = MarginalizedLocalPGO(window_size)
        assert self.local_pgo.window_size == int(window_size)

        state_start = self.local_pgo.window_start
        initial_log_scales = torch.log(
            self.buffer.scale[state_start:finalized_end, 0].clamp_min(1e-6)
        )
        moge_log_scales = torch.log(
            self.buffer.depths_sens_scale[state_start:finalized_end, 0].clamp_min(1e-6)
        )
        edge_start = self.local_pgo_edge_cursor
        edge_end = int(self.edges.ii.numel())
        start_time = time.perf_counter()
        window_start, result = self.local_pgo.step(
            finalized_end=finalized_end,
            poses=self.buffer.poses[state_start:finalized_end],
            log_scales=initial_log_scales,
            moge_log_scales=moge_log_scales,
            ii=self.edges.ii[edge_start:edge_end],
            jj=self.edges.jj[edge_start:edge_end],
            rel_poses=self.edges.relative_pose[edge_start:edge_end],
            rel_log_scales=self.edges.relative_log_scale[edge_start:edge_end],
            edge_conf=self.edges.confidence[edge_start:edge_end],
            n_iters=n_iters,
            damping=damping,
            lm_max_attempts=lm_max_attempts,
            huber_delta=huber_delta,
            nis_cutoff=moge_mode_nis_cutoff,
        )
        self.local_pgo_edge_cursor = edge_end
        result.info["runtime_sec"] = time.perf_counter() - start_time
        self.buffer.poses[window_start:finalized_end] = result.poses.to(dtype=self.buffer.poses.dtype)
        new_scales = torch.exp(result.log_scales).to(dtype=self.buffer.scale.dtype)
        self.buffer.scale[window_start:finalized_end, 0] = new_scales
        self.edges.pgo_info = result.info
        return result
