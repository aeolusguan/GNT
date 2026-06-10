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
from lietorch import SE3

from geont.geometry.projective_ops import induced_flow
from geont.models import GeoNTWrapper
from .buffer import GraphBuffer
from geont_runtime.slam.pgo import DEFAULT_LM_MAX_ATTEMPTS, optimize_sim3_pose_graph
from geont_runtime.slam.pgo.replay import make_pgo_replay_graph

# Disable all future warnings (mainly torch.cuda.amp related)
warnings.simplefilter(action="ignore", category=FutureWarning)


class PoseGraphEdges:
    def __init__(self, device: torch.device):
        self.ii = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj = torch.as_tensor([], dtype=torch.long, device=device)
        self.relative_pose = torch.zeros([0, 7], device=device, dtype=torch.float)
        self.relative_scale = torch.zeros([0], device=device, dtype=torch.float)
        self.confidence = torch.zeros([0, 2], device=device, dtype=torch.float)
        self.pgo_info = {}
        self.pgo_replay = {}

    def add(
        self,
        ii: torch.Tensor,
        jj: torch.Tensor,
        pose: torch.Tensor,
        scale: torch.Tensor,
        confidence: torch.Tensor,
    ) -> int:
        existing = {(int(i), int(j)) for i, j in zip(self.ii.tolist(), self.jj.tolist())}
        keep = []
        for idx, pair in enumerate(zip(ii.tolist(), jj.tolist())):
            if (int(pair[0]), int(pair[1])) not in existing:
                keep.append(idx)
        if not keep:
            return 0
        keep = torch.as_tensor(keep, device=ii.device, dtype=torch.long)
        self.ii = torch.cat((self.ii, ii[keep].long()), dim=0)
        self.jj = torch.cat((self.jj, jj[keep].long()), dim=0)
        self.relative_pose = torch.cat((self.relative_pose, pose[keep].float()), dim=0)
        self.relative_scale = torch.cat((self.relative_scale, scale[keep].float()), dim=0)
        self.confidence = torch.cat((self.confidence, confidence[keep].float()), dim=0)
        return int(keep.numel())


class InitializationFactorGraph:
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
    ):
        self.net = net
        self.buffer = buffer
        self.device = device

        # edge connection are the same for all the views.
        self.ii = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj = torch.as_tensor([], dtype=torch.long, device=device)

    @torch.no_grad()
    def patch_embed(self, ii, jj):
        fmap1 = self.buffer.fmaps[ii, 0].float()
        fmap2 = self.buffer.fmaps[jj, 0].float()
        bases = self.buffer.bases[ii, 0].float()
        intrinsics = self.buffer.intrinsics[0:1].expand(fmap1.shape[0], -1)
        with torch.amp.autocast(device_type=self.device.type, enabled=False):
            return self.net.encode_flow(fmap1, fmap2, bases, intrinsics)

    def frame_distance(self, ii, jj):
        fmap1 = self.buffer.fmaps[ii, 0].float()
        fmap2 = self.buffer.fmaps[jj, 0].float()
        bases = self.buffer.bases[ii, 0].float()
        with torch.amp.autocast(device_type=self.device.type, enabled=False):
            flow, _ = self.net.flow_init(fmap1, fmap2, bases)
        return flow.norm(dim=1).mean(dim=(1, 2))

    def projection_distance(self, ii, jj, stride: int = 8, chunk_size: int = 1024):
        n_frames = self.buffer.n_frames
        poses = SE3(self.buffer.poses[:n_frames].float())[None]
        depth = self.buffer.depths[:n_frames, 0].float()
        scale = self.buffer.depths_sens_scale[:n_frames, 0].float()
        metric_depth = depth * scale[:, None, None]
        disps = 1.0 / metric_depth.clamp_min(1e-6)
        disps = disps[:, stride // 2 - 1 :: stride, stride // 2 - 1 :: stride][None]
        source_masks = self.buffer.non_sky_masks[:n_frames, 0]
        source_masks = source_masks[:, stride // 2 - 1 :: stride, stride // 2 - 1 :: stride][None]
        intrinsics = (self.buffer.intrinsics[0] / float(stride)).expand(n_frames, -1)[None]

        distances = []
        for start in range(0, ii.numel(), chunk_size):
            end = min(start + chunk_size, ii.numel())
            flow, valid = induced_flow(poses, disps, intrinsics, ii[start:end], jj[start:end])
            flow_mag = flow.norm(dim=-1).clamp(max=256.0)
            valid = valid.squeeze(-1) * source_masks[:, ii[start:end]].float()
            valid_count = valid.sum(dim=(-1, -2)).clamp_min(1.0)
            distance = (flow_mag * valid).sum(dim=(-1, -2)) / valid_count
            valid_ratio = valid.mean(dim=(-1, -2))
            distance = torch.where(valid_ratio > 0.5, distance, torch.full_like(distance, torch.inf))
            distances.append(distance.squeeze(0))
        return torch.cat(distances, dim=0)

    def _filter_repeated_edges(self, ii, jj):
        """remove duplicate edges"""

        keep = torch.zeros(ii.shape[0], dtype=torch.bool, device=ii.device)
        eset = set((i.item(), j.item()) for i, j in zip(self.ii, self.jj))

        for k, (i, j) in enumerate(zip(ii, jj)):
            keep[k] = (i.item(), j.item()) not in eset

        return ii[keep], jj[keep]

    def add_factors(self, ii, jj):
        """add edges to factor graph"""

        ii, jj = self._filter_repeated_edges(ii, jj)
        if ii.numel() == 0:
            return

        self.ii = torch.cat([self.ii, ii], 0)
        self.jj = torch.cat([self.jj, jj], 0)

    def rm_factors(self, mask: torch.Tensor):
        """drop edges from factor graph"""

        self.ii = self.ii[~mask]
        self.jj = self.jj[~mask]

    def add_neighborhood_factors(
        self,
        t0: int,
        t1: int,
        r: int = 3,
        thresh: float = 16.0,
    ):
        ii, jj = torch.meshgrid(
            torch.arange(t0, t1, device=self.device),
            torch.arange(t0, t1, device=self.device),
            indexing="ij",
        )
        ii = ii.reshape(-1).long()
        jj = jj.reshape(-1).long()

        keep = ((ii - jj).abs() > 0) & ((ii - jj).abs() <= r)
        ii, jj = ii[keep], jj[keep]
        ii, jj = self._filter_repeated_edges(ii, jj)
        if ii.numel() == 0:
            return

        flow = self.frame_distance(ii, jj)
        keep = flow <= float(thresh)
        latest = t1 - 1
        latest_edges = (ii == latest) | (jj == latest)
        if not (keep & latest_edges).any():
            latest_edges = torch.nonzero(latest_edges, as_tuple=False).flatten()
            assert latest_edges.numel() > 0
            latest_dist = (ii[latest_edges] - jj[latest_edges]).abs()
            keep[latest_edges[torch.argmin(latest_dist)]] = True
        self.add_factors(ii[keep], jj[keep])

    def active_edge_count(self) -> int:
        return int(self.ii.numel())

    def active_keyframe_count(self) -> int:
        if self.ii.numel() == 0:
            return 0
        keyframes = torch.unique(self.ii)
        return int(keyframes.numel())

    def oldest_keyframe(self) -> int | None:
        if self.ii.numel() == 0:
            return None
        return int(self.ii.min().item())

    def _incident_neighbors(self, keyframe: int) -> torch.Tensor:
        neighbors = self.jj[self.ii == keyframe]
        return neighbors

    def _motion_tokens_from_keyframe(self, keyframe: int, neighbors: torch.Tensor) -> torch.Tensor:
        ii = torch.full_like(neighbors, keyframe)
        return self.patch_embed(ii, neighbors)

    def _canonicalize_edge_conf(self, conf: torch.Tensor, n_edges: int) -> torch.Tensor:
        """Return edge confidence as (E, 2)."""
        assert conf.shape[0] == n_edges

        edge_conf = 1 - 1 / conf
        return edge_conf

    @torch.no_grad()
    def marginalize_keyframe(self, keyframe: int, use_fp16: bool = False) -> dict | None:
        neighbors = self._incident_neighbors(keyframe)
        if neighbors.numel() == 0:
            return None

        motion_tokens = self._motion_tokens_from_keyframe(keyframe, neighbors)
        depth = self.buffer.depths_sens_normed[keyframe, 0].float()
        mask = self.buffer.non_sky_masks[keyframe, 0]
        intrinsics = self.buffer.intrinsics[0]

        output = self.net.refine_from_motion_tokens(
            motion_tokens,
            depth,
            mask,
            intrinsics,
            use_fp16=use_fp16,
        )

        refined_depth = output["depth"].float()
        edge_scale = self.buffer.depths_sens_scale[keyframe, 0]
        if mask.any():
            moge_mean = depth[mask].mean()
            refined_mean = refined_depth[mask].mean()
            edge_scale = edge_scale * moge_mean / refined_mean
        edge_scales = edge_scale.expand(neighbors.shape[0])
        self.buffer.depths[keyframe, 0] = refined_depth.to(dtype=self.buffer.depths.dtype)
        self.buffer.depths_sens_scale[keyframe, 0] = edge_scale.to(dtype=self.buffer.depths_sens_scale.dtype)

        ii = torch.full_like(neighbors, keyframe)
        pose = output["pose_enc"].float()
        pose[:, 3:7] = pose[:, 3:7] / pose[:, 3:7].norm(dim=-1, keepdim=True).clamp_min(1e-6)
        conf = output["pose_confidence"].float()
        conf = self._canonicalize_edge_conf(conf, int(ii.numel()))

        self.rm_factors(self.ii == keyframe)

        return {
            "ii": ii,
            "jj": neighbors,
            "relative_pose": pose,
            "relative_scale": edge_scales,
            "confidence": conf,
        }


class FactorGraph(InitializationFactorGraph):
    def __init__(
        self,
        net: GeoNTWrapper,
        buffer: GraphBuffer,
        device: torch.device,
        max_factors: int,
    ):
        super().__init__(net, buffer, device)
        self.max_factors = max_factors
        self.edges = PoseGraphEdges(device)

        ht = buffer.height // 16
        wd = buffer.width // 16
        motion_dim = self.net.gnt.motion_patch_embed.motion_embed.embed_dim
        self.age = torch.as_tensor([], dtype=torch.long, device=device)
        self.embed = torch.zeros([0, ht, wd, motion_dim], device=device, dtype=torch.float)

    def add_factors(self, ii, jj):
        """add edges to factor graph"""

        ii, jj = self._filter_repeated_edges(ii, jj)
        if ii.numel() == 0:
            return

        if self.max_factors > 0 and ii.shape[0] > self.max_factors:
            ii = ii[: self.max_factors]
            jj = jj[: self.max_factors]

        if self.max_factors > 0 and self.ii.shape[0] + ii.shape[0] > self.max_factors:
            remove_count = self.ii.shape[0] + ii.shape[0] - self.max_factors
            order = torch.argsort(self.age, descending=True)
            mask = torch.zeros_like(self.age, dtype=torch.bool)
            mask[order[:remove_count]] = True
            self.rm_factors(mask)

        embed = self.patch_embed(ii, jj)
        self.ii = torch.cat([self.ii, ii], 0)
        self.jj = torch.cat([self.jj, jj], 0)
        self.age = torch.cat([self.age, torch.zeros_like(ii)], 0)
        self.embed = torch.cat([self.embed, embed], 0)

    def add_proximity_factors(
        self,
        t0: int,
        t1: int,
        thresh: float = 16.0,
    ):
        ii, jj = torch.meshgrid(
            torch.arange(t0, t1, device=self.device),
            torch.arange(t0, t1, device=self.device),
            indexing="ij",
        )
        ii = ii.reshape(-1).long()
        jj = jj.reshape(-1).long()

        keep = (ii - jj).abs() > 0
        ii, jj = ii[keep], jj[keep]
        ii, jj = self._filter_repeated_edges(ii, jj)
        if ii.numel() == 0:
            return

        distance = self.projection_distance(ii, jj)
        keep = distance <= float(thresh)
        if not keep.any():
            return

        ii, jj, distance = ii[keep], jj[keep], distance[keep]
        order = torch.argsort(distance)
        if self.max_factors > 0:
            order = order[: self.max_factors]
        self.add_factors(ii[order], jj[order])

    def rm_factors(self, mask: torch.Tensor):
        """drop edges from factor graph"""

        self.ii = self.ii[~mask]
        self.jj = self.jj[~mask]
        self.age = self.age[~mask]
        self.embed = self.embed[~mask]

    def increment_age(self):
        if self.age.numel() > 0:
            self.age += 1

    def oldest_active_keyframe(self) -> int | None:
        return self.oldest_keyframe()

    def marginalize_keyframe(self, keyframe: int, use_fp16: bool = False) -> dict | None:
        result = super().marginalize_keyframe(keyframe, use_fp16=use_fp16)
        if result is None:
            return None
        result["n_finalized"] = self.edges.add(
            result["ii"],
            result["jj"],
            result["relative_pose"],
            result["relative_scale"],
            result["confidence"],
        )
        return result

    def optimize_finalized_pose_graph(
        self,
        anchor: int = 0,
        n_iters: int = 12,
        damping: float = 1e-3,
        lm_max_attempts: int = DEFAULT_LM_MAX_ATTEMPTS,
        huber_delta: float = 1.0,
        scale_conf: float = 0.01,
        mode: str = "staged",
        backend: str = "cuda_eigen",
    ):
        if self.edges.ii.numel() == 0:
            return None
        n_frames = self.buffer.n_frames
        initial_poses = self.buffer.poses[:n_frames].clone()
        initial_log_scales = torch.log(self.buffer.depths_sens_scale[:n_frames, 0].clamp_min(1e-6))
        self.edges.pgo_replay = make_pgo_replay_graph(
            n_nodes=n_frames,
            anchor=anchor,
            ii=self.edges.ii,
            jj=self.edges.jj,
            relative_pose=self.edges.relative_pose,
            relative_scale=self.edges.relative_scale,
            confidence=self.edges.confidence,
            initial_poses=initial_poses,
            initial_log_scales=initial_log_scales,
        )
        result = optimize_sim3_pose_graph(
            n_nodes=n_frames,
            ii=self.edges.ii,
            jj=self.edges.jj,
            rel_poses=self.edges.relative_pose,
            rel_scales=self.edges.relative_scale,
            edge_conf=self.edges.confidence,
            initial_poses=initial_poses,
            initial_log_scales=initial_log_scales,
            anchor=anchor,
            n_iters=n_iters,
            damping=damping,
            lm_max_attempts=lm_max_attempts,
            huber_delta=huber_delta,
            scale_conf=scale_conf,
            mode=mode,
            backend=backend,
        )
        self.buffer.poses[: self.buffer.n_frames] = result.poses.to(dtype=self.buffer.poses.dtype)
        new_scales = torch.exp(result.log_scales).to(dtype=self.buffer.depths_sens_scale.dtype)
        self.buffer.depths_sens_scale[: self.buffer.n_frames, 0] = new_scales
        self.edges.pgo_info = result.info
        return result

    def _motion_tokens_from_keyframe(self, keyframe: int, neighbors: torch.Tensor) -> torch.Tensor:
        tokens = []
        for neighbor in neighbors.tolist():
            mask = (self.ii == keyframe) & (self.jj == neighbor)
            token = self.embed[torch.nonzero(mask, as_tuple=False)[0, 0]]
            tokens.append(token)
        return torch.stack(tokens, dim=0)
