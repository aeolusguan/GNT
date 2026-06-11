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


def _frontend_outlier_info(trans_conf_thresh: float) -> dict:
    return {
        "frontend_outlier_trans_conf_thresh": float(trans_conf_thresh),
        "frontend_outlier_candidates": 0,
        "frontend_outlier_accepted": 0,
        "frontend_outlier_dropped": 0,
        "frontend_outlier_duplicate_filtered_after_gate": 0,
    }


def _filter_frontend_edges_by_confidence(
    ii: torch.Tensor,
    jj: torch.Tensor,
    pose: torch.Tensor,
    scale: torch.Tensor,
    confidence: torch.Tensor,
    trans_conf_thresh: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    info = _frontend_outlier_info(trans_conf_thresh)
    info["frontend_outlier_candidates"] = int(ii.numel())
    if ii.numel() == 0:
        return ii, jj, pose, scale, confidence, info

    trans_conf = confidence[:, 0]
    keep = trans_conf >= float(trans_conf_thresh)
    counts = torch.stack((keep.sum(), (~keep).sum())).cpu().tolist()
    info["frontend_outlier_accepted"] = int(counts[0])
    info["frontend_outlier_dropped"] = int(counts[1])

    return ii[keep], jj[keep], pose[keep], scale[keep], confidence[keep], info


class PoseGraphEdges:
    def __init__(self, device: torch.device):
        self.ii = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj = torch.as_tensor([], dtype=torch.long, device=device)
        self.relative_pose = torch.zeros([0, 7], device=device, dtype=torch.float)
        self.relative_scale = torch.zeros([0], device=device, dtype=torch.float)
        self.confidence = torch.zeros([0, 2], device=device, dtype=torch.float)
        self.pgo_info = {}
        self.pgo_replay = {}
        self._edge_set: set[tuple[int, int]] = set()

    def _sync_edge_set(self) -> set[tuple[int, int]]:
        if len(self._edge_set) != int(self.ii.numel()):
            self._edge_set = {
                (int(i), int(j)) for i, j in zip(self.ii.cpu().tolist(), self.jj.cpu().tolist())
            }
        return self._edge_set

    def add(
        self,
        ii: torch.Tensor,
        jj: torch.Tensor,
        pose: torch.Tensor,
        scale: torch.Tensor,
        confidence: torch.Tensor,
    ) -> int:
        existing = self._sync_edge_set()
        ii_cpu = ii.cpu().tolist()
        jj_cpu = jj.cpu().tolist()
        keep = []
        for idx, (i, j) in enumerate(zip(ii_cpu, jj_cpu)):
            directed_pair = (int(i), int(j))
            if directed_pair not in existing:
                keep.append(idx)
        if not keep:
            return 0
        keep = torch.as_tensor(keep, device=ii.device, dtype=torch.long)
        self.ii = torch.cat((self.ii, ii[keep].long()), dim=0)
        self.jj = torch.cat((self.jj, jj[keep].long()), dim=0)
        self.relative_pose = torch.cat((self.relative_pose, pose[keep].float()), dim=0)
        self.relative_scale = torch.cat((self.relative_scale, scale[keep].float()), dim=0)
        self.confidence = torch.cat((self.confidence, confidence[keep].float()), dim=0)
        self._edge_set.update(
            (int(i), int(j)) for i, j in zip(ii[keep].cpu().tolist(), jj[keep].cpu().tolist())
        )
        return int(keep.numel())

    def add_from(self, other) -> int:
        added = self.add(other.ii, other.jj, other.relative_pose, other.relative_scale, other.confidence)
        self.pgo_info = dict(other.pgo_info)
        self.pgo_replay = dict(other.pgo_replay)
        return added


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
            source_mask = source_masks[:, ii[start:end]].float()
            valid = valid.squeeze(-1) * source_mask
            valid_count = valid.sum(dim=(-1, -2))
            source_count = source_mask.sum(dim=(-1, -2))
            distance = (flow_mag * valid).sum(dim=(-1, -2)) / valid_count.clamp_min(1.0)
            valid_ratio = valid_count / source_count.clamp_min(1.0)
            distance = torch.where(valid_ratio > 0.5, distance, torch.full_like(distance, torch.inf))
            distances.append(distance.squeeze(0))
        return torch.cat(distances, dim=0)

    def _filter_repeated_edges(self, ii, jj):
        """remove duplicate edges"""

        eset = {(int(i), int(j)) for i, j in zip(self.ii.cpu().tolist(), self.jj.cpu().tolist())}
        ii_cpu = ii.cpu().tolist()
        jj_cpu = jj.cpu().tolist()

        keep = [(int(i), int(j)) not in eset for i, j in zip(ii_cpu, jj_cpu)]
        keep = torch.as_tensor(keep, dtype=torch.bool, device=ii.device)

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
        edge_conf[:, 1] = 1.0
        return edge_conf

    @torch.no_grad()
    def marginalize_keyframe(
        self,
        keyframe: int,
        use_fp16: bool = False,
        decode_depth: bool = True,
    ) -> dict | None:
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
            decode_depth=decode_depth,
        )

        edge_scale = self.buffer.depths_sens_scale[keyframe, 0]
        result = {}
        if decode_depth:
            refined_depth = output["depth"].float()
            result["refined_depth"] = refined_depth
            result["source_scale"] = edge_scale
        if decode_depth and mask.any():
            moge_mean = depth[mask].mean()
            refined_mean = refined_depth[mask].mean()
            edge_scale = edge_scale * moge_mean / refined_mean
            result["source_scale"] = edge_scale
        edge_scales = edge_scale.expand(neighbors.shape[0])

        ii = torch.full_like(neighbors, keyframe)
        pose = output["pose_enc"].float()
        pose[:, 3:7] = pose[:, 3:7] / pose[:, 3:7].norm(dim=-1, keepdim=True).clamp_min(1e-6)
        conf = output["pose_confidence"].float()
        conf = self._canonicalize_edge_conf(conf, int(ii.numel()))

        self.rm_factors(self.ii == keyframe)

        result.update(
            {
                "ii": ii,
                "jj": neighbors,
                "relative_pose": pose,
                "relative_scale": edge_scales,
                "confidence": conf,
            }
        )
        return result


class FactorGraph(InitializationFactorGraph):
    def __init__(
        self,
        net: GeoNTWrapper,
        buffer: GraphBuffer,
        device: torch.device,
        max_factors: int,
        frontend_outlier_trans_conf_thresh: float,
    ):
        super().__init__(net, buffer, device)
        self.max_factors = max_factors
        self.frontend_outlier_trans_conf_thresh = float(frontend_outlier_trans_conf_thresh)
        self.frontend_outlier_info = _frontend_outlier_info(self.frontend_outlier_trans_conf_thresh)
        self.edges = PoseGraphEdges(device)

        ht = buffer.height // 16
        wd = buffer.width // 16
        motion_dim = self.net.gnt.motion_patch_embed.motion_embed.embed_dim
        self.embed = torch.zeros([0, ht, wd, motion_dim], device=device, dtype=torch.float)
        self._active_edge_set: set[tuple[int, int]] = set()

    def _frontend_outlier_public_info(self) -> dict:
        return dict(self.frontend_outlier_info)

    def _update_frontend_outlier_info(self, info: dict, duplicate_filtered_after_gate: int):
        for key, value in info.items():
            if key in self.frontend_outlier_info and isinstance(value, int):
                self.frontend_outlier_info[key] += int(value)
            else:
                self.frontend_outlier_info[key] = value
        self.frontend_outlier_info["frontend_outlier_duplicate_filtered_after_gate"] += int(
            duplicate_filtered_after_gate
        )
        self.edges.pgo_info.update(self._frontend_outlier_public_info())

    def _sync_active_edge_set(self) -> set[tuple[int, int]]:
        if len(self._active_edge_set) != int(self.ii.numel()):
            self._active_edge_set = {
                (int(i), int(j)) for i, j in zip(self.ii.cpu().tolist(), self.jj.cpu().tolist())
            }
        return self._active_edge_set

    def add_factors(self, ii, jj):
        """add edges to factor graph"""

        ii, jj = self._filter_new_edges(ii, jj)
        if ii.numel() == 0:
            return

        embed = self.patch_embed(ii, jj)
        self.ii = torch.cat([self.ii, ii], 0)
        self.jj = torch.cat([self.jj, jj], 0)
        self.embed = torch.cat([self.embed, embed], 0)
        self._active_edge_set.update(
            (int(i), int(j)) for i, j in zip(ii.cpu().tolist(), jj.cpu().tolist())
        )

    def _filter_new_edges(self, ii, jj):
        if ii.numel() == 0:
            return ii, jj

        existing = set(self._sync_active_edge_set())
        existing.update(self.edges._sync_edge_set())
        ii_cpu = ii.cpu().tolist()
        jj_cpu = jj.cpu().tolist()
        keep = []
        for idx, (i, j) in enumerate(zip(ii_cpu, jj_cpu)):
            directed_pair = (int(i), int(j))
            if directed_pair not in existing:
                keep.append(idx)

        if not keep:
            empty = torch.as_tensor([], dtype=torch.long, device=ii.device)
            return empty, empty

        keep = torch.as_tensor(keep, dtype=torch.long, device=ii.device)
        return ii[keep], jj[keep]

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
                    if abs(i - selected_i) <= nms and abs(j - selected_j) <= nms:
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

    def add_proximity_factors(
        self,
        source_start: int,
        target_start: int,
        end: int,
        radius: int,
        nms: int,
        thresh: float = 16.0,
    ) -> int:
        before = int(self.ii.numel())
        sources = torch.arange(source_start, end, device=self.device)
        targets = torch.arange(target_start, end, device=self.device)
        if sources.numel() == 0 or targets.numel() == 0:
            return 0

        ii, jj = torch.meshgrid(
            sources,
            targets,
            indexing="ij",
        )
        ii = ii.reshape(-1).long()
        jj = jj.reshape(-1).long()

        local = (ii > jj) & ((ii - jj) <= int(radius))
        local_ii, local_jj = self._bidirectional_pairs(ii[local], jj[local])
        self.add_factors(local_ii, local_jj)

        proximity = (ii - jj) > int(radius)
        cand_ii, cand_jj = self._filter_new_edges(ii[proximity], jj[proximity])
        if cand_ii.numel() == 0:
            return int(self.ii.numel()) - before

        distance = self.projection_distance(cand_ii, cand_jj)
        keep = distance <= float(thresh)
        if not keep.any():
            return int(self.ii.numel()) - before

        cand_ii, cand_jj, distance = cand_ii[keep], cand_jj[keep], distance[keep]
        selected_ii, selected_jj = self._select_proximity_nms(
            cand_ii,
            cand_jj,
            distance,
            nms=int(nms),
            max_pairs=-1,
        )
        selected_ii, selected_jj = self._bidirectional_pairs(selected_ii, selected_jj)
        self.add_factors(selected_ii, selected_jj)
        return int(self.ii.numel()) - before

    def rm_factors(self, mask: torch.Tensor):
        """drop edges from factor graph"""

        self.ii = self.ii[~mask]
        self.jj = self.jj[~mask]
        self.embed = self.embed[~mask]
        self._active_edge_set = {
            (int(i), int(j)) for i, j in zip(self.ii.cpu().tolist(), self.jj.cpu().tolist())
        }

    def oldest_active_keyframe(self) -> int | None:
        return self.oldest_keyframe()

    def marginalize_keyframe(self, keyframe: int, use_fp16: bool = False) -> dict | None:
        result = super().marginalize_keyframe(keyframe, use_fp16=use_fp16, decode_depth=False)
        if result is None:
            return None
        ii, jj, pose, scale, confidence, outlier_info = _filter_frontend_edges_by_confidence(
            result["ii"],
            result["jj"],
            result["relative_pose"],
            result["relative_scale"],
            result["confidence"],
            trans_conf_thresh=self.frontend_outlier_trans_conf_thresh,
        )
        result["n_finalized"] = self.edges.add(
            ii,
            jj,
            pose,
            scale,
            confidence,
        )
        self._update_frontend_outlier_info(
            outlier_info,
            duplicate_filtered_after_gate=int(ii.numel()) - int(result["n_finalized"]),
        )
        result["ii"] = ii
        result["jj"] = jj
        result["relative_pose"] = pose
        result["relative_scale"] = scale
        result["confidence"] = confidence
        result["frontend_outlier_info"] = dict(outlier_info)
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
        n_frames = self.buffer.n_frames
        initial_poses = self.buffer.poses[:n_frames].clone()
        initial_log_scales = torch.log(self.buffer.depths_sens_scale[:n_frames, 0].clamp_min(1e-6))
        return self._optimize_finalized_pose_graph_with_initial_state(
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
            pgo_info={
                "scope": "frontend_full_graph",
                "global_anchor": int(anchor),
            },
        )

    def _optimize_finalized_pose_graph_with_initial_state(
        self,
        initial_poses: torch.Tensor,
        initial_log_scales: torch.Tensor,
        anchor: int,
        n_iters: int,
        damping: float,
        lm_max_attempts: int,
        huber_delta: float,
        scale_conf: float,
        mode: str,
        backend: str,
        pgo_info: dict,
    ):
        if self.edges.ii.numel() == 0:
            return None
        n_frames = self.buffer.n_frames
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
        result.info = dict(result.info)
        result.info.update(pgo_info)
        result.info.update(self._frontend_outlier_public_info())
        self.edges.pgo_info = result.info
        return result

    def optimize_finalized_pose_graph_window(
        self,
        window_start: int,
        window_end: int,
        n_iters: int = 12,
        damping: float = 1e-3,
        lm_max_attempts: int = DEFAULT_LM_MAX_ATTEMPTS,
        huber_delta: float = 1.0,
        scale_conf: float = 0.01,
        mode: str = "staged",
        backend: str = "cuda_eigen",
    ):
        assert 0 <= window_start < window_end <= self.buffer.n_frames
        if self.edges.ii.numel() == 0:
            return None

        in_window = (
            (self.edges.ii >= int(window_start))
            & (self.edges.ii < int(window_end))
            & (self.edges.jj >= int(window_start))
            & (self.edges.jj < int(window_end))
        )
        if not in_window.any():
            return None

        window_size = int(window_end) - int(window_start)
        local_ii = self.edges.ii[in_window] - int(window_start)
        local_jj = self.edges.jj[in_window] - int(window_start)
        relative_pose = self.edges.relative_pose[in_window]
        relative_scale = self.edges.relative_scale[in_window]
        confidence = self.edges.confidence[in_window]
        initial_poses = self.buffer.poses[window_start:window_end].clone()
        initial_log_scales = torch.log(
            self.buffer.depths_sens_scale[window_start:window_end, 0].clamp_min(1e-6)
        )
        old_window_tail_pose = self.buffer.poses[window_end - 1].clone()

        self.edges.pgo_replay = make_pgo_replay_graph(
            n_nodes=window_size,
            anchor=0,
            ii=local_ii,
            jj=local_jj,
            relative_pose=relative_pose,
            relative_scale=relative_scale,
            confidence=confidence,
            initial_poses=initial_poses,
            initial_log_scales=initial_log_scales,
        )
        result = optimize_sim3_pose_graph(
            n_nodes=window_size,
            ii=local_ii,
            jj=local_jj,
            rel_poses=relative_pose,
            rel_scales=relative_scale,
            edge_conf=confidence,
            initial_poses=initial_poses,
            initial_log_scales=initial_log_scales,
            anchor=0,
            n_iters=n_iters,
            damping=damping,
            lm_max_attempts=lm_max_attempts,
            huber_delta=huber_delta,
            scale_conf=scale_conf,
            mode=mode,
            backend=backend,
        )
        self.buffer.poses[window_start:window_end] = result.poses.to(dtype=self.buffer.poses.dtype)
        self._propagate_window_suffix_poses(window_end, old_window_tail_pose)
        new_scales = torch.exp(result.log_scales).to(dtype=self.buffer.depths_sens_scale.dtype)
        self.buffer.depths_sens_scale[window_start:window_end, 0] = new_scales
        pgo_info = dict(result.info)
        pgo_info.update(
            {
                "scope": "frontend_window",
                "window_start": int(window_start),
                "window_end": int(window_end),
                "global_anchor": int(window_start),
            }
        )
        pgo_info.update(self._frontend_outlier_public_info())
        result.info = pgo_info
        self.edges.pgo_info = pgo_info
        return result

    def _propagate_window_suffix_poses(self, window_end: int, old_window_tail_pose: torch.Tensor):
        if window_end >= self.buffer.n_frames:
            return

        suffix = SE3(self.buffer.poses[window_end : self.buffer.n_frames].float())
        old_tail = SE3(old_window_tail_pose.float().view(1, 7))
        new_tail = SE3(self.buffer.poses[window_end - 1].float().view(1, 7))
        suffix_from_tail = suffix * old_tail.inv()
        self.buffer.poses[window_end : self.buffer.n_frames] = (suffix_from_tail * new_tail).data.to(
            dtype=self.buffer.poses.dtype
        )

    def _motion_tokens_from_keyframe(self, keyframe: int, neighbors: torch.Tensor) -> torch.Tensor:
        return self.embed[self.ii == keyframe]
