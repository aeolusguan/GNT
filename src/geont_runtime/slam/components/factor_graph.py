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

from .buffer import GraphBuffer
from geont_runtime.slam.pgo import DEFAULT_LM_MAX_ATTEMPTS, optimize_sim3_pose_graph
from geont_runtime.slam.pgo.replay import make_pgo_replay_graph

warnings.simplefilter(action="ignore", category=FutureWarning)


class PoseGraphEdges:
    def __init__(self, device: torch.device):
        self.ii = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj = torch.as_tensor([], dtype=torch.long, device=device)
        self.relative_pose = torch.zeros([0, 7], device=device, dtype=torch.float)
        self.relative_scale = torch.zeros([0], device=device, dtype=torch.float)
        self.confidence = torch.zeros([0, 2], device=device, dtype=torch.float)
        self.depth_observability_score = torch.zeros([0], device=device, dtype=torch.float)
        self.depth_observability_rank = torch.zeros([0], device=device, dtype=torch.uint8)
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
        observability_score: torch.Tensor,
        observability_rank: torch.Tensor,
    ) -> torch.Tensor:
        assert ii.shape == jj.shape
        assert observability_score.shape == ii.shape
        assert observability_rank.shape == ii.shape
        assert (observability_rank > 0).all()
        assert torch.isfinite(observability_score).all()
        existing = set(self._sync_edge_set())
        keep = []
        for idx, (i, j) in enumerate(zip(ii.cpu().tolist(), jj.cpu().tolist())):
            directed_pair = (int(i), int(j))
            if directed_pair in existing:
                continue
            keep.append(idx)
            existing.add(directed_pair)
        if not keep:
            return torch.zeros(ii.shape[0], device=ii.device, dtype=torch.bool)

        keep_idx = torch.as_tensor(keep, device=ii.device, dtype=torch.long)
        keep_mask = torch.zeros(ii.shape[0], device=ii.device, dtype=torch.bool)
        keep_mask[keep_idx] = True
        self.ii = torch.cat((self.ii, ii[keep_idx].long()), dim=0)
        self.jj = torch.cat((self.jj, jj[keep_idx].long()), dim=0)
        self.relative_pose = torch.cat((self.relative_pose, pose[keep_idx].float()), dim=0)
        self.relative_scale = torch.cat((self.relative_scale, scale[keep_idx].float()), dim=0)
        self.confidence = torch.cat((self.confidence, confidence[keep_idx].float()), dim=0)
        self.depth_observability_score = torch.cat(
            (self.depth_observability_score, observability_score[keep_idx].float()),
            dim=0,
        )
        self.depth_observability_rank = torch.cat(
            (self.depth_observability_rank, observability_rank[keep_idx].to(dtype=torch.uint8)),
            dim=0,
        )
        self._edge_set = existing
        return keep_mask

    def add_from(self, other) -> int:
        keep = self.add(
            other.ii,
            other.jj,
            other.relative_pose,
            other.relative_scale,
            other.confidence,
            other.depth_observability_score,
            other.depth_observability_rank,
        )
        self.pgo_info = dict(other.pgo_info)
        self.pgo_replay = dict(other.pgo_replay)
        return int(keep.sum().item())

    def update_depth_observability(
        self,
        ii: torch.Tensor,
        jj: torch.Tensor,
        observability_logits: torch.Tensor,
        *,
        rank: int,
    ) -> None:
        assert ii.numel() == jj.numel()
        assert observability_logits.shape[0] == int(ii.numel())
        scores = observability_logits.float().flatten(1).mean(dim=1).to(device=self.depth_observability_score.device)
        assert torch.isfinite(scores).all()
        edge_to_index = {
            (int(i), int(j)): idx
            for idx, (i, j) in enumerate(zip(self.ii.cpu().tolist(), self.jj.cpu().tolist()))
        }
        edge_indices = torch.as_tensor(
            [edge_to_index[(int(i), int(j))] for i, j in zip(ii.cpu().tolist(), jj.cpu().tolist())],
            dtype=torch.long,
            device=self.depth_observability_score.device,
        )
        if int(rank) == 1:
            update = self.depth_observability_rank[edge_indices] == 0
            edge_indices = edge_indices[update]
            scores = scores[update]
        if edge_indices.numel() == 0:
            return
        self.depth_observability_score[edge_indices] = scores
        self.depth_observability_rank[edge_indices] = int(rank)

    def select_depth_observability_topk(
        self,
        source: int,
        neighbors: torch.Tensor,
        topk: int | None,
    ) -> torch.Tensor:
        if topk is None or int(topk) >= int(neighbors.numel()):
            return neighbors

        edge_to_index = {(int(i), int(j)): idx for idx, (i, j) in enumerate(zip(self.ii.cpu().tolist(), self.jj.cpu().tolist()))}
        edge_indices = torch.as_tensor(
            [edge_to_index[(int(source), int(neighbor))] for neighbor in neighbors.cpu().tolist()],
            dtype=torch.long,
            device=neighbors.device,
        )
        ranks = self.depth_observability_rank[edge_indices]
        if (ranks == 0).any():
            missing = neighbors[ranks == 0].cpu().tolist()
            raise RuntimeError(f"missing depth observability cache for directed edges from {source}: {missing[:8]}")

        score_tensor = self.depth_observability_score[edge_indices].to(device=neighbors.device)
        topk_indices = torch.topk(score_tensor, min(int(topk), int(neighbors.numel())), dim=0).indices
        return neighbors[topk_indices]


class FactorGraph:
    def __init__(
        self,
        buffer: GraphBuffer,
        device: torch.device,
    ):
        self.buffer = buffer
        self.device = device
        self.edges = PoseGraphEdges(device)

    def projection_distance(self, ii: torch.Tensor, jj: torch.Tensor, stride: int = 8) -> torch.Tensor:
        if ii.numel() == 0:
            return torch.empty_like(ii, dtype=torch.float)

        from . import geom_cuda

        n_frames = self.buffer.n_frames
        return geom_cuda.projection_distance(
            self.buffer.poses[:n_frames].float(),
            self.buffer.depths[:n_frames, 0].float(),
            self.buffer.depths_sens_scale[:n_frames, 0].float(),
            self.buffer.non_sky_masks[:n_frames, 0],
            self.buffer.intrinsics[0].float() / float(stride),
            ii,
            jj,
            stride,
        )

    def _filter_new_edges(self, ii: torch.Tensor, jj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        existing = set(self.edges._sync_edge_set())
        keep = []
        for idx, (i, j) in enumerate(zip(ii.cpu().tolist(), jj.cpu().tolist())):
            directed_pair = (int(i), int(j))
            if directed_pair in existing:
                continue
            keep.append(idx)
            existing.add(directed_pair)
        if not keep:
            empty = torch.as_tensor([], dtype=torch.long, device=ii.device)
            return empty, empty
        keep = torch.as_tensor(keep, dtype=torch.long, device=ii.device)
        return ii[keep], jj[keep]

    def add_measurement_result(self, result: dict) -> dict:
        n_candidates = int(result["ii"].numel())
        observability_logits = result["depth_observability_logits"]
        assert observability_logits.shape[0] == n_candidates
        observability_score = observability_logits.float().flatten(1).mean(dim=1)
        observability_rank = torch.full(
            result["ii"].shape,
            int(result["depth_observability_rank"]),
            device=result["ii"].device,
            dtype=torch.uint8,
        )
        keep = self.edges.add(
            result["ii"],
            result["jj"],
            result["relative_pose"],
            self.buffer.depths_sens_scale[result["ii"], 0].float(),
            result["confidence"],
            observability_score,
            observability_rank,
        )
        n_added = int(keep.sum().item())
        added_ii = result["ii"][keep] if n_added > 0 else torch.as_tensor([], dtype=torch.long, device=self.device)
        out = dict(result)
        out["n_added"] = n_added
        out["n_duplicates"] = n_candidates - n_added
        out["changed_sources"] = torch.unique(added_ii) if n_added > 0 else added_ii
        return out

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

    def select_proximity_edges(
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
        selected_ii_chunks: list[torch.Tensor] = []
        selected_jj_chunks: list[torch.Tensor] = []

        local = (ii > jj) & ((ii - jj) <= int(radius))
        local_ii, local_jj = self._bidirectional_pairs(ii[local], jj[local])
        local_ii, local_jj = self._filter_new_edges(local_ii, local_jj)
        if local_ii.numel() > 0:
            selected_ii_chunks.append(local_ii)
            selected_jj_chunks.append(local_jj)

        proximity = (ii - jj) > int(radius)
        cand_ii, cand_jj = self._filter_new_edges(ii[proximity], jj[proximity])
        if cand_ii.numel() > 0:
            distance = self.projection_distance(cand_ii, cand_jj)
            keep = distance <= float(thresh)
            if keep.any():
                cand_ii, cand_jj, distance = cand_ii[keep], cand_jj[keep], distance[keep]
                prox_ii, prox_jj = self._select_proximity_nms(
                    cand_ii,
                    cand_jj,
                    distance,
                    nms=int(nms),
                    max_pairs=-1,
                )
                prox_ii, prox_jj = self._bidirectional_pairs(prox_ii, prox_jj)
                prox_ii, prox_jj = self._filter_new_edges(prox_ii, prox_jj)
                if prox_ii.numel() > 0:
                    selected_ii_chunks.append(prox_ii)
                    selected_jj_chunks.append(prox_jj)

        if not selected_ii_chunks:
            return empty, empty
        return torch.cat(selected_ii_chunks), torch.cat(selected_jj_chunks)

    def edge_neighbors(self, source: int, *, window_start: int | None = None, window_end: int | None = None) -> torch.Tensor:
        mask = self.edges.ii == int(source)
        if window_start is not None:
            mask &= self.edges.jj >= int(window_start)
        if window_end is not None:
            mask &= self.edges.jj < int(window_end)
        return self.edges.jj[mask]

    def optimize_pose_graph(
        self,
        anchor: int = 0,
        n_iters: int = 12,
        damping: float = 1e-3,
        lm_max_attempts: int = DEFAULT_LM_MAX_ATTEMPTS,
        huber_delta: float = 1.0,
        scale_conf: float = 0.01,
        mode: str = "staged",
        backend: str = "cuda_eigen",
        extra_info: dict | None = None,
    ):
        n_frames = self.buffer.n_frames
        initial_poses = self.buffer.poses[:n_frames].clone()
        initial_log_scales = torch.log(self.buffer.depths_sens_scale[:n_frames, 0].clamp_min(1e-6))
        pgo_info = dict(self.edges.pgo_info)
        pgo_info.update({"scope": "full_graph", "global_anchor": int(anchor)})
        if extra_info is not None:
            pgo_info.update(extra_info)
        return self._optimize_pose_graph_with_initial_state(
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
            pgo_info=pgo_info,
        )

    def _optimize_pose_graph_with_initial_state(
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
        self.edges.pgo_info = result.info
        return result

    def optimize_pose_graph_window(
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
