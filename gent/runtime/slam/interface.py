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

from dataclasses import dataclass

import numpy as np
import torch

from lietorch import SE3


@dataclass(kw_only=True)
class SLAMMap:
    # (M, 3) tensor of XYZ coordinates
    dense_depth_xyz: torch.Tensor
    # (M, 3) tensor of RGB colors (0-1)
    dense_depth_rgb: torch.Tensor
    # (N, V, 2) range of corresponding keyframe and view indices
    dense_depth_packinfo: torch.Tensor
    # Actual frame indices of the dense_depth_xyz (assert sorted)
    dense_depth_frame_inds: list[int]
    # (Q, 2) keyframe graphs (index into dense_disp_frame_inds)
    backend_graph: torch.Tensor | None = None


@dataclass(kw_only=True)
class SLAMOutput:
    trajectory: SE3  # (N,)
    intrinsics: torch.Tensor  # (N, 4), keyframe-aligned

    frame_intrinsics: torch.Tensor | None = None  # (F, 4), frame-aligned
    log_scales: torch.Tensor | None = None
    moge_log_scales: torch.Tensor | None = None
    depths: torch.Tensor | None = None  # (N, 1, H, W), keyframe-aligned
    depth_masks: torch.Tensor | None = None  # (N, 1, H, W), keyframe-aligned
    depth_status: torch.Tensor | None = None  # 0 empty, 1 mono, 2 refined
    depth_dirty: torch.Tensor | None = None  # keyframe-aligned dirty depth flags
    pose_edges: dict[str, torch.Tensor] | None = None
    pgo_info: dict | None = None
    slam_map: SLAMMap | None = None
    timestamps: np.ndarray | None = None
    frame_trajectory: SE3 | None = None
    frame_timestamps: np.ndarray | None = None

    @property
    def scales(self) -> torch.Tensor | None:
        if self.log_scales is None:
            return None
        return torch.exp(self.log_scales)

    @property
    def moge_scales(self) -> torch.Tensor | None:
        if self.moge_log_scales is None:
            return None
        return torch.exp(self.moge_log_scales)

    @property
    def keyframe_ids(self) -> np.ndarray:
        if self.timestamps is not None:
            return self.timestamps
        assert self.slam_map is not None, "SLAM map not available."
        return np.array(self.slam_map.dense_depth_frame_inds)

    def get_trajectory(self, n_frames: int | None = None):
        if self.frame_trajectory is not None:
            n = self.frame_trajectory.data.shape[0]
            if n_frames is not None:
                assert int(n_frames) == n
            return [self.frame_trajectory[i] for i in range(n)]

        trajectory = self.trajectory
        n = trajectory.data.shape[0]
        if n == 0:
            return []
        keyframe_poses = [trajectory[i] for i in range(n)]
        if n_frames is None or self.timestamps is None:
            return keyframe_poses

        out = []
        ts = list(map(int, self.timestamps))
        k = 0
        for frame_idx in range(n_frames):
            while k + 1 < len(ts) and ts[k + 1] <= frame_idx:
                k += 1
            out.append(keyframe_poses[k])
        return out
