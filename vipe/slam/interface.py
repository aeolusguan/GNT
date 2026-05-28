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
    intrinsics: torch.Tensor  # (V, 4)

    slam_map: SLAMMap | None = None

    @property
    def keyframe_ids(self) -> np.ndarray:
        assert self.slam_map is not None, "SLAM map not available."
        return np.array(self.slam_map.dense_depth_frame_inds)