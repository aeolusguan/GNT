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

import logging
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, cast

import cv2
import imageio
import Imath
import numpy as np
import OpenEXR
import torch

import SE3
from vipe.streams.base import FrameAttribute, VideoFrame, VideoStream
from vipe.utils.cameras import CameraType
from vipe.utils.geometry import se3_matrix_to_se3
from vipe.utils.visualization import VideoWriter

logger = logging.getLogger(__name__)


@dataclass
class ArtifactPath:
    base_path: Path
    artifact_name: str

    @property
    def rgb_path(self) -> Path:
        return self.base_path / "rgb" / f"{self.artifact_name}.mp4"
    
    @property
    def pose_path(self) -> Path:
        return self.base_path / "pose" / f"{self.artifact_name}.npz"

    @property
    def depth_path(self) -> Path:
        return self.base_path / "depth" / f"{self.artifact_name}.zip"
    
    @property
    def mask_path(self) -> Path:
        return self.base_path / "mask" / f"{self.artifact_name}.zip"
    
    @property
    def sky_mask_path(self) -> Path:
        return self.base_path / "sky_mask" / f"{self.artifact_name}.zip"
    
    @property
    def meta_info_path(self) -> Path:
        return self.base_path / "vipe" / f"{self.artifact_name}_info.pkl"