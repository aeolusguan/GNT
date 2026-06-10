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
from pathlib import Path

import json
import numpy as np
import torch
from omegaconf import DictConfig

from geont_runtime.slam.pgo.replay import pgo_replay_npz_payload
from geont_runtime.slam.system import SLAMOutput, SLAMSystem
from geont_runtime.streams.base import (
    AssignAttributesProcessor,
    FrameAttribute,
    ProcessedVideoStream,
    StreamProcessor,
    VideoStream,
)
from geont_runtime.utils import io

from . import AnnotationPipelineOutput, Pipeline

logger = logging.getLogger(__name__)


class DefaultAnnotationPipeline(Pipeline):
    def __init__(self, init: DictConfig, slam: DictConfig, post: DictConfig, output: DictConfig) -> None:
        super().__init__()
        self.init_cfg = init
        self.slam_cfg = slam
        self.post_cfg = post
        self.out_cfg = output
        self.out_path = Path(self.out_cfg.path)
        self.out_path.mkdir(exist_ok=True, parents=True)
        
    def _add_post_processors(
        self, video_stream: VideoStream, slam_output: SLAMOutput
    ) -> ProcessedVideoStream:
        post_processors: list[StreamProcessor] = [
            AssignAttributesProcessor(
                {
                    FrameAttribute.POSE: slam_output.get_trajectory(len(video_stream)),  # type: ignore
                    FrameAttribute.INTRINSICS: [slam_output.intrinsics] * len(video_stream),
                }
            )
        ]
        return ProcessedVideoStream(video_stream, post_processors)

    def _save_slam_output(self, artifact_path: io.ArtifactPath, slam_output: SLAMOutput) -> None:
        artifact_path.pose_path.parent.mkdir(exist_ok=True, parents=True)

        trajectory = slam_output.trajectory.data
        if isinstance(trajectory, torch.Tensor):
            trajectory = trajectory.cpu().numpy()

        intrinsics = slam_output.intrinsics.cpu().numpy()
        log_scales = np.array([], dtype=np.float32)
        if slam_output.log_scales is not None:
            log_scales = slam_output.log_scales.cpu().numpy()
        scales = np.exp(log_scales).astype(np.float32) if log_scales.size else np.array([], dtype=np.float32)
        pgo_info = slam_output.pgo_info or {}
        finalized_edges = slam_output.finalized_edges or {}
        finalized_np = {
            f"edge_{key}": value.cpu().numpy()
            for key, value in finalized_edges.items()
        }
        replay_np = pgo_replay_npz_payload(slam_output.pgo_replay)

        np.savez_compressed(
            artifact_path.pose_path,
            trajectory=trajectory,
            intrinsics=intrinsics,
            timestamps=slam_output.keyframe_ids,
            log_scales=log_scales,
            scales=scales,
            pgo_info=np.array(json.dumps(pgo_info)),
            **finalized_np,
            **replay_np,
        )

        if slam_output.depths is not None:
            artifact_path.depth_npz_path.parent.mkdir(exist_ok=True, parents=True)
            depth_payload = {
                "depths": slam_output.depths.cpu().numpy(),
                "timestamps": slam_output.keyframe_ids,
            }
            if slam_output.depth_masks is not None:
                depth_payload["masks"] = slam_output.depth_masks.cpu().numpy()
            np.savez_compressed(artifact_path.depth_npz_path, **depth_payload)
    
    def run(self, video_data: VideoStream) -> AnnotationPipelineOutput:
        artifact_path = io.ArtifactPath(self.out_path, video_data.name())

        annotate_output = AnnotationPipelineOutput()

        if self.should_filter(video_data.name()):
            logger.info(f"{video_data.name()} has been processed already, skip it!!")
            return annotate_output

        slam_stream = ProcessedVideoStream(video_data, []).cache("process", online=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        slam_pipeline = SLAMSystem(device=device, config=self.slam_cfg)
        slam_output = slam_pipeline.run(slam_stream)
        self._save_slam_output(artifact_path, slam_output)

        if self.return_payload:
            annotate_output.payload = slam_output
            return annotate_output

        output_streams = [self._add_post_processors(slam_stream, slam_output).cache("depth", online=True)]
        if self.return_output_streams:
            annotate_output.output_streams = output_streams
        return annotate_output
