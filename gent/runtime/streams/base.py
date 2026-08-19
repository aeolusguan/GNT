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

import copy
import importlib
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Iterator, Protocol, cast

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import  IterableDataset

from lietorch import SE3
from gent.runtime.utils.logging import pbar

logger = logging.getLogger(__name__)


class FrameAttribute(Enum):
    POSE = "pose"
    INTRINSICS = "intrinsics"
    MASK = "mask"
    SKY_MASK = "sky_mask"
    METRIC_DEPTH = "metric_depth"


@dataclass(kw_only=True, slots=True)
class VideoFrame:
    """
    Frame data from a single video frame.
    - raw_frame_idx: The index of the frame in the video.
    - rgb: The RGB image of the frame. The shape is (H, W, 3), RGB, with range 0-1.
    - pose: The pose of the camera at the time the frame was captured (c2w, aka. Twc, opencv convention).
    - intrinsics: Pinhole intrinsics torch Tensor of shape (4,), [fx, fy, cx, cy].
    - mask: Binary mask of the frame. The shape is (H, W), with 0 for invalid pixels.
    - sky_mask: Binary mask for the sky. The shape is (H, W), with 1 for sky pixels.
    - metric_depth: The depth map of the frame. THe shape is (H, W). Value is in metric scale.
    - information: Additional information about the frame
    """

    raw_frame_idx: int
    rgb: torch.Tensor
    pose: SE3 | None = None
    intrinsics: torch.Tensor | None = None
    mask: torch.Tensor | None = None
    sky_mask: torch.Tensor | None = None
    metric_depth: torch.Tensor | None = None
    information: str = ""

    def size(self) -> tuple[int, int]:
        return (self.rgb.shape[0], self.rgb.shape[1])
    
    @property
    def device(self) -> torch.device:
        return self.rgb.device
    
    def attributes(self) -> set[FrameAttribute]:
        attributes = set()
        if self.pose is not None:
            attributes.add(FrameAttribute.POSE)
        if self.intrinsics is not None:
            attributes.add(FrameAttribute.INTRINSICS)
        if self.mask is not None:
            attributes.add(FrameAttribute.MASK)
        if self.sky_mask is not None:
            attributes.add(FrameAttribute.SKY_MASK)
        if self.metric_depth is not None:
            attributes.add(FrameAttribute.METRIC_DEPTH)

        return attributes

    def get_attribute(self, attribute: FrameAttribute) -> Any:
        if attribute == FrameAttribute.POSE:
            return self.pose
        if attribute == FrameAttribute.INTRINSICS:
            return self.intrinsics
        if attribute == FrameAttribute.MASK:
            return self.mask
        if attribute == FrameAttribute.SKY_MASK:
            return self.sky_mask
        if attribute == FrameAttribute.METRIC_DEPTH:
            return self.metric_depth
        raise ValueError(f"Attribute {attribute} is not available in the frame.")
    
    def set_attribute(self, attribute: FrameAttribute, value: Any) -> None:
        if attribute == FrameAttribute.POSE:
            self.pose = value
        elif attribute == FrameAttribute.INTRINSICS:
            self.intrinsics = value
        elif attribute == FrameAttribute.MASK:
            self.mask = value
        elif attribute == FrameAttribute.SKY_MASK:
            self.sky_mask = value
        elif attribute == FrameAttribute.METRIC_DEPTH:
            self.metric_depth = value
        else:
            raise ValueError(f"Attribute {attribute} is not available in the frame.")
        
    def cpu(self) -> "VideoFrame":
        def map_cpu(x):
            return x.cpu() if x is not None else None
        
        return VideoFrame(
            raw_frame_idx=self.raw_frame_idx,
            rgb=self.rgb.cpu(),
            mask=map_cpu(self.mask),
            sky_mask=map_cpu(self.sky_mask),
            metric_depth=map_cpu(self.metric_depth),
            pose=map_cpu(self.pose),
            intrinsics=map_cpu(self.intrinsics),
            information=self.information,
        )
    
    def cuda(self) -> "VideoFrame":
        def map_cuda(x):
            return x.cuda() if x is not None and torch.cuda.is_available() else x
        
        return VideoFrame(
            raw_frame_idx=self.raw_frame_idx,
            rgb=map_cuda(self.rgb),
            mask=map_cuda(self.mask),
            sky_mask=map_cuda(self.sky_mask),
            metric_depth=map_cuda(self.metric_depth),
            pose=map_cuda(self.pose),
            intrinsics=map_cuda(self.intrinsics),
            information=self.information,
        )
    
    def resize(self, size: tuple[int, int]) -> "VideoFrame":
        """
        Resize the frame to a given size.
        """
        h0, w0 = self.size()
        h1, w1 = size

        new_rgb = (
            torch.nn.functional.interpolate(self.rgb.permute(2, 0, 1)[None], size, mode="bilinear")
            .squeeze(0)
            .permute(1, 2, 0)
        )

        new_mask = None
        if self.mask is not None:
            new_mask = torch.nn.functional.interpolate(self.mask[None, None].float(), size, mode="bilinear")[0, 0] > 0.9

        new_sky_mask = None
        if self.sky_mask is not None:
            new_sky_mask = torch.nn.functional.interpolate(self.sky_mask[None, None].float(), size, mode="bilinear")[0, 0] > 0.9

        new_metric_depth = None
        if self.metric_depth is not None:
            new_metric_depth = torch.nn.functional.interpolate(self.metric_depth[None, None], size, mode="bilinear")[
                0, 0
            ]

        new_intrinsics = None
        if self.intrinsics is not None:
            new_intrinsics = self.intrinsics.clone()
            new_intrinsics[0:4:2] *= w1 / w0
            new_intrinsics[1:4:2] *= h1 / h0

        return VideoFrame(
            raw_frame_idx=self.raw_frame_idx,
            rgb=new_rgb,
            mask=new_mask,
            sky_mask=new_sky_mask,
            metric_depth=new_metric_depth,
            pose=self.pose,
            intrinsics=new_intrinsics,
            information=self.information,
        )
    
    def crop(self, top: int, bottom: int, left: int, right: int) -> "VideoFrame":
        """
        Crop the frame with given top, bottom, left, right.
        """
        bottom = self.size()[0] - bottom
        right = self.size()[1] - right

        new_rgb = self.rgb[top:bottom, left:right]

        new_mask = None
        if self.mask is not None:
            new_mask = self.mask[top:bottom, left:right]

        new_sky_mask = None
        if self.sky_mask is not None:
            new_sky_mask = self.sky_mask[top:bottom, left:right]

        new_metric_depth = None
        if self.metric_depth is not None:
            new_metric_depth = self.metric_depth[top:bottom, left:right]

        new_intrinsics = None
        if self.intrinsics is not None:
            new_intrinsics = self.intrinsics.clone()
            new_intrinsics[2] -= left
            new_intrinsics[3] -= top

        return VideoFrame(
            raw_frame_idx=self.raw_frame_idx,
            rgb=new_rgb,
            mask=new_mask,
            sky_mask=new_sky_mask,
            metric_depth=new_metric_depth,
            pose=self.pose,
            intrinsics=new_intrinsics,
            information=self.information,
        )
    

class VideoStream(IterableDataset[VideoFrame]):
    """
    Base class for video streams.
    """

    def frame_size(self) -> tuple[int, int]:
        raise NotImplementedError
    
    def name(self) -> str:
        raise NotImplementedError
    
    def fps(self) -> int:
        raise NotImplementedError
    
    def __len__(self) -> int:
        raise NotImplementedError
    
    def attributes(self) -> set[FrameAttribute]:
        return set()
    
    def get_stream_attribute(self, attribute: FrameAttribute) -> list[Any]:
        stream_attribute = []
        for frame in self:
            stream_attribute.append(frame.get_attribute(attribute))
        return stream_attribute
    
    def get_gt_stream_attribute(self, attribute: FrameAttribute) -> list[Any]:
        raise NotImplementedError(f"{type(self).__name__} does not provide ground truth {attribute.value} data.")
    

class CachedVideoStream(VideoStream):
    """
    Cache a video stream.
    """

    DISPLAY_THRESH = 20

    def __init__(self, video_stream: VideoStream, desc: str = "Caching") -> None:
        self._frame_size = video_stream.frame_size()
        self._fps = video_stream.fps()
        self._name = video_stream.name()
        self._attributes = video_stream.attributes()
        self._len = len(video_stream)
        self.iterator: Iterator[VideoFrame] | None = iter(video_stream)
        self.data: list[VideoFrame] = []
        self.desc = desc

    def fps(self) -> float:
        return self._fps
    
    def frame_size(self) -> tuple[int, int]:
        return self._frame_size
    
    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
        return self._len
    
    def __getitem__(self, index) -> VideoFrame:
        assert index < len(self)
        n_iters_needed = index - len(self.data) + 1
        if n_iters_needed <= 0:
            return self.data[index].cuda()
        
        itr = range(n_iters_needed)
        if n_iters_needed > self.DISPLAY_THRESH:
            itr = pbar(itr, total=n_iters_needed, desc=self.desc)

        for _ in itr:
            assert self.iterator is not None
            try:
                self.data.append(next(self.iterator).cpu())
            except StopIteration:
                logger.warning(
                    "Iterator is exhausted -- expecting total frames = %d, stopped at %d",
                    len(self),
                    len(self.data),
                )
                self._len = len(self.data)
                index = min(index, self._len - 1)
                break
        
        # If iteration is finished, we can release the iterator
        if len(self.data) == len(self):
            self.iterator = None
            torch.cuda.empty_cache()
        
        return self.data[index].cuda()

    def __iter__(self):
        for idx in range(len(self)):
            # Since len(self) might change during iteration, we check again here:
            if idx >= len(self):
                break

            yield self[idx]

    def attributes(self) -> set[FrameAttribute]:
        return self._attributes
    

class StreamProcessor(Protocol):
    """
    Interface of a stream processor that processes each video frame.
    """

    n_passes_required: int = 1

    def update_fps(self, previous_fps: float) -> float:
        return previous_fps
    
    def update_frame_size(self, previous_frame_size: tuple[int, int]):
        return previous_frame_size

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes

    def update_iterator(self, previous_iterator: Iterator[VideoFrame], pass_idx: int) -> Iterator[VideoFrame]:
        for frame_idx, frame in enumerate(previous_iterator):
            yield self(frame_idx, frame)

    def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame: ...


class AssignAttributesProcessor(StreamProcessor):
    def __init__(self, stream_attributes: dict[FrameAttribute, list[Any]]):
        self.stream_attributes = stream_attributes

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes.union(self.stream_attributes.keys())

    def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame:
        for attribute, attribute_values in self.stream_attributes.items():
            frame.set_attribute(attribute, attribute_values[frame_idx])
        return frame
    

class ProcessedVideoStream(VideoStream):
    """
    A video stream from a raw video stream, with processing applied.
    """

    def __init__(self, stream: VideoStream, processors: list[StreamProcessor]) -> None:
        super().__init__()
        self.stream = stream
        self.processors = processors
        self.n_passes_required = max(processor.n_passes_required for processor in processors) if processors else 1

    def frame_size(self) -> tuple[int, int]:
        frame_size = self.stream.frame_size()
        for processor in self.processors:
            frame_size = processor.update_frame_size(frame_size)
        return frame_size

    def fps(self) -> float:
        fps = self.stream.fps()
        for processor in self.processors:
            fps = processor.update_fps(fps)
        return fps

    def attributes(self) -> set[FrameAttribute]:
        attributes = self.stream.attributes()
        for processor in self.processors:
            attributes = processor.update_attributes(attributes)
        return attributes

    def name(self) -> str:
        return self.stream.name()

    def cache(self, desc: str = "Caching", online: bool = False) -> CachedVideoStream:
        vs = CachedVideoStream(self, desc)

        # If not online, we trigger __getitem__ of the last element to force storing all frames.
        if not online:
            _ = vs[len(vs) - 1]

        return vs

    def _build_iterator(self, pass_idx: int) -> Iterator[VideoFrame]:
        iterator = iter(self.stream)
        for processor in self.processors:
            iterator = processor.update_iterator(iterator, pass_idx)
        return iterator

    def __len__(self) -> int:
        return len(self.stream)

    def __iter__(self):
        for pass_idx in range(self.n_passes_required):
            iterator = self._build_iterator(pass_idx)
            # Iterate through the processors to update internal state.
            if pass_idx != self.n_passes_required - 1:
                for _ in pbar(iterator, desc=f"Pre-iterating for pass {pass_idx}"):
                    pass
        return iterator
    

class StreamList:
    @staticmethod
    def make(config: DictConfig) -> "StreamList":
        module_path, class_name = config.instance.rsplit(".", 1)
        module = importlib.import_module(module_path)
        config = copy.deepcopy(config)
        del config.instance
        return getattr(module, class_name)(**cast(dict[str, Any], config))
    
    def __len__(self) -> int:
        raise NotImplementedError
    
    def __getitem__(self, index) -> VideoStream:
        raise NotImplementedError
    
    def stream_name(self, index: int) -> str:
        # This can be overriden by subclasses to avoid instantiating the stream.
        return self[index].name()
