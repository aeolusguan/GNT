import os
import os.path as osp
import pickle
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch

from .augmentation import RGBDAugmentor
from .easy_dataset import EasyDataset
from .rgbd_utils import compute_distance_matrix_flow


DATASET_CACHE_DIR = osp.join(osp.dirname(osp.abspath(__file__)), "cache")


class RGBDDataset(EasyDataset):
    z_far = 0

    def __init__(
        self,
        name,
        datapath,
        n_frames=4,
        crop_size=(384, 512),
        fmin=8.0,
        fmax=75.0,
        do_aug=True,
    ):
        """Base class for RGB-D training datasets."""
        self.aug = None
        self.root = datapath
        self.name = name

        self.n_frames = n_frames
        self.fmin = fmin
        self.fmax = fmax

        if do_aug:
            self.aug = RGBDAugmentor(crop_size=crop_size)

        os.makedirs(DATASET_CACHE_DIR, exist_ok=True)
        cache_path = osp.join(DATASET_CACHE_DIR, f"{self.name}.pickle")

        if osp.isfile(cache_path):
            with open(cache_path, "rb") as cache_file:
                scene_info = pickle.load(cache_file)[0]
        else:
            scene_info = self._build_dataset()
            with open(cache_path, "wb") as cache_file:
                pickle.dump((scene_info,), cache_file)

        self.scene_info = scene_info
        self._build_dataset_index()

    @staticmethod
    def is_test_scene(scene):
        return False

    def _build_dataset_index(self):
        self.dataset_index = []
        for scene, scene_info in self.scene_info.items():
            if self.is_test_scene(scene):
                print(f"Reserving {scene} for validation")
                continue

            for source, (neighbors, distance) in scene_info["graph"].items():
                eligible = (distance > self.fmin) & (distance < self.fmax)
                if len(neighbors) > self.n_frames and eligible.any():
                    self.dataset_index.append((scene, source))

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    @staticmethod
    def depth_read(depth_file):
        depth = np.load(depth_file)
        valid = np.ones_like(depth, dtype=bool)
        return depth, valid

    @classmethod
    def build_frame_graph(
        cls,
        poses,
        depths,
        intrinsics,
    ):
        """Compute the symmetric projection-flow graph of an ordered trajectory."""
        downsample = 16

        def read_disp(depth_file):
            depth, valid = cls.depth_read(depth_file)
            offset = downsample // 2
            depth = np.asarray(depth)[offset::downsample, offset::downsample].copy()
            valid = np.asarray(valid)[offset::downsample, offset::downsample].copy()
            valid &= np.isfinite(depth) & (depth >= 0.01)
            if cls.z_far > 0:
                valid &= depth < cls.z_far

            disparity = np.zeros_like(depth)
            disparity[valid] = 1.0 / depth[valid]
            return disparity, valid

        poses = np.array(poses)
        intrinsics = np.array(intrinsics) / downsample

        with ThreadPoolExecutor(max_workers=min(8, len(depths))) as executor:
            disps, valid = zip(*executor.map(read_disp, depths))
        distance = downsample * compute_distance_matrix_flow(
            poses,
            np.stack(disps),
            intrinsics,
            valid=np.stack(valid),
        )

        graph = {}
        for source in range(distance.shape[0]):
            neighbors = np.flatnonzero(np.isfinite(distance[source]))
            graph[source] = (neighbors, distance[source, neighbors])

        return graph

    def __getitem__(self, index):
        """Return one sampled training video."""

        index = index % len(self.dataset_index)
        scene_id, ix = self.dataset_index[index]

        scene_info = self.scene_info[scene_id]
        frame_graph = scene_info["graph"]

        inds = [ix]
        while len(inds) < self.n_frames:
            k = (frame_graph[ix][1] > self.fmin) & (frame_graph[ix][1] < self.fmax)
            frames = frame_graph[ix][0][k]
            forward_frames = frames[frames > ix]
            ix = np.random.choice(forward_frames if len(forward_frames) else frames)
            inds.append(ix)

        images, depths, depths_valid, poses, intrinsics = [], [], [], [], []
        for i in inds:
            images.append(self.image_read(scene_info["images"][i]))
            depth, depth_valid = self.depth_read(scene_info["depths"][i])
            depths.append(depth)
            if self.z_far > 0:
                depth_valid = depth_valid & (depth < self.z_far)
            depths_valid.append(depth_valid)
            poses.append(scene_info["poses"][i])
            intrinsics.append(scene_info["intrinsics"][i])

        images = np.stack(images).astype(np.float32)
        depths = np.stack(depths).astype(np.float32)
        depths_valid = np.stack(depths_valid).astype(np.bool_)
        poses = np.stack(poses).astype(np.float32)
        intrinsics = np.stack(intrinsics).astype(np.float32)

        images = torch.from_numpy(images).float()
        images = images.permute(0, 3, 1, 2)

        depths = torch.from_numpy(depths)
        depths_valid = torch.from_numpy(depths_valid)
        poses = torch.from_numpy(poses)
        intrinsics = torch.from_numpy(intrinsics)

        if self.aug is not None:
            images, poses, depths, depths_valid, intrinsics = self.aug(
                images,
                poses,
                depths,
                depths_valid,
                intrinsics,
            )

        return images, poses, depths, depths_valid, intrinsics

    def __len__(self):
        return len(self.dataset_index)
