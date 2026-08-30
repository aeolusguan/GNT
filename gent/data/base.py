import os
import os.path as osp
import pickle
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch

from .augmentation import RGBDAugmentor
from .easy_dataset import EasyDataset
from .rgbd_utils import compute_sparse_distance_matrix_flow


DATASET_CACHE_DIR = osp.join(osp.dirname(osp.abspath(__file__)), "cache")


class RGBDDataset(EasyDataset):
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
        """Compute a projection-flow graph from one valid point per depth patch."""
        def read_points(depth_file):
            depth, valid = cls.depth_read(depth_file)
            depth = np.asarray(depth)
            valid = np.asarray(valid)
            valid = valid & np.isfinite(depth) & (depth >= 0.01)
            return depth.shape, cls.sample_depth_points(depth, valid)

        with ThreadPoolExecutor(max_workers=min(16, len(depths))) as executor:
            depth_records = list(executor.map(read_points, depths))
        image_size = depth_records[0][0]
        assert all(shape == image_size for shape, _ in depth_records)
        records = [record for _, record in depth_records]

        point_count = max(len(record[2]) for record in records)
        coords = np.zeros((len(records), point_count, 2), dtype=np.float32)
        sampled_depths = np.zeros((len(records), point_count), dtype=np.float32)
        sampled_valid = np.zeros((len(records), point_count), dtype=bool)
        for frame, (x, y, depth) in enumerate(records):
            count = len(depth)
            coords[frame, :count, 0] = x
            coords[frame, :count, 1] = y
            sampled_depths[frame, :count] = depth
            sampled_valid[frame, :count] = True

        distance = compute_sparse_distance_matrix_flow(
            np.array(poses),
            coords,
            sampled_depths,
            sampled_valid,
            np.array(intrinsics),
        )
        max_flow = 256.0
        graph = {}
        for source in range(distance.shape[0]):
            neighbors = np.flatnonzero(distance[source] < max_flow)
            graph[source] = (neighbors, distance[source, neighbors])
        return graph

    @staticmethod
    def sample_depth_points(depth, valid):
        """Select the first valid depth sample from each 16x16 patch."""
        patch_size = 16
        height, width = depth.shape
        patch_rows = (height + patch_size - 1) // patch_size
        patch_cols = (width + patch_size - 1) // patch_size
        pad_height = patch_rows * patch_size - height
        pad_width = patch_cols * patch_size - width
        padded_valid = np.pad(
            valid,
            ((0, pad_height), (0, pad_width)),
            constant_values=False,
        )
        # Rearrange to [patch row, patch column, pixel row, pixel column].
        patch_valid = padded_valid.reshape(
            patch_rows, patch_size, patch_cols, patch_size
        ).transpose(0, 2, 1, 3)
        patch_valid = patch_valid.reshape(-1, patch_size * patch_size)

        has_valid = patch_valid.any(axis=1)
        patch_indices = np.flatnonzero(has_valid)
        point_offsets = patch_valid.argmax(axis=1)[has_valid]
        patch_y, patch_x = np.divmod(patch_indices, patch_cols)
        point_y = patch_y * patch_size + point_offsets // patch_size
        point_x = patch_x * patch_size + point_offsets % patch_size
        return (
            point_x.astype(np.float32),
            point_y.astype(np.float32),
            depth[point_y, point_x].astype(np.float32),
        )

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
