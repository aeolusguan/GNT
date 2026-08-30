from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from .base import RGBDDataset


class DynamicReplica(RGBDDataset):
    """Dynamic Replica sequences in the CUT3R processed layout."""

    def __init__(self, **kwargs):
        super().__init__(name="DynamicReplica", **kwargs)

    @staticmethod
    def scene_names(root):
        split_root = Path(root) / "train"
        return sorted(path.name for path in split_root.iterdir() if path.is_dir())

    @classmethod
    def build_scene(cls, root, scene):
        scene_root = Path(root) / "train" / scene
        images = sorted(str(path) for path in (scene_root / "images").glob("*.png"))
        depths = sorted(str(path) for path in (scene_root / "depths").glob("*.png"))
        camera_files = sorted((scene_root / "cameras").glob("*.npz"))

        camera_to_world = []
        intrinsics = []
        for camera_file in camera_files:
            with np.load(camera_file) as camera:
                camera_to_world.append(camera["pose"])
                K = camera["intrinsics"]
                intrinsics.append([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])

        camera_to_world = np.stack(camera_to_world)
        poses = np.concatenate(
            (
                camera_to_world[:, :3, 3],
                Rotation.from_matrix(camera_to_world[:, :3, :3]).as_quat(),
            ),
            axis=1,
        )
        intrinsics = np.asarray(intrinsics)
        graph = cls.build_frame_graph(poses, depths, intrinsics)
        return {
            "images": images,
            "depths": depths,
            "poses": poses,
            "intrinsics": intrinsics,
            "graph": graph,
        }

    def _build_dataset(self):
        print("Building Dynamic Replica dataset")
        scene_info = {}
        for scene in tqdm(self.scene_names(self.root)):
            scene_info[scene] = self.build_scene(self.root, scene)
        return scene_info

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file, cv2.IMREAD_COLOR)

    @staticmethod
    def depth_read(depth_file):
        encoded = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
        depth = encoded.view(np.float16).astype(np.float32)
        valid = np.isfinite(depth) & (depth > 0)
        return depth, valid
