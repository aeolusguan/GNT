import os.path as osp

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from .base import RGBDDataset


class ARKitScenes(RGBDDataset):
    """ARKitScenes training sequences in the CUT3R processed layout."""

    def __init__(self, **kwargs):
        super().__init__(name="ARKitScenes", **kwargs)

    @staticmethod
    def scene_names(root):
        metadata_path = osp.join(root, "Training", "all_metadata.npz")
        with np.load(metadata_path) as metadata:
            return [str(scene) for scene in metadata["scenes"]]

    @classmethod
    def build_scene(cls, root, scene):
        split_root = osp.join(root, "Training")
        scene = str(scene)
        scene_root = osp.join(split_root, scene)
        with np.load(osp.join(scene_root, "new_scene_metadata.npz")) as metadata:
            basenames = metadata["images"]
            trajectories = metadata["trajectories"]
            intrinsics = metadata["intrinsics"][:, 2:6]

        images = [
            osp.join(scene_root, "vga_wide", basename.replace(".png", ".jpg"))
            for basename in basenames
        ]
        depths = [
            osp.join(scene_root, "lowres_depth", basename)
            for basename in basenames
        ]
        poses = np.concatenate(
            (
                trajectories[:, :3, 3],
                Rotation.from_matrix(trajectories[:, :3, :3]).as_quat(),
            ),
            axis=1,
        )
        graph = cls.build_frame_graph(poses, depths, intrinsics)
        return {
            "images": images,
            "depths": depths,
            "poses": poses,
            "intrinsics": intrinsics,
            "graph": graph,
        }

    def _build_dataset(self):
        print("Building ARKitScenes dataset")
        scene_info = {}
        for scene in tqdm(self.scene_names(self.root)):
            scene_info[scene] = self.build_scene(self.root, scene)
        return scene_info

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    @staticmethod
    def depth_read(depth_file):
        depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth /= 1000.0
        depth[~np.isfinite(depth)] = 0.0
        valid = depth > 0.0
        return depth, valid
