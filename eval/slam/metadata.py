import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np


dataset_metadata = {
    "bonn": {
        "img_path": "data/bonn/rgbd_bonn_dataset",
        "dir_path_func": lambda img_path, seq: os.path.join(
            img_path,
            seq if seq.startswith("rgbd_bonn_") else f"rgbd_bonn_{seq}",
            "rgb_110",
        ),
        "gt_traj_func": lambda img_path, seq: os.path.join(
            img_path,
            seq if seq.startswith("rgbd_bonn_") else f"rgbd_bonn_{seq}",
            "groundtruth_110.txt",
        ),
        "traj_format": "tum",
        "seq_list": ["balloon2", "crowd2", "crowd3", "person_tracking2", "synchronous"],
        "full_seq": False,
        "intrinsics_path": None,
        "intrinsics_func": lambda _calib_path, _sequence, frames: np.repeat(
            np.array(
                [542.822841, 542.576870, 315.593520, 237.756098],
                dtype=np.float32,
            )[None],
            len(frames),
            axis=0,
        ),
        "skip_condition": None,
    },
    "tum": {
        "img_path": "data/tum",
        "dir_path_func": lambda img_path, seq: os.path.join(
            img_path, seq, "rgb_amb3r"
        ),
        "gt_traj_func": lambda img_path, seq: os.path.join(
            img_path, seq, "groundtruth_amb3r.txt"
        ),
        "traj_format": "tum",
        "seq_list": None,
        "full_seq": True,
        "intrinsics_path": None,
        "intrinsics_func": lambda _calib_path, sequence, frames: load_tum_intrinsics(
            sequence, frames
        ),
    },
}


TUM_RGB_INTRINSICS = {
    "freiburg1": (517.3, 516.5, 318.6, 255.3),
    "freiburg2": (520.9, 521.0, 325.1, 249.7),
    "freiburg3": (535.4, 539.2, 320.1, 247.6),
}
TUM_CROP_MARGINS = (24, 18)


def load_tum_intrinsics(sequence, frames):
    """Return calibrated RGB intrinsics for a TUM Freiburg sequence."""

    camera = next(name for name in TUM_RGB_INTRINSICS if name in sequence)
    fx, fy, cx, cy = TUM_RGB_INTRINSICS[camera]
    crop_x, crop_y = TUM_CROP_MARGINS
    return np.repeat(
        np.asarray([fx, fy, cx - crop_x, cy - crop_y], dtype=np.float32)[None],
        len(frames),
        axis=0,
    )


def load_slam_intrinsics(
    dataset: str,
    sequence: str,
    frames: Sequence[Path],
) -> np.ndarray:
    """Return one ``[fx, fy, cx, cy]`` row for each selected frame."""

    metadata = dataset_metadata[dataset]
    return metadata["intrinsics_func"](metadata["intrinsics_path"], sequence, frames)
