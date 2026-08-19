import glob
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# Define the merged dataset metadata dictionary
dataset_metadata = {
    "kitti": {
        "img_path": "data/kitti/depth_selection/val_selection_cropped/image_gathered",  # Default path
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq),
        "gt_traj_func": lambda img_path, anno_path, seq: None,
        "traj_format": None,
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": lambda args, img_path: process_kitti(args, img_path),
        "intrinsics_path": "data/kitti/depth_selection/val_selection_cropped",
        "intrinsics_func": lambda calib_root, sequence, frames: load_kitti_intrinsics(
            calib_root, sequence, frames
        ),
    },
    "bonn": {
        "img_path": "data/bonn/rgbd_bonn_dataset",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(
            img_path, seq if seq.startswith("rgbd_bonn_") else f"rgbd_bonn_{seq}", "rgb_110"
        ),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            img_path, seq if seq.startswith("rgbd_bonn_") else f"rgbd_bonn_{seq}", "groundtruth_110.txt"
        ),
        "traj_format": "tum",
        "seq_list": ["balloon2", "crowd2", "crowd3", "person_tracking2", "synchronous"],
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": lambda args, img_path: process_bonn(args, img_path),
        "intrinsics_path": None,
        "intrinsics_func": lambda calib_root, sequence, frames: load_bonn_intrinsics(
            calib_root, sequence, frames
        ),
    },
    "sintel": {
        "img_path": "data/sintel/training/final",
        "anno_path": "data/sintel/training/camdata_left",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(anno_path, seq),
        "traj_format": None,
        "seq_list": [
            "alley_2",
            "ambush_4",
            "ambush_5",
            "ambush_6",
            "cave_2",
            "cave_4",
            "market_2",
            "market_5",
            "market_6",
            "shaman_3",
            "sleeping_1",
            "sleeping_2",
            "temple_2",
            "temple_3",
        ],
        "full_seq": False,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": lambda args, img_path: process_sintel(args, img_path),
        "intrinsics_path": "data/sintel/training/camdata_left",
        "intrinsics_func": lambda calib_root, sequence, frames: load_sintel_intrinsics(
            calib_root, sequence, frames
        ),
    }
}


# Define processing functions for each dataset
def process_kitti(args, img_path):
    for dir in tqdm(sorted(glob.glob(f"{img_path}/*"))):
        filelist = sorted(glob.glob(f"{dir}/*.png"))
        save_dir = f"{args.output_dir}/{os.path.basename(dir)}"
        yield filelist, save_dir


def process_bonn(args, img_path):
    if args.full_seq:
        for dir in tqdm(sorted(glob.glob(f"{img_path}/*/"))):
            filelist = sorted(glob.glob(f"{dir}/rgb/*.png"))
            save_dir = f"{args.output_dir}/{os.path.basename(os.path.dirname(dir))}"
            yield filelist, save_dir
    else:
        seq_list = (
            ["balloon2", "crowd2", "crowd3", "person_tracking2", "synchronous"]
            if args.seq_list is None
            else args.seq_list
        )
        for seq in tqdm(seq_list):
            filelist = sorted(glob.glob(f"{img_path}/rgbd_bonn_{seq}/rgb_110/*.png"))
            save_dir = f"{args.output_dir}/{seq}"
            yield filelist, save_dir


def process_sintel(args, img_path):
    if args.full_seq:
        for dir in tqdm(sorted(glob.glob(f"{img_path}/*/"))):
            filelist = sorted(glob.glob(f"{dir}/*.png"))
            save_dir = f"{args.output_dir}/{os.path.basename(os.path.dirname(dir))}"
            yield filelist, save_dir
    else:
        seq_list = [
            "alley_2",
            "ambush_4",
            "ambush_5",
            "ambush_6",
            "cave_2",
            "cave_4",
            "market_2",
            "market_5",
            "market_6",
            "shaman_3",
            "sleeping_1",
            "sleeping_2",
            "temple_2",
            "temple_3",
        ]
        for seq in tqdm(seq_list):
            filelist = sorted(glob.glob(f"{img_path}/{seq}/*.png"))
            save_dir = f"{args.output_dir}/{seq}"
            yield filelist, save_dir


def _intrinsics4(K):
    K = np.asarray(K, dtype=np.float32).reshape(3, 3)
    return K[[0, 1, 0, 1], [0, 1, 2, 2]]


def load_video_intrinsics(dataset, sequence, frames):
    metadata = dataset_metadata[dataset]
    return metadata["intrinsics_func"](
        metadata["intrinsics_path"],
        sequence,
        frames,
    )


def load_sintel_intrinsics(calib_root, sequence, frames):
    intrinsics = []

    for frame in frames:
        path = Path(calib_root, sequence, f"{Path(frame).stem}.cam")
        with path.open("rb") as f:
            if np.fromfile(f, np.float32, 1).item() != 202021.25:
                raise ValueError(f"Invalid sintel camera file: {path}")
            intrinsics.append(_intrinsics4(np.fromfile(f, np.float64, 9)))

    return np.stack(intrinsics)


def load_kitti_intrinsics(calib_root, sequence, frames):
    sequence_name, camera = sequence.rsplit("_", 1)
    calib_root = Path(calib_root)
    calibration_path = sorted(
        (calib_root / "intrinsics").glob(
            f"{sequence_name}_image_*_image_{camera}.txt"
        )
    )[0]
    cropped_image_path = calib_root / "image" / f"{calibration_path.stem}.png"

    intrinsics = _intrinsics4(np.loadtxt(calibration_path))
    with Image.open(frames[0]) as raw_image, Image.open(cropped_image_path) as cropped_image:
        crop_left = (raw_image.width - cropped_image.width) // 2
        crop_top = raw_image.height - cropped_image.height

    # KITTI depth-selection calibration is expressed in the horizontally
    # centered, bottom-aligned crop. `image_gathered` contains the original
    # full-resolution frames, so restore the principal point to that grid.
    intrinsics[2] += crop_left
    intrinsics[3] += crop_top
    return np.repeat(intrinsics[None], len(frames), axis=0)


def load_bonn_intrinsics(calib_root, sequence, frames):
    # The benchmark keeps the registered RGB/depth pixel grid and ignores the
    # small published RGB lens distortion.
    intrinsics = np.array(
        [542.822841, 542.576870, 315.593520, 237.756098],
        dtype=np.float32,
    )
    return np.repeat(intrinsics[None], len(frames), axis=0)
