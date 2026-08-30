"""GeNT video-depth launcher following CUT3R's evaluation dispatch."""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from eval.video_depth.metadata import dataset_metadata, load_video_intrinsics
from eval.video_depth.utils import save_depth_maps
from gent.runtime.inference import build_streaming_config
from gent.runtime.slam.system import SLAMSystem
from gent.runtime.streams.frame_dir_stream import FrameDirStream


FrameDepth = tuple[torch.Tensor, torch.Tensor]


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="path to the model weights",
    )
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="sintel",
        choices=list(dataset_metadata.keys()),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="directory for saved depth maps",
    )
    parser.add_argument(
        "--pose_eval_stride",
        default=1,
        type=int,
        help="stride for video depth evaluation",
    )
    parser.add_argument(
        "--depth_normalization_min_cutoff",
        default=80.0,
        type=float,
        help="minimum runtime cutoff for non-sky MoGe depth normalization",
    )
    parser.add_argument(
        "--depth_normalization_quantile",
        default=0.8,
        type=float,
        help="fraction of non-sky MoGe depths used for runtime normalization",
    )
    parser.add_argument(
        "--full_seq",
        action="store_true",
        default=False,
        help="use the full sequence set for evaluation",
    )
    parser.add_argument(
        "--seq_list",
        nargs="+",
        default=None,
        help="list of sequences for evaluation",
    )
    return parser


def split_for_rank(items, rank: int, world_size: int):
    items_per_rank, remainder = divmod(len(items), world_size)
    start = rank * items_per_rank + min(rank, remainder)
    end = start + items_per_rank + int(rank < remainder)
    return items[start:end]


def eval_pose_estimation(args, save_dir=None):
    metadata = dataset_metadata.get(args.eval_dataset)
    img_path = metadata["img_path"]
    mask_path = metadata["mask_path"]

    ate_mean, rpe_trans_mean, rpe_rot_mean = eval_pose_estimation_dist(
        args, img_path=img_path, save_dir=save_dir, mask_path=mask_path
    )
    return ate_mean, rpe_trans_mean, rpe_rot_mean


def eval_pose_estimation_dist(args, img_path, save_dir=None, mask_path=None) -> dict[str, list[FrameDepth]]:
    metadata = dataset_metadata.get(args.eval_dataset)

    seq_list = args.seq_list
    if seq_list is None:
        if metadata.get("full_seq", False):
            args.full_seq = True
        else:
            seq_list = metadata.get("seq_list", [])
        if args.full_seq:
            seq_list = os.listdir(img_path)
            seq_list = [
                seq for seq in seq_list if os.path.isdir(os.path.join(img_path, seq))
            ]
        seq_list = sorted(seq_list)

    if save_dir is None:
        save_dir = args.output_dir
    if save_dir is None:
        save_dir = Path("eval_results/video_depth") / args.eval_dataset
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    seqs = split_for_rank(seq_list, rank, world_size)
    results: dict[str, list[FrameDepth]] = {}
    error_log_path = save_dir / f"_error_log_{rank}.txt"

    for seq in tqdm(seqs):
        try:
            dir_path = metadata["dir_path_func"](img_path, seq)

            skip_condition = metadata.get("skip_condition", None)
            if skip_condition is not None and skip_condition(str(save_dir), seq):
                continue

            frame_dir = Path(dir_path)
            stream = FrameDirStream(
                frame_dir,
                seek_range=range(0, -1, args.pose_eval_stride),
                name=seq,
            )
            filelist = stream.frame_files[stream.start : stream.end : stream.step]
            intrinsics = load_video_intrinsics(args.eval_dataset, seq, filelist)
            slam_overrides = {
                "depth_normalization_min_cutoff": args.depth_normalization_min_cutoff,
                "depth_normalization_quantile": args.depth_normalization_quantile,
            }
            slam_config = build_streaming_config(
                frame_dir,
                args.weights,
                intrinsics.tolist(),
                slam_config=slam_overrides,
            ).pipeline.slam

            system = SLAMSystem(device=device, config=slam_config)
            slam_output, frame_depths = system.run_video_depth_benchmark(stream)

            os.makedirs(f"{save_dir}/{seq}", exist_ok=True)
            save_depth_maps(
                [metric_depth for metric_depth, _valid_mask in frame_depths],
                f"{save_dir}/{seq}",
            )

        except Exception as e:
            if "out of memory" in str(e):
                # Handle OOM
                torch.cuda.empty_cache()  # Clear the CUDA memory
                with open(error_log_path, "a") as f:
                    f.write(
                        f"OOM error in sequence {seq}, skipping this sequence.\n"
                    )
                print(f"OOM error in sequence {seq}, skipping...")
            elif "Degenerate covariance rank" in str(
                e
            ) or "Eigenvalues did not converge" in str(e):
                # Handle Degenerate covariance rank exception and Eigenvalues did not converge exception
                with open(error_log_path, "a") as f:
                    f.write(f"Exception in sequence {seq}: {str(e)}\n")
                print(f"Traj evaluation error in sequence {seq}, skipping.")
            else:
                raise e  # Rethrow if it's not an expected exception
    return None, None, None


def main() -> None:
    args = get_args_parser()
    args = args.parse_args()
    if args.eval_dataset == "sintel":
        args.full_seq = True
    eval_pose_estimation(args)


if __name__ == "__main__":
    main()
