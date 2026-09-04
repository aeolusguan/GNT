import os
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.slam.metadata import (
    dataset_metadata,
    load_slam_intrinsics,
)
from eval.slam.evo_utils import load_traj
from gent.runtime.inference import build_streaming_config
from gent.runtime.slam.system import SLAMSystem
from gent.runtime.streams.frame_dir_stream import FrameDirStream
from eval.slam.utils import get_tum_poses
from eval.slam.evo_utils import eval_metrics, plot_trajectory


SLAM_OUTPUT_CONVENTION = "w2c"
SLAM_CONFIG_PATH = ROOT / "configs/slam_eval.yaml"


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="path to GeNT model weights")
    parser.add_argument(
        "--eval_dataset",
        default="bonn",
        choices=list(dataset_metadata),
        help="dataset metadata entry to evaluate",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="directory for evaluation summaries and evo plots",
    )
    parser.add_argument(
        "--full_seq",
        action="store_true",
        default=False,
        help="use full sequence for pose evaluation",
    )
    parser.add_argument(
        "--seq_list",
        nargs="+",
        default=None,
        help="list of sequences for pose evaluation",
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

    ate_mean, rpe_trans_mean, rpe_rot_mean, ate_kf_mean = eval_pose_estimation_dist(
        args, save_dir=save_dir, img_path=img_path,
    )
    return ate_mean, rpe_trans_mean, rpe_rot_mean, ate_kf_mean


def eval_pose_estimation_dist(
    args, img_path, save_dir=None,
) -> tuple[float, float, float, float]:
    """Return global means for ATE, RPE-trans, RPE-rot, and keyframe ATE."""
    dataset = args.eval_dataset
    metadata = dataset_metadata[dataset]
    img_path = Path(img_path)

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
        save_dir = Path("eval_results/slam") / args.eval_dataset
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
        )
    device = torch.device("cuda", local_rank)

    seqs = split_for_rank(seq_list, rank, world_size)
    error_log_path = save_dir / f"_error_log_{rank}.txt"

    slam_overrides = OmegaConf.load(SLAM_CONFIG_PATH)
    ate_list = []
    ate_kf_list = []
    rpe_trans_list = []
    rpe_rot_list = []

    for seq in tqdm(seqs):
        try:
            dir_path = metadata["dir_path_func"](img_path, seq)

            # Handle skip_condition
            skip_condition = metadata.get("skip_condition", None)
            if skip_condition is not None and skip_condition(str(save_dir), seq):
                continue

            frame_dir = Path(dir_path)
            stream = FrameDirStream(
                frame_dir,
                seek_range=range(0, -1, 1),
                name=seq,
            )
            filelist = stream.frame_files[stream.start : stream.end : stream.step]
            intrinsics = load_slam_intrinsics(args.eval_dataset, seq, filelist)
            slam_config = build_streaming_config(
                frame_dir,
                args.weights,
                intrinsics.tolist(),
                slam_config=slam_overrides,
            ).pipeline.slam

            system = SLAMSystem(device=device, config=slam_config)
            slam_output = system.run(stream)

            poses_pred = np.linalg.inv(slam_output.poses.cpu().numpy())
            kf_idx = slam_output.kf_idx.cpu().numpy()
            pred_traj = get_tum_poses(poses_pred)

            gt_traj_file = metadata["gt_traj_func"](img_path, seq)
            traj_format = metadata.get("traj_format", None)

            if args.eval_dataset == "sintel":
                gt_traj = load_traj(
                    gt_traj_file=gt_traj_file, stride=1
                )
            else:
                assert traj_format is not None
                gt_traj = load_traj(
                    gt_traj_file=gt_traj_file,
                    traj_format=traj_format,
                    stride=1,
                )

            ate, rpe_trans, rpe_rot = eval_metrics(
                pred_traj,
                gt_traj,
                seq=seq,
                filename=f"{save_dir}/{seq}_eval_metric.txt",
            )
            plot_trajectory(
                pred_traj, gt_traj, title=seq, filename=f"{save_dir}/{seq}.png"
            )
            ate_list.append(ate)
            rpe_trans_list.append(rpe_trans)
            rpe_rot_list.append(rpe_rot)

            # KF
            pred_kf_traj = [x[kf_idx] for x in pred_traj]
            gt_kf_traj = [x[kf_idx] for x in gt_traj]
            ate_kf, _, _ = eval_metrics(
                pred_kf_traj,
                gt_kf_traj,
                seq=seq,
                filename=f"{save_dir}/{seq}_eval_kf_metric.txt",
            )
            ate_kf_list.append(ate_kf)

            # Write to error log after each sequence
            with open(error_log_path, "a") as f:
                f.write(
                    f"{args.eval_dataset}-{seq: <16} | ATE: {ate:.5f}, ATE_KF: {ate_kf:.5f}, RPE trans: {rpe_trans:.5f}, RPE rot: {rpe_rot:.5f}\n"
                )
                f.write(f"{ate:.5f}\n")
                f.write(f"{ate_kf:.5f}\n")
                f.write(f"{rpe_trans:.5f}\n")
                f.write(f"{rpe_rot:.5f}\n")

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

    torch.distributed.barrier(device_ids=[local_rank])

    gathered_metrics = [None] * world_size
    torch.distributed.all_gather_object(
        gathered_metrics,
        (ate_list, rpe_trans_list, rpe_rot_list, ate_kf_list),
    )

    ate_values = [value for metrics in gathered_metrics for value in metrics[0]]
    rpe_trans_values = [value for metrics in gathered_metrics for value in metrics[1]]
    rpe_rot_values = [value for metrics in gathered_metrics for value in metrics[2]]
    ate_kf_values = [value for metrics in gathered_metrics for value in metrics[3]]

    def mean_or_zero(values):
        return float(np.mean(values)) if values else 0.0

    avg_ate, avg_rpe_trans, avg_rpe_rot, avg_ate_kf = mean_or_zero(ate_values), mean_or_zero(rpe_trans_values), mean_or_zero(rpe_rot_values), mean_or_zero(ate_kf_values)

    # Write the average to the error log (only on the main process)
    if rank == 0:
        with open(f"{save_dir}/_error_log.txt", "a") as f:
            # Copy the error log from each process to the main error log
            for i in range(world_size):
                if not os.path.exists(f"{save_dir}/_error_log_{i}.txt"):
                    break
                with open(f"{save_dir}/_error_log_{i}.txt", "r") as f_sub:
                    f.write(f_sub.read())
            f.write(
                f"Average ATE: {avg_ate:.5f}, Average ATE KF: {avg_ate_kf:.5f} Average RPE trans: {avg_rpe_trans:.5f}, Average RPE rot: {avg_rpe_rot:.5f}\n"
            )

    return avg_ate, avg_rpe_trans, avg_rpe_rot, avg_ate_kf


def main() -> None:
    args = get_args_parser().parse_args()
    eval_pose_estimation(args, save_dir=args.output_dir)


if __name__ == "__main__":
    main()
