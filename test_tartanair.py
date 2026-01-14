import sys
import os
import argparse
import datetime
import json
import numpy as np
import time
import math
from pathlib import Path
from typing import Sized

import cv2
import numpy as np
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

# network
from GeoNT.models.GeoNT.model import GeoNTWrapper
from GeoNT.data_readers import get_data_loader
from GeoNT.data_readers.factory import dataset_factory
from GeoNT.losses import MultitaskLoss

from GeoNT.geom.graph_utils import build_frame_graph

import GeoNT.utils.misc as misc
from GeoNT.utils.misc import NativeScalerWithGradNormCount as NativeScaler  # noqa

from alignment import align
from GeoNT.geom.losses import geodesic_loss, flow_loss
from lietorch import SE3
import matplotlib


def colorize_depth(depth: np.ndarray, mask: np.ndarray = None, normalize: bool = True, cmap: str = 'Spectral') -> np.ndarray:
    if mask is None:
        depth = np.where(depth > 0, depth, np.nan)
    else:
        depth = np.where((depth > 0) & mask, depth, np.nan)
    disp = 1 / depth
    if normalize:
        min_disp, max_disp = np.nanquantile(disp, 0.001), np.nanquantile(disp, 0.99)
        disp = (disp - min_disp) / (max_disp - min_disp)
    colored = np.nan_to_num(matplotlib.colormaps[cmap](1.0 - disp)[..., :3], 0)
    colored = np.ascontiguousarray((colored.clip(0, 1) * 255).astype(np.uint8))
    return colored


def get_args_parser():
    parser = argparse.ArgumentParser('GeoNT Training', add_help=False)
    # model
    parser.add_argument('--pretrained', default=None, help='path of a starting checkpoint')
    parser.add_argument('--fmin', type=float, default=8.0)
    parser.add_argument('--fmax', type=float, default=96.0)
    
    # training
    parser.add_argument('--seed', default=0, type=int, help="Random seed")
    parser.add_argument('--batch_size', default=1, type=int, 
                        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus)")
    parser.add_argument('--accum_iter', default=1, type=int,
                        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)")
    parser.add_argument('--epochs', default=800, type=int, help="Maximum number of epochs for the scheduler")
    parser.add_argument('--weight_decay', type=float, default=0.05, help="weight decay (default: 0.05)")
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 8')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N', help='epochs to warmup LR')
    parser.add_argument('--amp', type=int, default=0,
                        choices=[0, 1], help="Use Automatic Mixed Precision for pretraining")
    parser.add_argument("--disable_cudnn_benchmark", action='store_true', default=False,
                        help="set cudnn.benchmark = False")
    parser.add_argument('--resume', default=None, type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--n_frames', type=int, default=7)
    parser.add_argument('--edges', type=int, default=24)
    parser.add_argument('--datapath', default='datasets/TartanAir', help="path to dataset directory")
    parser.add_argument('--w_pose', type=float, default=5.0)
    parser.add_argument('--w_flow', type=float, default=0.05)
    parser.add_argument('--w_depth', type=float, default=1.0)
    parser.add_argument('--depth_valid_range', type=float, default=0.98)

    # others
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--save_freq', default=1, type=int,
                        help='frequence (number of epochs) to save checkpoint in checkpoint-last.pth')
    parser.add_argument('--keep_freq', default=5, type=int,
                        help='frequence (number of epochs) to save checkpoint in checkpoint-%d.pth')
    parser.add_argument('--print_freq', default=20, type=int,
                        help='frequence (number of iterations) to print infos while training')

    # output dir
    parser.add_argument('--output_dir', default='./output/', type=str, help="path where to save the output")
    return parser


def load_model(args, device):
    # model
    model = GeoNTWrapper()
    model.to(device)
    if args.pretrained and not args.resume:
        print('Loading pretrained: ', args.pretrained)
        ckpt = torch.load(args.pretrained, map_location=device)
        print(model.load_state_dict(ckpt, strict=False))
        del ckpt  # in case it occupies memory
    
    return model.eval()


def evaluate(predictions, batch):
    pr_depth = predictions["depth"].flatten(0, 1)
    gt_depth = batch["depth"].flatten(0, 1)
    mono_pr_depth = predictions["mono_depth"].flatten(0, 1)
    valid_mask = torch.logical_and(predictions["valid"], batch["valid"])
    valid_mask_flatten = valid_mask.flatten(0, 1)

    B = len(pr_depth)
    scale = torch.ones((B,), dtype=torch.float32, device=pr_depth.device)
    mono_scale = scale.clone()
    for i in range(B):
        pr_depth_masked = pr_depth[i][valid_mask_flatten[i]]
        mono_pr_depth_masked = mono_pr_depth[i][valid_mask_flatten[i]]
        gt_depth_masked = gt_depth[i][valid_mask_flatten[i]]
        w = 1.0 / gt_depth_masked
        scale[i], _, _ = align(pr_depth_masked, gt_depth_masked, w)
        mono_scale[i], _, _ = align(mono_pr_depth_masked, gt_depth_masked, w)

    pr_depth = pr_depth.view(*predictions["depth"].shape)
    gt_depth = gt_depth.view(*batch["depth"].shape)
    scale = scale.view(*gt_depth.shape[:2])
    mono_scale = mono_scale.view(*gt_depth.shape[:2])

    pr_rel_poses = SE3(predictions["pose_graph"])
    graph = batch["graph"]
    gt_pose = SE3(batch["poses"]).inv()  # convert poses w2c -> c2w

    intrinsics = batch["intrinsics"]

    cam_loss, geo_metrics = geodesic_loss(gt_pose, pr_rel_poses, graph, torch.ones_like(scale), 1.0 / scale)
    # cam_loss, geo_metrics = self.compute_camera_loss(gt_pose, pr_rel_poses, graph, pr_scale, gt_scale)
    flo_loss, flo_metrics = flow_loss(gt_pose, 1.0 / gt_depth, pr_rel_poses, 1.0 / pr_depth, intrinsics, graph, valid=valid_mask)
    
    pr_depth = pr_depth * scale[..., None, None]
    mono_pr_depth = mono_pr_depth * mono_scale[..., None, None]

    absrel = torch.abs(pr_depth - gt_depth) / gt_depth
    absrel = torch.mean(absrel[valid_mask])

    mono_absrel = torch.abs(mono_pr_depth - gt_depth) / gt_depth
    mono_absrel = torch.mean(mono_absrel[valid_mask])

    return absrel, mono_absrel, geo_metrics, flo_metrics


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()

    print("output_dir: " + args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # auto resume if not specified
    if args.resume is None:
        last_ckpt_fname = os.path.join(args.output_dir, f'checkpoint-last.pth')
        args.resume = last_ckpt_fname if os.path.isfile(last_ckpt_fname) else None

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # fix the seed
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = not args.disable_cudnn_benchmark

    model = load_model(args, device)
    
    # fetch dataloader
    db = dataset_factory(['tartan'], datapath=args.datapath, n_frames=args.n_frames, fmin=args.fmin, fmax=args.fmax)
    data_loader = get_data_loader(db, batch_size=args.batch_size, num_workers=args.num_workers, pin_mem=True, shuffle=True, drop_last=True)
    print("train dataset length: ", len(data_loader))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 8
    print("base lr: %.2e" % (args.lr * 8 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    def write_log_stats(epoch, train_stats):
        if misc.is_main_process():
            log_stats = dict(epoch=epoch, **{f'train_{k}': v for k, v in train_stats.items()})
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    print("Resume checkpoint %s" % args.resume)
    model.load_state_dict(checkpoint['model'], strict=False)
    del checkpoint
    torch.cuda.empty_cache()

    metric_logger = misc.MetricLogger(delimiter="  ")

    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(0)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(0)

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq)):
        # poses w2c
        images, poses, depths, depths_valid, intrinsics = [x.to(device) for x in batch]

        graph = build_frame_graph(poses, 1.0 / depths, intrinsics, num=args.edges)

        with torch.no_grad():
            prediction = model(images, intrinsics, graph, depths, depths_valid, poses, use_fp16=bool(args.amp))

        with torch.cuda.amp.autocast(enabled=False):
            absrel, mono_absrel, geo_metrics, flo_metrics = evaluate(
                prediction,
                {
                    "poses": poses,
                    "depth": depths,
                    "valid": depths_valid,
                    "intrinsics": intrinsics,
                    "graph": graph,
                }
            )

        if data_iter_step % 20 == 0:
            rgb = images[0, 0]
            depth_pr = prediction["depth"][0, 0]
            mono_depth_pr = prediction["mono_depth"][0, 0]
            depth_gt = depths[0, 0]
            mask = depths_valid[0, 0]
            depth_pr_vis = colorize_depth(depth_pr.cpu().numpy(), mask.cpu().numpy())
            mono_depth_pr_vis = colorize_depth(mono_depth_pr.cpu().numpy(), mask.cpu().numpy())
            depth_gt_vis = colorize_depth(depth_gt.cpu().numpy(), mask.cpu().numpy())
            vis1 = np.concatenate((rgb.permute(1, 2, 0)[:, :, [2,1,0]].cpu().numpy(), depth_gt_vis), axis=1)
            vis2 = np.concatenate((mono_depth_pr_vis, depth_pr_vis), axis=1)
            vis = np.concatenate((vis1, vis2), axis=0)
            cv2.imwrite(f"{data_iter_step}.png", vis[:, :, [2,1,0]])    
        
        metric_logger.update(absrel=absrel, mono_absrel=mono_absrel, **geo_metrics, **flo_metrics)

        del prediction, depths, images, depths_valid
        torch.cuda.empty_cache()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)