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
    parser.add_argument('--w_depth_aux', type=float, nargs='+', default=[0.1, 0.3, 0.5], 
                        help="weights for auxiliary depth losses, should be a list of length num_out_layers-1")
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
    model_without_ddp = model
    if args.pretrained and not args.resume:
        print('Loading pretrained: ', args.pretrained)
        ckpt = torch.load(args.pretrained, map_location=device)
        print(model.load_state_dict(ckpt, strict=False))
        del ckpt  # in case it occupies memory
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False, static_graph=True
        )
        model_without_ddp = model.module
    
    return model, model_without_ddp


def train(args):
    misc.init_distributed_mode(args)

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

    model, model_without_ddp = load_model(args, device)
    
    # fetch dataloader
    db = dataset_factory(['tartan'], datapath=args.datapath, n_frames=args.n_frames, fmin=args.fmin, fmax=args.fmax)
    data_loader_train = get_data_loader(db, batch_size=args.batch_size, num_workers=args.num_workers, pin_mem=True, shuffle=True, drop_last=True)
    print("train dataset length: ", len(data_loader_train))

    train_criterion = MultitaskLoss(args).to(device)

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 8
    print("base lr: %.2e" % (args.lr * 8 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # following timm: set wd as 0 for bias and norm layers
    param_groups = misc.get_parameter_groups(model_without_ddp.model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    def write_log_stats(epoch, train_stats):
        if misc.is_main_process():
            log_stats = dict(epoch=epoch, **{f'train_{k}': v for k, v in train_stats.items()})
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    def save_model(epoch, fname):
        misc.save_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch, fname=fname)
        
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    train_stats = {}
    for epoch in range(args.start_epoch, args.epochs + 1):
        already_saved = False

        # Save more stuff
        write_log_stats(epoch, train_stats)

        if epoch > args.start_epoch:
            if args.keep_freq and epoch % args.keep_freq == 0:
                save_model(epoch - 1, str(epoch))
                already_saved = True
        
        # Save immediately the last checkpoint
        if epoch > args.start_epoch:
            if args.save_freq and epoch % args.save_freq == 0 or epoch == args.epochs and not already_saved:
                save_model(epoch - 1, 'last')
        
        if epoch >= args.epochs:
            break  # exit after writing last test to disk

        # Train
        train_stats = train_one_epoch(
            model, train_criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args=args
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def save_final_model(args, epoch, model_without_ddp):
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / 'checkpoint-final.pth'
    to_save = {
        'args': args,
        'model': model_without_ddp if isinstance(model_without_ddp, dict) else model_without_ddp.cpu().state_dict(),
        'epoch': epoch,
    }
    print(f'>> Saving model to {checkpoint_path} ...')
    misc.save_on_master(to_save, checkpoint_path)

    
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Sized, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    args,
):
    assert torch.backends.cuda.matmul.allow_tf32 == True

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    accum_iter = args.accum_iter

    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)

    optimizer.zero_grad()

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        epoch_f = epoch + data_iter_step / len(data_loader)

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            misc.adjust_learning_rate(optimizer, epoch_f, args)

        # poses w2c
        images, poses, depths, depths_valid, intrinsics = [x.to(device) for x in batch]

        # randomize frame graph
        if np.random.rand() < 0.5:
            graph = build_frame_graph(poses, 1.0 / depths, intrinsics, num=args.edges)
        else:
            graph = OrderedDict()
            for i in range(args.n_frames):
                graph[i] = [j for j in range(args.n_frames) if i!=j and abs(i-j) <= 2]

        prediction = model(images, intrinsics, graph, depths, depths_valid, poses, use_fp16=bool(args.amp))

        with torch.cuda.amp.autocast(enabled=False):
            loss, geo_metrics, flo_metrics, depth_metrics = criterion(
                prediction,
                {
                    "poses": poses,
                    "depth": depths,
                    "valid": depths_valid,
                    "intrinsics": intrinsics,
                    "graph": graph,
                }
            )
        loss_value = float(loss)
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), force=True)
            sys.exit(1)
        
        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.module.model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        
        del loss
        del prediction
        del batch

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(epoch=epoch_f)
        metric_logger.update(lr=lr)
        metric_logger.update(loss=loss_value, **geo_metrics, **flo_metrics, **depth_metrics)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}