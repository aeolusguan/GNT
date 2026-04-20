# This file includes code from SEA-RAFT (https://github.com/princeton-vl/SEA-RAFT)
# Copyright (c) 2024, Princeton Vision & Learning Lab
# Licensed under the BSD 3-Clause License

import sys
sys.path.append('GeoNT/models/flow')

import argparse
import numpy as np
import random

from config.parser import parse_args
from core.model import FlowModel

import torch
import torch.optim as optim

from core.datasets import fetch_dataloader
from core.utils.utils import load_ckpt
from core.loss import sequence_loss, init_loss
import os
import GeoNT.utils.misc as misc
from GeoNT.models.external import load_moge

os.system("export KMP_INIT_AT_FORK=FALSE")

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps + 100,
                                              pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    
    return optimizer, scheduler

def train(args):
    """ Full training loop """

    device = torch.device("cuda")
    model = FlowModel(args).to(device)
    mono = load_moge('v2').to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False, static_graph=True
        )
        model_without_ddp = model.module

    if args.restore_ckpt is not None:
        load_ckpt(model_without_ddp, args.restore_ckpt)
        print(f"restore ckpt from {args.restore_ckpt}")

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    with open('%s/command.txt'%args.savedir, 'w') as f:
        f.write(' '.join(sys.argv))
        f.write('\n\n')
        f.write(str(args))
        f.write('\n\n')

    model.train()
    rank = args.rank
    world_size = args.world_size

    train_loader, train_sampler = fetch_dataloader(args, rank=rank, world_size=world_size)
    optimizer, scheduler = fetch_optimizer(args, model_without_ddp)
    total_steps = 0
    VAL_FREQ = 10000
    epoch = 0
    should_keep_training = True

    while should_keep_training:
        epoch += 1

        # manual change random seed for shuffling every epoch
        if world_size > 1:
            train_sampler.set_epoch(epoch)
            if hasattr(train_loader.dataset, "set_epoch"):
                train_loader.dataset.set_epoch(epoch)

        metric_logger = misc.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)

        for i_batch, data_blob in enumerate(metric_logger.log_every(train_loader, args.print_freq, header)):
            optimizer.zero_grad()
            image1, image2, flow, valid = [x.to(device) for x in data_blob]
            with torch.no_grad():
                depth_predictions = mono.infer(image1 / 255.0)
                mono_depth, valid_ = depth_predictions["depth"], depth_predictions["mask"]
                mono_depth = mono_depth.clamp_min(0.01)
                disp = torch.zeros_like(mono_depth)
                disp[valid_] = 1.0 / mono_depth[valid_]
                mono1 = depth_predictions["feature"]
                mono2 = mono.infer(image2 / 255.0)["feature"]
            output = model(image1, image2, disp.clone(), mono1.clone().float(), mono2.clone().float(), flow_gt=flow, iters=args.iters)
            i_loss, i_epe = init_loss(output, flow, valid)
            f_loss, f_epe = sequence_loss(output, flow, valid, args.gamma)
            loss = 0.5 * i_loss + f_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()

            if total_steps % VAL_FREQ == VAL_FREQ - 1 and rank == 0:
                PATH = '%s/%d_%s.pth' % (args.savedir, total_steps+1, args.name)
                torch.save(model_without_ddp.state_dict(), PATH)

            if total_steps > args.num_steps:
                should_keep_training = False
                break

            total_steps += 1

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)
            metric_logger.update(i_epe=float(i_epe.detach()), f_epe=float(f_epe.detach()))

    PATH = '%s/%s.pth' % (args.savedir, args.name)
    if rank == 0:
        torch.save(model_without_ddp.state_dict(), PATH)

    return PATH

def main(args):

    train(args)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--restore_ckpt', help='restore previews weights', default=None)

    parser.add_argument('--savedir', help='enable Depth Anything v2', type=str)
    parser.add_argument('--seed', help='set random seed', type=float, default=0)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--print_freq', default=20, type=int,
                        help='frequence (number of iterations) to print infos while training')
    args = parse_args(parser)

    misc.init_distributed_mode(args)

    # setting random seeds
    seed = args.seed + args.rank
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    main(args)
    print("Done!")