# This file includes code from SEA-RAFT (https://github.com/princeton-vl/SEA-RAFT)
# Copyright (c) 2024, Princeton Vision & Learning Lab
# Licensed under the BSD 3-Clause License

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('GeoNT/models/flow')
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.data as data

from config.parser import parse_args

import core.datasets as datasets
from core.model import FlowModel
from tqdm import tqdm
from core.utils.utils import load_ckpt
from GeoNT.models.external import load_moge

def forward_flow(args, model, mono, image1, image2):
    depth_predictions = mono.infer(image1 / 255.0)
    mono_depth, valid_ = depth_predictions["depth"], depth_predictions["mask"]
    mono_depth = mono_depth.clamp_min(0.01)
    disp = torch.zeros_like(mono_depth)
    disp[valid_] = 1.0 / mono_depth[valid_]
    mono1 = depth_predictions["feature"]
    mono2 = mono.infer(image2 / 255.0)["feature"]
    output = model(image1, image2, disp, mono1.float(), mono2.float(), iters=args.iters, test_mode=True)
    flow_final = output['flow'][-1]
    info_final = output['info'][-1]
    return flow_final, info_final

def calc_flow(args, model, mono, image1, image2):
    img1 = F.interpolate(image1, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    img2 = F.interpolate(image2, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    H, W = img1.shape[2:]
    flow, info = forward_flow(args, model, mono, img1, img2)
    flow_down = F.interpolate(flow, scale_factor=0.5 ** args.scale, mode='bilinear', align_corners=False) * (0.5 ** args.scale)
    info_down = F.interpolate(info, scale_factor=0.5 ** args.scale, mode='area')
    return flow_down, info_down

@torch.no_grad()
def validate_sintel(args, model, mono):
    """ Perform validation using the Sintel (train) split """
    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype)
        # val_dataset = datasets.TartanAir()
        val_loader = data.DataLoader(val_dataset, batch_size=4,
                                     pin_memory=False, shuffle=False, num_workers=8, drop_last=False)
        epe_list = np.array([], dtype=np.float32)
        px1_list = np.array([], dtype=np.float32)
        px3_list = np.array([], dtype=np.float32)
        px5_list = np.array([], dtype=np.float32)
        for i_batch, data_blob in enumerate(tqdm(val_loader)):
            image1, image2, flow_gt, valid = [x.cuda(non_blocking=True) for x in data_blob]
            flow, info = calc_flow(args, model, mono, image1, image2)
            epe = torch.sum((flow - flow_gt)**2, dim=1).sqrt()
            epe[valid <= 0.5] = 0
            px1 = (epe < 1.0).float().mean(dim=[1, 2]).cpu().numpy()
            px3 = (epe < 3.0).float().mean(dim=[1, 2]).cpu().numpy()
            px5 = (epe < 5.0).float().mean(dim=[1, 2]).cpu().numpy()
            epe = epe.mean(dim=[1, 2]).cpu().numpy()
            epe_list = np.append(epe_list, epe)
            px1_list = np.append(px1_list, px1)
            px3_list = np.append(px3_list, px3)
            px5_list = np.append(px5_list, px5)
        
        epe = np.mean(epe_list)
        px1 = np.mean(px1_list)
        px3 = np.mean(px3_list)
        px5 = np.mean(px5_list)
        # print("Validation %s EPE: %.2f, 1px: %.2f"%(dstype,epe,100 * (1 - px1)))
        print("Validation %s EPE: %.2f"%(dstype,epe,))

def eval(args):
    args.gpus = [0]
    model = FlowModel(args)
    if args.model is not None:
        load_ckpt(model, args.model)
    model = model.cuda()
    model.eval()
    mono = load_moge('v2').cuda()
    mono.eval()
    with torch.no_grad():
        if args.dataset == 'sintel':
            validate_sintel(args, model, mono)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--model', help='checkpoint path', type=str)
    parser.add_argument('--scale', help='input scale', type=int, default=0)
    parser.add_argument('--dataset', help='dataset type', type=str, required=True)
    args = parse_args(parser)
    eval(args)

if __name__ == '__main__':
    main()