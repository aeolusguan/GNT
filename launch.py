from GeoNT.training import get_args_parser, train, load_model
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import os


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    train(args)