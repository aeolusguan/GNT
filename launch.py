from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from geont.training import get_args_parser, train, load_model
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import os


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    train(args)
