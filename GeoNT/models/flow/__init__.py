import argparse
import json
import os
from pathlib import Path

from .core.model import FlowModel
from .core.utils.utils import load_ckpt

REPO_DIR = Path(__file__).parent.parent.parent.parent

def load_flow():
    json_path = os.path.join(os.path.dirname(__file__), 'config/train/flow-T.json')
    with open(json_path, 'r') as f:
        data = json.load(f)

    args = argparse.Namespace()
    args_dict = args.__dict__
    for key, value in data.items():
        args_dict[key] = value
    net = FlowModel(args)

    # Download ckpt if needed.
    ckpt_path = REPO_DIR / "checkpoints" / "flow_T_TartanCT_TSKH.pth"
    assert ckpt_path.exists(), f"Checkpoint not found at {ckpt_path}"

    load_ckpt(net, ckpt_path)
    return net