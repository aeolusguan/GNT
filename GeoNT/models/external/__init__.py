from collections import OrderedDict
import sys
import argparse
from os.path import dirname, join
from pathlib import Path
import torch
# RAFT_PATH_ROOT = join(dirname(__file__), 'SEA-RAFT')
# sys.path.append(RAFT_PATH_ROOT)

from .raft import RAFT, InputPadder
from .moge import import_model_class_by_version
from .romav2 import RoMaV2, _interpolate_warp_and_confidence, to_pixel


def load_raft():
    import gdown

    args = argparse.Namespace(
        use_var=True,
        var_min=0,
        var_max=10,
        pretrain="resnet34",
        initial_dim=64,
        block_dims=[64, 128, 256],
        radius=4,
        dim=128,
        num_blocks=2,
        iters=4,
    )
    net = RAFT(args)

    # Download ckpt if needed.
    ckpt_path = Path(torch.hub.get_dir()) / "raft" / "Tartan-C-T-TSKH-spring540x960-M.pth"
    if not ckpt_path.exists():
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(
            "https://drive.google.com/file/d/1a0C5FTdhjM4rKrfXiGhec7eq2YM141lu/view?usp=drive_link",
            output=str(ckpt_path),
            fuzzy=True,
        )

    state_dict = OrderedDict(
        [(k.replace("module.", ""), v) for (k, v) in torch.load(ckpt_path, map_location="cpu").items()]
    )
    net.load_state_dict(state_dict)
    return net.eval()


def load_moge(version: str):
    model = import_model_class_by_version(version).from_pretrained("Ruicheng/moge-2-vitl")
    return model.eval()


def load_romav2():
    model = RoMaV2()
    model.apply_setting("base")
    return model