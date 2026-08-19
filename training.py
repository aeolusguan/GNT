import sys
import datetime
import json
import numpy as np
import time
import math
from pathlib import Path
from typing import Sized

import hydra
import torch
import torch.backends.cudnn as cudnn
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from gent.model.gent import GeNTWrapper
from gent.data import get_data_loader
from gent.data.factory import dataset_factory
from gent.losses import MultitaskLoss

from gent.geometry.graph_utils import build_frame_graph

import gent.utils.misc as misc
from gent.utils.misc import NativeScalerWithGradNormCount as NativeScaler


torch.backends.cuda.matmul.allow_tf32 = True


def load_da3_state(source: str | Path):
    from safetensors.torch import load_file

    source = Path(source)
    if source.exists():
        weight_path = source / "model.safetensors" if source.is_dir() else source
    else:
        from huggingface_hub import hf_hub_download

        weight_path = Path(hf_hub_download(repo_id=str(source), filename="model.safetensors"))
    return load_file(weight_path, device="cpu"), weight_path


def load_da3_model_weights(model: GeNTWrapper, source: str | Path) -> None:
    da3_state, weight_path = load_da3_state(source)
    update_state = {}
    for key, value in da3_state.items():
        if key.startswith("model.backbone."):
            update_state[f"backbone.{key.removeprefix('model.backbone.')}"] = value
        elif key.startswith("model.cam_dec."):
            update_state[f"cam_dec.{key.removeprefix('model.cam_dec.')}"] = value
        elif model.gnt.cam_enc is not None and key.startswith("model.cam_enc."):
            update_state[f"cam_enc.{key.removeprefix('model.cam_enc.')}"] = value
    load_report = model.gnt.load_state_dict(update_state, strict=False)
    print(f"DA3 init from {weight_path}")
    print(f"Missing DA3 init parameters: {load_report.missing_keys}")
    print(f"Unexpected DA3 init parameters: {load_report.unexpected_keys}")


def load_da3_camera_encoder(model: GeNTWrapper, source: str | Path) -> None:
    da3_state, weight_path = load_da3_state(source)
    camera_state = {
        key.removeprefix("model.cam_enc."): value
        for key, value in da3_state.items()
        if key.startswith("model.cam_enc.")
    }
    model.gnt.cam_enc.load_state_dict(camera_state)
    print(f"Camera encoder init from {weight_path}")


def load_initial_checkpoint(model: GeNTWrapper, checkpoint_path: str | Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    load_report = model.load_state_dict(checkpoint["model"], strict=False)
    allowed_missing = set()
    if model.gnt.cam_enc is not None:
        allowed_missing = {
            f"gnt.cam_enc.{key}"
            for key in model.gnt.cam_enc.state_dict()
        }
    if set(load_report.missing_keys) != allowed_missing or load_report.unexpected_keys:
        raise RuntimeError(
            "Invalid weights-only initialization: "
            f"missing={load_report.missing_keys}, unexpected={load_report.unexpected_keys}"
        )
    print(f"Model init from {checkpoint_path}")


def build_training_model(args, device):
    model = GeNTWrapper(args.model.gnt)
    model.to(device)
    model_without_ddp = model
    if args.resume is None:
        if args.init.checkpoint is None:
            load_da3_model_weights(model, args.init.da3.path)
        else:
            load_initial_checkpoint(model, args.init.checkpoint)
            if model.gnt.cam_enc is not None:
                load_da3_camera_encoder(model, args.init.da3.path)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=False,
            static_graph=True,
            gradient_as_bucket_view=True,
        )
        model_without_ddp = model.module

    return model, model_without_ddp


def train(args):
    args.output_dir = to_absolute_path(args.output_dir)
    args.data.tartanair.root = to_absolute_path(args.data.tartanair.root)
    args.data.arkitscenes.root = to_absolute_path(args.data.arkitscenes.root)
    args.data.dynamic_replica.root = to_absolute_path(
        args.data.dynamic_replica.root
    )
    if args.resume is not None:
        args.resume = to_absolute_path(args.resume)
    if args.init.checkpoint is not None:
        args.init.checkpoint = to_absolute_path(args.init.checkpoint)

    misc.init_distributed_mode(args)

    output_dir = Path(args.output_dir)
    print(f"output_dir: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"job dir: {Path(__file__).resolve().parent}")
    print("{}".format(args).replace(', ', ',\n'))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = not args.disable_cudnn_benchmark

    model, model_without_ddp = build_training_model(args, device)
    
    db = dataset_factory(
        args.data,
        n_frames=args.n_frames,
        fmin=args.fmin,
        fmax=args.fmax,
    )
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
    param_groups = misc.get_parameter_groups(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    misc.resume_training_state(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    train_stats = {}
    for epoch in range(args.start_epoch, args.epochs + 1):
        already_saved = False

        if misc.is_main_process():
            log_stats = dict(epoch=epoch, **{f"train_{k}": v for k, v in train_stats.items()})
            with (output_dir / "log.txt").open(mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        if epoch > args.start_epoch:
            if args.keep_freq and epoch % args.keep_freq == 0:
                misc.save_model(
                    args=args,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch - 1,
                    fname=str(epoch),
                )
                already_saved = True

            if (args.save_freq and epoch % args.save_freq == 0) or (epoch == args.epochs and not already_saved):
                misc.save_model(
                    args=args,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch - 1,
                    fname="last",
                )
        
        if epoch >= args.epochs:
            break

        train_stats = train_one_epoch(
            model, train_criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args=args
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Sized, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    args,
):
    assert torch.backends.cuda.matmul.allow_tf32

    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    accum_iter = args.accum_iter

    data_loader.dataset.set_epoch(epoch)
    data_loader.sampler.set_epoch(epoch)

    optimizer.zero_grad(set_to_none=True)

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        epoch_f = epoch + data_iter_step / len(data_loader)

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            misc.adjust_learning_rate(optimizer, epoch_f, args)

        # Build the frame graph from the CPU batch before transferring training tensors.
        images, poses, depths, depths_valid, intrinsics = batch
        disps = torch.where(
            depths_valid,
            depths.reciprocal(),
            torch.zeros_like(depths),
        )
        graph = build_frame_graph(
            poses,
            disps,
            intrinsics,
            valid=depths_valid,
            num=args.edges,
        )
        images, poses, depths, depths_valid, intrinsics = [
            x.to(device, non_blocking=True) for x in batch
        ]

        prediction = model(
            images,
            intrinsics,
            graph,
            depths_valid,
            use_fp16=bool(args.amp),
        )

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
        loss_value = float(loss.detach())
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), force=True)
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad(set_to_none=True)
        
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


@hydra.main(version_base=None, config_path="configs", config_name="gent_train")
def main(args: DictConfig) -> None:
    train(args)


if __name__ == "__main__":
    main()
