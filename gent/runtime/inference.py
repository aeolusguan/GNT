from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf


ROOT = Path(__file__).resolve().parents[2]


def _config_path(value) -> Path:
    return Path(to_absolute_path(str(value)))


def _validate_frame_range(frame_start: int, frame_end: int, frame_skip: int) -> None:
    if frame_start < 0:
        raise ValueError("frame_start must be non-negative")
    if frame_skip <= 0:
        raise ValueError("frame_skip must be positive")
    if frame_end != -1 and frame_end <= frame_start:
        raise ValueError("frame_end must be greater than frame_start, or -1 for the full sequence")


def build_streaming_config(
    frame_dir: str | Path,
    ckpt_path: str | Path,
    intrinsics: Sequence[float] | Sequence[Sequence[float]],
    output_dir: str | Path = "outputs",
    slam_config: DictConfig | dict | None = None,
) -> DictConfig:
    """Build a streaming config by binding scene-specific fields to the Hydra defaults."""
    cfg = OmegaConf.load(ROOT / "configs/default.yaml")
    cfg.streams.base_path = str(frame_dir)
    cfg.pipeline.output.path = str(output_dir)
    if slam_config is not None:
        cfg.pipeline.slam = OmegaConf.merge(cfg.pipeline.slam, slam_config)
    cfg.pipeline.slam.ckpt_path = str(ckpt_path)
    cfg.pipeline.slam.intrinsics = [list(value) if isinstance(value, Sequence) else value for value in intrinsics]
    return cfg


def run_streaming_config(cfg: DictConfig):
    """Run streaming inference from a fully resolved Hydra config."""
    cfg = cfg.copy()
    cfg.streams.base_path = str(_config_path(cfg.streams.base_path))
    cfg.pipeline.output.path = str(_config_path(cfg.pipeline.output.path))
    cfg.pipeline.slam.ckpt_path = str(_config_path(cfg.pipeline.slam.ckpt_path))

    frame_start = int(cfg.streams.frame_start)
    frame_end = int(cfg.streams.frame_end)
    frame_skip = int(cfg.streams.frame_skip)
    _validate_frame_range(frame_start, frame_end, frame_skip)

    from gent.runtime.pipeline.default import DefaultAnnotationPipeline
    from gent.runtime.streams.frame_dir_stream import FrameDirStreamList

    stream_list = FrameDirStreamList(
        str(cfg.streams.base_path),
        frame_start=frame_start,
        frame_end=frame_end,
        frame_skip=frame_skip,
        cached=bool(cfg.streams.cached),
    )
    pipeline = DefaultAnnotationPipeline(
        init=cfg.pipeline.init,
        slam=cfg.pipeline.slam,
        post=cfg.pipeline.post,
        output=cfg.pipeline.output,
    )
    pipeline.return_payload = True

    outputs = []
    for idx in range(len(stream_list)):
        outputs.append(pipeline.run(stream_list[idx]).payload)
    return outputs


def _main_task(cfg: DictConfig) -> None:
    run_streaming_config(cfg)


@hydra.main(version_base=None, config_path="../../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    _main_task(cfg)


def console_main() -> None:
    from hydra._internal.utils import _run_hydra, get_args_parser

    args_parser = get_args_parser()
    args = args_parser.parse_args()
    if args.config_dir is None:
        args.config_dir = str(ROOT / "configs")
    _run_hydra(
        args=args,
        args_parser=args_parser,
        task_function=_main_task,
        config_path=None,
        config_name="default",
    )


if __name__ == "__main__":
    main()
