import hydra
from omegaconf import DictConfig
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@hydra.main(version_base=None, config_path="configs", config_name="default")
def run(args: DictConfig) -> None:
    from geont_runtime.streams.base import StreamList

    # Gather all video streams
    stream_list = StreamList.make(args.streams)

    from geont_runtime.pipeline import make_pipeline
    from geont_runtime.utils.logging import configure_logging

    # Process each video stream
    logger = configure_logging()
    for stream_idx in range(len(stream_list)):
        video_stream = stream_list[stream_idx]
        logger.info(f"Processing {video_stream.name()} ({stream_idx + 1} / {len(stream_list)})")
        pipeline = make_pipeline(args.pipeline)
        pipeline.run(video_stream)
        logger.info(f"Finished processing {video_stream.name()}")


if __name__ == "__main__":
    run()
