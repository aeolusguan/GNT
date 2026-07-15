# GeoNT

GeoNT is a research codebase for geometry-aware neural tracking from monocular
video. It combines a GeoNT neural model with a streaming runtime for keyframe
selection, depth/pose prediction, and pose-graph optimization.

The codebase is organized as a modern `src/` Python project:

```text
src/geont/          # model architecture, training, losses, data, geometry
src/geont_runtime/  # inference, streams, SLAM, pose graph optimization
configs/            # Hydra runtime and experiment configs
scripts/            # evaluation, ablation, diagnostics, and visualization tools
third_party/        # vendored external dependencies
```

## Installation

Create and activate your Python environment, then install the checkout in
editable mode:

```bash
pip install -e .
```

The project expects PyTorch, CUDA-enabled dependencies, Hydra/OmegaConf, OpenCV,
NumPy, and the model-specific dependencies used by the GeoNT and flow modules to
be available in the environment.

Vendored dependencies live under `third_party/`. The CUDA PGO backend uses Eigen
headers from:

```text
third_party/eigen/upstream
```

Set `GNT_EIGEN_INCLUDE_DIR` only if you explicitly want to override that
project-local Eigen dependency.

## Checkpoints and data

Place model checkpoints under `checkpoints/` or pass explicit checkpoint paths
through the config or command line.

The flow frontend expects:

```text
checkpoints/flow_T_TartanCT_TSKH.pth
```

Training defaults expect TartanAir-style data under:

```text
datasets/TartanAir
```

Dataset index caches are written under:

```text
src/geont/data/cache/
```

## Streaming inference

After editable install, run inference with the package CLI:

```bash
inference \
  streams.base_path=/path/to/frames \
  pipeline.slam.ckpt_path=/path/to/geont_checkpoint.pth \
  pipeline.slam.intrinsics=[128,128,64,48] \
  pipeline.output.path=outputs
```

You can inspect the resolved default config without running the model:

```bash
inference --cfg job
```

All runtime parameters are configured through Hydra. The default streaming
configuration is:

```text
configs/default.yaml
```

Common overrides:

```bash
inference \
  streams.base_path=/path/to/frames \
  pipeline.slam.ckpt_path=checkpoints/geont.pth \
  pipeline.slam.intrinsics=[320,320,320,240] \
  pipeline.slam.pgo_backend=cuda_eigen \
  pipeline.slam.pgo_iters=12 \
  pipeline.output.path=outputs/example
```

The inference runtime is currently single-view. Camera intrinsics are fixed for
the sequence and should be specified as:

```text
[fx, fy, cx, cy]
```

## Outputs

Streaming inference writes pose and depth artifacts under the configured output
directory:

```text
outputs/pose/<sequence_name>.npz
outputs/depth/<sequence_name>.npz
```

The pose artifact contains keyframe trajectory, timestamps, intrinsics,
optimized scales, immediate pose-graph edges, and PGO diagnostics. The depth
artifact contains keyframe-aligned depth maps, masks, and timestamps.

See `docs/streaming_inference.md` for the detailed artifact schema and pose
graph conventions.

## Training

GeoNT training uses the root training script:

```bash
python training.py \
  --config-name=geont_train \
  datapath=datasets/TartanAir \
  output_dir=output/geont_train
```

Useful options include:

```text
init.da3.path                    DA3 safetensors initialization source for non-resume training
resume                           resume training checkpoint path
batch_size                       per-GPU batch size
epochs                           number of training epochs
n_frames                         number of frames sampled per training example
edges                            training graph edge budget
```

Set `resume=...` to restore a training checkpoint. Otherwise, set
`init.da3.path`.

For DA3-SMALL initialization, set the model shape from config and use the
HuggingFace safetensors source:

```bash
python training.py --config-name=geont_train \
  model.gnt.backbone.name=vits \
  model.gnt.backbone.img_size=518 \
  model.gnt.backbone.patch_size=14 \
  init.da3.path=depth-anything/DA3-SMALL
```

## Evaluation and analysis scripts

The `scripts/` directory contains experiment helpers for TartanAir evaluation,
PGO ablation, diagnostics, and visualization. Examples:

```bash
python scripts/evaluate_tartanair_pgo.py
python scripts/ablate_pgo_moge_mode_count.py --artifact-root outputs/streaming_eval
python scripts/visualize_init_graph.py outputs/pose/example.npz --output outputs/pose/example_init_graph.svg
```

Most experiment scripts are Hydra-configured. Check the matching files under
`configs/` before launching a run.
