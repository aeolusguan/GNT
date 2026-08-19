# GeNT

GeNT is a research codebase for geometry-aware neural tracking from monocular
video. It combines a GeNT neural model with a streaming runtime for keyframe
selection, depth/pose prediction, and pose-graph optimization.

The codebase uses a flat Python package layout:

```text
gent/
├── model/          # GeNT architecture, flow, MoGe integration, and heads
├── runtime/        # streaming inference, SLAM, PGO, and visualization
├── data/           # training and evaluation datasets
├── geometry/       # projective geometry and graph construction
├── retrieval/      # optional proximity retrieval
└── utils/          # shared utilities
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
NumPy, and the model-specific dependencies used by the GeNT and flow modules to
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
gent/data/cache/
```

## Streaming inference

After editable install, run inference with the package CLI:

```bash
inference \
  streams.base_path=/path/to/frames \
  pipeline.slam.ckpt_path=/path/to/gent_checkpoint.pth \
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
  pipeline.slam.ckpt_path=checkpoints/gent.pth \
  pipeline.slam.intrinsics=[320,320,320,240] \
  pipeline.slam.pgo_backend=cuda_eigen \
  pipeline.slam.pgo_iters=12 \
  pipeline.output.path=outputs/example
```

The inference runtime is monocular, but calibration may change between frames.
Pass one pinhole row for every selected frame:

```text
[[fx_0, fy_0, cx_0, cy_0], ..., [fx_F-1, fy_F-1, cx_F-1, cy_F-1]]
```

For a fixed camera, `[fx, fy, cx, cy]` is accepted as shorthand and expanded
to all selected frames.

## Outputs

Streaming inference writes pose and depth artifacts under the configured output
directory:

```text
outputs/pose/<sequence_name>.npz
outputs/depth/<sequence_name>.npz
```

The pose artifact contains keyframe trajectory, timestamps, keyframe-aligned
intrinsics, frame-aligned intrinsics,
optimized scales, immediate pose-graph edges, and PGO diagnostics. The depth
artifact contains keyframe-aligned depth maps, masks, and timestamps.

See `docs/streaming_inference.md` for the detailed artifact schema and pose
graph conventions.

## Training

GeNT training uses the root training script:

```bash
python training.py \
  --config-name=gent_train \
  init.da3.path=depth-anything/DA3-BASE \
  data.tartanair.root=datasets/TartanAir \
  output_dir=output/gent_train
```

Useful options include:

```text
init.da3.path                    DA3 safetensors initialization source
init.checkpoint                 weights-only initialization checkpoint
resume                           strict full-state resume checkpoint
batch_size                       per-GPU batch size
epochs                           number of training epochs
n_frames                         number of frames sampled per training example
edges                            training graph edge budget
```

Stage one uses `gent_train` with `model.gnt.camera_encoder=null`. Stage two uses
the fixed camera-prior protocol:

```bash
python training.py --config-name=gent_camera_train
```

The depth/validity and motion patch embedders start from random weights in stage
one. They allocate `D/4` and `3D/4` token dimensions, respectively.
`gent_camera_train` then loads the learned patch embedders and the remaining
GeNT weights from the completed stage-one
checkpoint through `init.checkpoint`, and initializes only the new camera
encoder from DA3. Camera-conditioned source groups replace the learned camera tokens;
the two token types are never added or mixed. Each batch uses a balanced split
between learned-token groups and self-conditioned groups. For a self-conditioned
group, flow, monocular depth, and GeNT patch tokens are computed once; only the
GeNT solver runs a second time with the detached first-pass pose.

Use `resume=...` only for a camera-prior checkpoint. Resume restores the model,
optimizer, scaler, and epoch strictly.

At runtime, accepted frontend tracking groups retain their direct pose and
relative-scale measurements. Backend temporal aggregation uses the direct past
measurements and source-gauge-converted inverse future measurements as its
CameraEnc prior, then runs one conditioned pose-depth solver pass. Frontend
tracking and nonlocal pose-only measurements remain one-pass learned-token
inference.

`training.py` is the training entrypoint, and `train.sh` is the corresponding
launcher.

For DA3-SMALL initialization, set the model shape from config and use the
HuggingFace safetensors source:

```bash
python training.py --config-name=gent_train \
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
