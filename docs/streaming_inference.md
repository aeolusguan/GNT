# Streaming Inference

This branch contains a first runnable implementation of the proposed streaming
strategy:

1. Keyframes are selected by the motion filter.
2. After `warmup` keyframes are accepted, `OnePassInitializer.run()` adds
   temporal neighborhood edges inside the most recent `warmup` keyframe window. It keeps
   an edge when the mean flow magnitude is at most `frontend_thresh`.
3. The one-pass initializer uses `InitializationFactorGraph`, whose active graph
   state is only directed `ii`/`jj` edges. Edge ages, motion-token caching, and
   `max_factors` belong to the derived refinement `FactorGraph`.
4. `OnePassInitializer.finalize()` marginalizes all active source keyframes.
   Marginalization runs GeoNT with each finalized keyframe as common reference,
   saves the relative pose/depth constraints in `OnePassInitializer.edges`, and
   makes them available to later refinement.
5. Finalized relative constraints are optimized by the configured pose graph
   optimizer on the same device as the SLAM buffers, producing initialized
   keyframe depth scales and poses.
6. The second pass builds a `FactorGraph` over the initialized keyframes. It
   scores candidate directed pairs by coarse reprojection flow from the
   initialized pose/depth state, keeps projection-proximity factors below
   `frontend_thresh`, runs GeoNT refinement on those factors, and optimizes the
   resulting pose graph.

Install the checkout in editable mode once so package modules and console
entrypoints resolve from the `src/` layout:

```bash
pip install -e .
```

Run on a directory of RGB frames:

```bash
inference \
  streams.base_path=/path/to/frames \
  pipeline.slam.ckpt_path=/path/to/geont_checkpoint.pth \
  pipeline.slam.intrinsics=[128,128,64,48] \
  pipeline.output.path=outputs
```

All pipeline parameters are configured through Hydra YAML and Hydra command-line
overrides. Use `configs/default.yaml` as the source of truth for streaming
defaults; avoid adding separate argparse defaults for the same parameters.
For example:

```bash
inference \
  streams.base_path=/path/to/frames \
  pipeline.slam.ckpt_path=/path/to/geont_checkpoint.pth \
  pipeline.slam.intrinsics=[128,128,64,48] \
  pipeline.slam.warmup=8 \
  pipeline.slam.pgo_backend=torch
```

Frame directories are read in natural filename order, so numbered frame names
such as `1.png`, `2.png`, `10.png` keep their temporal order.

## Config Protocol

The streaming code expects an OmegaConf/DictConfig-style config with explicit
fields instead of defensive defaults inside the runtime code. The minimal
`pipeline.slam` schema is:

```yaml
ckpt_path: /path/to/geont_checkpoint.pth
intrinsics: [128.0, 128.0, 64.0, 48.0]
visualize: false
buffer: 512
filter_thresh: 2.5
warmup: 8
frontend_radius: 2
frontend_thresh: 16.0
max_factors: 256
seq_init: false
pgo_iters: 12
pgo_damping: 1.0e-3
pgo_lm_max_attempts: 5
pgo_huber_delta: 1.0
pgo_scale_conf: 0.01
pgo_mode: se3_scale
pgo_rotation_only: false
pgo_backend: cuda_eigen
use_fp16: true
```

The current streaming implementation is single-view. `intrinsics` is fixed for
the whole video and is specified in the original image coordinates. The stream
processor applies it before resize/crop so the model receives transformed
intrinsics, while saved artifacts recover the original configured intrinsics.
`pgo_scale_conf` is the soft per-node pose-scale prior strength used by the
active SE3+scale PGO path.
`max_factors` caps the second-pass projection-proximity factors selected from
the initialized keyframe graph before GeoNT refinement.
`pgo_lm_max_attempts` is the maximum number of Levenberg-Marquardt damping
attempts per linearized update before the solve stops.
`pgo_mode: se3_scale` runs the joint SE3+scale optimizer and is the default
path for the current network predictions. Use `pgo_mode: staged` for the
rotation-first, fixed-rotation translation+scale ablation. `pgo_backend:
cuda_eigen` is the production PGO backend: it
builds fixed-layout CUDA Jacobian blocks, copies them to CPU double precision,
assembles a sparse Eigen normal matrix, and solves it with `SimplicialLLT`.
`torch` remains available as the dense reference path for CPU/unit tests.
`normal_equation_assembly` and `linear_solver_impl` in `pgo_info` distinguish
the concrete path. The CUDA extension uses vendored Eigen headers from
`third_party/eigen/upstream`; set `GNT_EIGEN_INCLUDE_DIR` only to override
that project-local dependency.

DROID-SLAM's native inference BA is a useful accuracy reference but not the
same solve path: it builds BA/Schur blocks with CUDA kernels, transfers sparse
blocks to CPU double precision, solves with Eigen `SimplicialLLT`, then copies
the increment back to CUDA for SE3/disparity retraction. The `cuda_eigen` PGO
backend follows that CPU sparse-solve pattern for the PGO normal equations.

The pose artifact is written to:

```text
outputs/pose/<sequence_name>.npz
```

The keyframe depth artifact is written to:

```text
outputs/depth/<sequence_name>.npz
```

The `.npz` contains:

- `trajectory`: keyframe pose encodings as `[tx, ty, tz, qx, qy, qz, qw]`
- `log_scales`: optimized per-keyframe coherent pose-scale log-scales,
  initialized from MoGe/reference scale
- `scales`: `exp(log_scales)` convenience values for pose translation and
  metric-ish depth recovery
- `timestamps`: keyframe indices in the processed stream
- `intrinsics`: recovered original `[fx, fy, cx, cy]`
- `edge_ii`, `edge_jj`, `edge_relative_pose`, `edge_relative_scale`,
  `edge_confidence`: finalized fixed pose-graph factors.
  `edge_confidence` stores the confidence/precision head; older code paths may
  still expose it under the legacy `pose_log_variance` name. The rotation confidence column is fixed to `1.0` because edge rotations are accurate while
  the learned rotation confidence is currently uninformative.
  `edge_relative_scale` stores the marginalized source/reference depth scale as
  a separate scalar side channel, not as a raw Sim3 group scale.
- `pgo_info`: JSON-encoded diagnostics from the last PGO pass, including
  initial/final cost, accepted iteration count, rejected LM attempts, and solver
  failure count.

The depth `.npz` contains:

- `depths`: scale-invariant keyframe-aligned depth maps as `(N, 1, H, W)`.
  To recover metric-ish depth, multiply by the matching `scales` entry from
  the pose artifact.
- `masks`: corresponding non-sky/valid masks
- `timestamps`: keyframe indices in the processed stream

The finalized edge convention is source-depth-normalized:

```text
edge_relative_pose_ij[:3] ~= translation(T_j * inv(T_i)) / scale_i
edge_relative_pose_ij[3:7] ~= rotation(T_j * inv(T_i))
edge_relative_scale_ij ~= scale_i * mean(depth_moge_i) / mean(depth_refined_i)
```

The active PGO path first optimizes rotations from the initialized pose tree,
then keeps those rotations fixed while optimizing translation and node scale.
The source-normalized translation model is active in the second stage.
Internally, each finalized prediction is represented as a source-normalized
lietorch `SE3` edge plus its separate `edge_relative_scale_ij` scalar. The
translation+scale PGO residual uses the SE3 edge translation and the soft
node-scale prior:

```text
translation(T_j * inv(T_i)) / exp(log_s_i)
  ~= edge_relative_pose_ij[:3]
```

The separate `edge_relative_scale_ij` scalar is still preserved as the
marginalized reference-depth scale diagnostic; it is not interpreted as a raw
Sim3 group scale. PGO fixes the first keyframe pose as the origin/reference;
all node scales, including the first keyframe scale, remain optimized under
the soft scale prior. No per-edge translation scale correction is optimized in
the active PGO path. With the current network predictions, joint `se3_scale`
PGO is the default accuracy path; staged PGO remains available for ablation.

PGO implementation lives under `src/geont_runtime/slam/pgo/`:

- `optimizer.py`: public optimizer entry point and staged/joint mode dispatch
- `common.py`: shared residuals, initialization, Jacobian helpers, and result types
- `torch_backend.py`: dense Torch reference implementation
- `cuda_eigen.py`: CUDA block assembly plus CPU Eigen sparse Cholesky backend
- `cuda_backend.py` and `cuda/`: lazy-built native CUDA extension wrapper/source
- `replay.py`: frozen PGO replay graph serialization helpers

## Visualize One-Pass Factor Graph

The one-pass initializer saves finalized directed graph edges in the pose
artifact. Render a timeline view with:

```bash
python scripts/visualize_init_graph.py \
  outputs/pose/<sequence_name>.npz \
  --output outputs/pose/<sequence_name>_init_graph.svg
```

Nodes are keyframes ordered by keyframe index, labels show keyframe index and
original timestamp, blue arcs are forward directed edges, orange arcs are
backward directed edges, node size shows incident degree, and edge opacity shows
the mean pose confidence.

## TartanAir PGO Evaluation

The TartanAir PGO evaluator is Hydra-configured through
`configs/tartanair_pgo_eval.yaml`. Keep experiment hyperparameters there so a
run can be reproduced from the config file instead of from a long command:

```bash
python scripts/evaluate_tartanair_pgo.py
python scripts/evaluate_tartanair_pgo.py max_scenes=3 slam.pgo_lm_max_attempts=1
python scripts/evaluate_tartanair_pgo.py slam.pgo_backend=torch
```

For incremental rotation debugging, set `slam.pgo_rotation_only=true`. The
TartanAir evaluator writes `optimized_rotation_error_deg` and
`rotation_error_improvement_deg` in `edge_relative_pose_errors.csv` to compare
the optimized relative rotations against the raw edge rotation predictions.

Each evaluated scene also writes `pgo_replay_graph.npz`, a frozen PGO input
snapshot with the finalized edge predictions, confidence, initial poses, and
initial log scales. Reuse it to debug only PGO without running the network
again:

```bash
python scripts/generate_tartanair_pgo_replay.py \
  ckpt=output/checkpoint-best.pth \
  output=outputs/pgo_replay_full_p011_best

python scripts/replay_pgo.py \
  graph=outputs/tartanair_pgo_eval/<scene>/pgo_replay_graph.npz \
  backend=cuda \
  mode=se3_scale \
  rotation_only=false \
  iters=12

python scripts/replay_pgo.py \
  graph=outputs/tartanair_pgo_eval/<scene>/pgo_replay_graph.npz \
  backend=cuda_eigen \
  mode=se3_scale \
  rotation_only=false \
  iters=12

python scripts/replay_pgo.py \
  graph=outputs/tartanair_pgo_eval/<scene>/pgo_replay_graph.npz \
  backend=torch \
  device=cpu \
  repeat=5
```

PGO variant sweeps are also Hydra-configured through
`configs/pgo_variant_ablation.yaml`:

```bash
python scripts/ablate_pgo_variants.py \
  graph=outputs/pgo_replay_abandonedfactory_easy_p011/abandonedfactory__abandonedfactory__Easy__P011/pgo_replay_graph.npz \
  scene=datasets/TartanAir/abandonedfactory/Easy/P011 \
  output=outputs/pgo_variant_ablation_p011 \
  scale_conf=[0.01,0.03] \
  huber_delta=[0.05]
```
