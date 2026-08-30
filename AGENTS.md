# Codex Project Notes

## Code style preference

让代码更像研究代码，而不是兼容层集合。

This is research code. Prefer simple, explicit, easy-to-read implementations
over defensive compatibility layers. Encode required assumptions in docs/config
instead of hiding them behind broad fallbacks. When a runtime protocol is fixed
for the current research stage, keep the code direct: one timeline, one
single-view stream, per-frame video intrinsics, and no unused multi-view/rig
plumbing.

Do not add local robustness guards that silently sanitize inputs already defined
by the protocol. For example, graph helper callers should pass valid `t0`/`t1`
frame ranges directly instead of having the helper clamp or reinterpret them.
Required neural-network outputs and pose-edge fields should be required
in code, not hidden behind optional compatibility paths. If the current
research protocol guarantees a value, type it and handle it as required; avoid
`None` branches, legacy fallbacks, and broad compatibility code unless the
research protocol itself changes.
For fixed training/runtime config protocols, read the agreed attribute directly
instead of wrapping it in dict/getattr/type-conversion compatibility helpers.
The caller owns passing the right config object and value type.
For factor graphs, directed duplicate edges are filtered when edges are inserted.

In SLAM inference codes, including python and C++, don't use unnecessary detach() method as that in training code.

## Research motivation and system target

The current system is GeNT-centered streaming local mapping for robot
perception. The primary goal is high-quality streaming video depth and local
trajectory/scale consistency, not competing with dense BA systems such as
DROID-SLAM or MASt3R-SLAM on global ATE as the main claim.

The paper/system motivation should emphasize local system perception:

- GeNT means Geometry-Native Transformer. The model directly consumes geometry
  signals such as optical flow and monocular depth, rather than using visual
  feature representations as the main token signal.
- GeNT's key advantage is depth estimation and local consistency. The runtime
  should preserve this advantage by producing low-latency refined keyframe
  depths and a locally coherent pose/scale estimate useful for robotic local
  mapping.
- The pose graph optimizer is still important, but its role is local pose and
  monocular-scale smoothing for depth consistency. It is not the central claim
  that the pose graph alone should beat BA-based SLAM methods globally.
- MASt3R/ASMK loop closure is optional experimental infrastructure for
  large-baseline loop/relocalization. It must not replace GeNT local
  measurements in the default runtime path.
- Frontend tracking and nonlocal pose-only GeNT measurements use the learned
  camera tokens. Accepted frontend tracking groups cache direct pose and
  relative-scale measurements. Backend temporal multi-view aggregation builds
  CameraEnc tokens from those measurements and runs one pose-depth solver pass.
  Learned and encoded camera tokens are never mixed within one solver pass.

## Streaming local mapping design state

`SLAMSystem.run()` is now a streaming loop. It does not run the old one-pass
initializer, offline frontend sweep, or backend proximity replay path. The
default flow is:

1. Preprocess each frame with its own intrinsics and the standard resize/crop.
2. Encode GeNT feature maps, normalized monocular depth, non-sky mask, bases,
   and initial source scale into a `KeyframeCandidate`.
3. `SLAMFrontend.track()` runs tracking multi-view aggregation for every frame
   against up to `local_mapping_radius` previous keyframes, then uses
   coarse optical-flow motion only to decide whether the frame becomes a
   keyframe.
4. If accepted, the frontend commits an immediate keyframe depth and seeds the
   new keyframe pose from `current -> last`. If not accepted, the refined depth
   is transient and only the frame pose record is kept.
5. After `local_mapping_radius` later keyframes, `SLAMBackend` runs one
   temporal multi-view finalization for that source keyframe, inserts the
   resulting temporal pose edges, may add nonlocal proximity pose edges, runs
   marginalized local PGO, and at the end of the sequence runs one final full-graph
   PGO before output.

Frontend responsibilities:

- Keyframe admission uses coarse optical-flow motion against the last accepted
  keyframe. Online tracking currently assumes success for every processed
  frame; there is no relocalization module in the default path.
- The first frame is accepted directly with monocular depth. Later frames run
  GeNT tracking multi-view aggregation with up to two previous keyframes,
  newest first, before keyframe admission.
- Tracking multi-view aggregation uses the uncommitted candidate as source and
  previous committed keyframes as neighbors. It produces immediate refined
  depth and relative poses for tracking.
- Frontend tracking writes immediate refined depth only for accepted keyframes.
  Non-keyframe refined depth is a one-time transient result and is not stored in
  `GraphBuffer` or artifacts.
- Tracking-stage relative pose is used to seed accepted keyframe poses and to
  store non-keyframe frame pose records. Tracking-stage edges are not inserted
  into `PoseGraphEdges`.
- Accepted keyframes retain their complete tracking group until backend
  temporal finalization. Non-keyframes do not enter this cache.
- `SLAMOutput` keeps keyframe-aligned `trajectory/timestamps/depths` for
  backward compatibility and additionally stores full-frame
  `frame_trajectory/frame_timestamps`. Final non-keyframe poses are rebuilt
  after full-graph PGO from their tracking-time relative pose and relative
  log-scale against the optimized reference keyframe pose/scale.

Backend responsibilities:

- Temporal graph construction waits until a keyframe has
  `local_mapping_radius` later keyframes available. It then runs one GeNT
  multi-view aggregation with all temporal neighbors in the radius, commits the
  refined source depth/scale, and inserts the source-to-neighbor relative poses
  into `PoseGraphEdges`.
- Backend temporal aggregation reuses direct source-to-past tracking poses and
  flow. Future tracking poses are inverted and their translations are converted
  to the source normalized-depth gauge with the predicted relative log scale.
  Source-to-future flow remains directional and is computed normally. The
  complete prior drives one CameraEnc-conditioned pose-depth solver pass.
- Each keyframe receives at most one backend temporal depth update. After
  temporal finalization, do not use GeNT to update that keyframe depth again.
- Nonlocal proximity edges are selected outside the temporal radius and use
  one-view pose-only GeNT inference before insertion into the pose graph.
- Nonlocal proximity selection scans the current `local_mapping_window` on each
  backend update, filtering only already inserted directed duplicates. Do not
  reduce it to a single source keyframe scan; a candidate that fails the
  projection-distance threshold once should still be reconsidered while it
  remains inside the sliding local window.
- Nonlocal proximity edges are filtered by pose magnitude and translation
  confidence before insertion. Do not preserve low-confidence proximity edges
  merely for connectivity.
- Local PGO is a bounded marginalized smoother over 25 finalized nodes, advanced every 10
  finalized keyframes. It carries a dense FEJ Schur prior and never reopens
  marginalized nodes. Recent keyframes whose temporal GeNT measurements and
  refined depths are not finalized stay outside the active solve. After the
  stream ends, run one full-graph PGO before creating `SLAMOutput`.

Pose graph edge lifecycle:

- There is no active/finalized edge split in the current runtime graph model.
  Temporal pose edges are inserted when the source keyframe is finalized by
  multi-view aggregation. Nonlocal proximity pose edges are computed with
  one-view pose-only inference and inserted immediately.
- `PoseGraphEdges.add()` requires directed `ii/jj`, relative pose,
  source scale diagnostic, and confidence.
- Directed duplicate edges are filtered only during insertion.

Depth and scale convention:

- `depths_sens_scale` is the raw MoGe scale for the normalized monocular prior.
  It is recorded at keyframe commit and then kept fixed.
- `scale` is the current coherent node scale state used by PGO and runtime
  metric-depth consumers. It is initialized from `depths_sens_scale` at
  keyframe commit.
- `buffer.depths_sens_normed` stores the original monocular/reference
  normalized depth prior. Backend temporal finalization must not update it.
- Hard rule: every depth tensor fed into GeNT inference must be MoGe
  normalized depth from `buffer.depths_sens_normed` or the uncommitted
  `KeyframeCandidate` MoGe depth before frontend writes tracking refined depth.
  Never feed `buffer.depths` refined/published depth back into GeNT.
- MoGe inference uses `apply_mask=False`, so the raw prediction is retained at
  every pixel. During training, the predicted non-sky mask intersected with
  dataset validity defines both the MoGe-prior and GT-depth mean-normalization
  support. During inference, the predicted non-sky mask defines the prior
  support. The active support remains an explicit depth-token channel, but it
  does not zero normalized MoGe depth outside the support. The runtime cutoff
  is the maximum of a configured metric floor and the configured quantile of
  each frame's non-sky MoGe depths. This preserves the established near-depth
  support while adapting to scenes whose predicted depths shift farther away.
  Depth-derived flow bases use reciprocal raw MoGe depth at every pixel.
- Training depth regression and confidence use uniform GT-valid pixel
  weighting. The 98% depth-loss quantile filtering suppresses noisy
  supervision.
- `buffer.depths` stores the currently published/refined depth tensor. Runtime
  consumers should interpret metric depth as:

```text
metric_depth = buffer.depths * buffer.scale
```

- GeNT refined depth is expected to stay in the same MoGe-normalized source
  gauge as the depth prior. Frontend tracking writes the immediate refined
  keyframe depth for accepted keyframes, but it must not update
  `KeyframeCandidate.scale` or `depths_sens_scale`.
  Tracking pose seeding uses the original `KeyframeCandidate.scale` directly.
- Backend temporal finalization updates `buffer.depths` once for the source
  keyframe and inserts temporal pose edges, but it must not update
  `depths_sens_scale`, `scale`, or `buffer.depths_sens_normed`.
- The backend temporal depth commit is one-shot per source keyframe.
  `_finalize_temporal_keyframe()` owns this guarded write; do not split a
  second helper that can update `buffer.depths` outside that finalized-keyframe
  guard.
- PGO updates poses and `scale`, but never `depths_sens_scale`. Stored depth
  tensors are not rescaled by PGO. If PGO changes a node scale, mark that
  keyframe `depth_dirty=True` so downstream metric-depth consumers can refresh.
  Non-keyframe final pose reconstruction derives its source scale from the
  optimized reference scale and the tracking-time GeNT relative-scale
  measurement; it does not reuse the raw MoGe source/reference scale ratio.

## PGO design state

Current preferred/default PGO path is joint `se3_scale` with
`backend: cuda_eigen`. With the most recent network predictions, `se3_scale`
is more accurate than the older staged rotation-then-translation+scale path.
Keep staged PGO available as an ablation/debug mode, but do not treat it as the
default accuracy path.

Scale convention:

- `depths_sens_scale` is the fixed raw MoGe scale, while `scale` is the single
  coherent node/reference scale state optimized by PGO.
- Do not add a separate `pose_scales` buffer unless the research protocol
  changes.
- `edge_relative_scale` supplies the GeNT relative log-scale residual used by
  the joint PGO solve.
- No direct per-node scale prior is active. MoGe enters joint PGO only through
  the low-frequency NIS factor.
- No per-edge translation scale correction is active. Previous edge-scale slack
  experiments reduced residuals by hiding error and degraded trajectory
  quality.

PGO edge convention:

- Relative translation measurements are source-scale normalized:
  `translation(T_j * inv(T_i)) / exp(log_s[i]) ~= edge_relative_pose_ij[:3]`.
- Relative rotation residual is measured as:
  `Log(R_ij_meas^{-1} * R_j * R_i^{-1})`.
- Full-graph PGO fixes the complete anchor pose-scale delta. Fixed-lag PGO fixes
  node 0 only in the first window; after node 0 is marginalized, the full-rank
  Schur prior carries the gauge and no moving anchor is introduced.

Implementation layout:

- PGO code lives under `gent/runtime/slam/pgo/`.
- `optimizer.py` is the public PGO interface/dispatch entry point.
- `common.py` contains shared residuals, initialization, and result types.
- `torch_backend.py` is the dense readable/reference implementation.
- `cuda_eigen.py` is the production backend using CUDA block assembly plus
  cached CPU Eigen sparse solves.
- `marginalized_local.py` is the bounded online solver. It reuses CUDA SE3-scale blocks,
  but assembles and solves at most 25-node dense normal systems in GPU float64.
- `cuda_backend.py` contains Python bindings/wrappers for native CUDA/Eigen PGO
  helpers.
- Native extension sources live in `gent/runtime/slam/pgo/cuda/`.
- Projection geometry CUDA helpers live separately under
  `gent/runtime/slam/components/cuda/` and should not be put under the PGO
  CUDA namespace.
- Public CUDA backend selection is `backend="cuda_eigen"`. The dense
  `backend="torch"` path is kept as the reference/debug backend. Legacy public
  `cuda` and `cuda_cudss` backends are not active in this path.

CUDA backend notes:

- Profile the sparse solve/data path before changing CUDA residual/Jacobian
  block assembly; recent work found solver-side costs can dominate.
- `cuda_eigen` uses native CUDA block assembly for rotation,
  translation+scale, and SE3+scale, then solves fixed-layout CPU double Eigen
  normal systems. All systems use one cached sparse path with Natural ordering;
  joint SE3+scale uses `SimplicialLDLT`, while rotation and staged ablations use
  `SimplicialLLT`.
- Sparse joint SE3+scale normal equations are assembled deterministically on
  CUDA from cached contribution groups. Only the compressed Hessian values and
  gradient are copied to CPU for Eigen factorization. Contribution metadata is
  packed into one int64 per entry, and fixed-topology GPU/CPU normal buffers are
  reused across LM attempts. Keep the CPU assembly as the reference path used
  by the CUDA/CPU equality test.
- Low-frequency MoGe NIS uses one multi-RHS sparse factorization per LM
  attempt for the base step and all DCT mode responses. Independent RHS columns
  are solved in parallel. Cache the fixed MoGe mode precision once for repeated
  quadratic-cost evaluations; only the changing innovation covariance requires
  a small solve per iteration. Keep the fixed DCT RHS as a CPU float64 tensor;
  Eigen consumes it directly without per-iteration CUDA-to-CPU copies.
- Online marginalized local PGO always uses MoGe NIS with `K=2`, which is one
  non-DC DCT slope mode. The NIS factor is ephemeral: it participates in the
  active solve but is excluded when the Schur prior is assembled. Final sparse
  PGO continues to use `K=8` from configuration.
- Eigen solves stay in CPU double precision, but their runtime outputs are cast
  to float32 before host-to-device transfer to match the fixed CUDA float32 PGO
  protocol and avoid uploading double multi-RHS tensors.
- Each public Eigen solve assembles a fresh normal system before applying LM
  damping. Do not cache and recopy a full undamped Hessian; retries naturally
  reassemble before adding their new damping value.
- Cached Eigen solvers should reuse symbolic sparsity analysis across LM
  iterations/attempts when topology is fixed.
- The current `cuda_eigen` robust-weight flow avoids a redundant Torch/LieTorch
  residual pass: the native weighted block builder computes Huber weights,
  weighted Jacobians/residuals, and cost summaries in one CUDA pass. Keep the
  cost summary on CUDA during LM; candidate cost, current cost, and step norm
  use one scalar host transfer for the acceptance decision.
- In the full SE3+scale CUDA kernel, quaternion-to-matrix is intentional
  because `R_i` and `R_j` are reused for Jacobian blocks. Do not replace it
  with direct quaternion-vector rotation unless the Jacobian structure is
  changed.
- Translation+scale-only may use direct quaternion rotation where matrices are
  not reused.
- Avoid unnecessary `.detach()` in inference Python/C++/CUDA paths.

Evaluation:

- GT scene path on A6000:
  `datasets/TartanAir/abandonedfactory/Easy/P011`
- Useful scripts: `scripts/evaluate_tartanair_pgo.py`,
  `scripts/evaluate_tartanair_artifacts.py`,
  `scripts/ablate_pgo_moge_mode_count.py`,
  `scripts/diagnose_local_pgo_nis.py`, and
  `scripts/plot_pgo_trajectory.py`.
- Do not record one-off artifact benchmark result tables in `AGENTS.md`; keep
  those in output summaries or experiment notes.

## A6000 remote execution

Use the local helper when commands need the remote GPU/server Python environment:

```bash
bash scripts/a6000-run <command> [args...]
```

Defaults:

- SSH host: `A6000`
- Remote repo: `/data/disk_7t/tongfan/GNT`
- Conda environment: `base`
- Conda hook: `/data/disk_7t/tongfan/miniconda3/etc/profile.d/conda.sh`
- The helper prepends `${CONDA_PREFIX}/lib` to `LD_LIBRARY_PATH` after
  activation so compiled packages such as `lietorch_backends` use conda's
  `libstdc++`.
- The helper enables SSH keepalives and short-lived multiplexing by default:
  `ServerAliveInterval=30`, `ServerAliveCountMax=6`,
  `ControlMaster=auto`, and `ControlPersist=10m`, with sockets under
  `/tmp/gnt-a6000-ssh-$USER`. These settings reduce dropped idle sessions and
  repeated connection handshakes across concurrent Codex threads. Override with
  `A6000_SSH_*` environment variables if debugging raw SSH behavior.

Examples:

```bash
bash scripts/a6000-run python -V
bash scripts/a6000-run python -m pytest tests
bash scripts/a6000-run bash -lc 'python -V && nvidia-smi'
```

The helper uses `ssh -o BatchMode=yes`, so passwordless SSH must be configured for `ssh A6000`. In Codex, SSH may require sandbox approval. Local edits in `/Users/guantongfan/Documents/GNT` must be synced to the remote repo before remote test results reflect those edits.

Active sync rule: after any local Python code modification, immediately sync
the changed `.py` files to A6000 before running or trusting remote commands.
Treat the local workspace as the source of truth and verify parity for the
files involved, for example with `sha256sum`/`shasum`, before interpreting A6000
results. Remote test or smoke-test output is stale if this sync step has not
happened after the latest local Python edit.
