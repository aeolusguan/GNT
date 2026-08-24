# GeNT Streaming Local Mapping

The default SLAM runtime is a streaming local mapping pipeline centered on
GeNT depth and local pose consistency. It no longer runs the previous one-pass
initializer, offline frontend sweep, or full-sequence backend proximity pass.

## Runtime Flow

For each incoming frame, `SLAMSystem` builds an uncommitted
`KeyframeCandidate` containing GeNT features, MoGe-normalized depth, the fixed
raw MoGe scale, its non-sky normalization mask, a cached depth/mask patch token,
and flow bases. The patch embedders allocate `D/4` to normalized depth plus
validity and `3D/4` to motion.

The first frame is accepted directly. Every later frame runs GeNT tracking
aggregation against up to `local_mapping_radius` previous keyframes.
The resulting `current -> last` relative pose seeds the frame pose. Coarse flow
magnitude only decides whether the frame becomes a keyframe; tracking-stage
edges are never inserted into the pose graph.

An accepted keyframe immediately publishes its tracking-refined depth. Once it
has `local_mapping_radius` later keyframes, the backend performs its single
temporal finalization. It builds a source-centered camera prior from the direct
frontend tracking measurements, tokenizes the source group, and runs one
CameraEnc-conditioned pose-depth pass. The backend then replaces the published
depth once and inserts the finalized temporal pose and relative log-scale
measurements. Nonlocal proximity measurements use one-view pose-only inference
with learned camera tokens and the same graph edge store.

Every 10 finalized keyframes, marginalized local PGO optimizes the latest 25
finalized pose-scale nodes. A dense float64 Schur prior carries GeNT edge
information between windows. Local MoGe NIS uses one non-DC DCT slope mode
(`K=2`) only during the active solve and is not written into the Schur prior.
The sequence ends with one sparse full-graph PGO using the configured final NIS
mode count, currently `K=8`.

`depths_sens_scale` remains the raw MoGe scale. PGO updates the coherent
`scale` state and never rescales stored depth tensors. Consumers should use:

```text
metric_depth = stored_depth * scale
```

When PGO changes a node scale, the corresponding depth is marked dirty.

## Camera Intrinsics

Each selected video frame carries its own pinhole calibration row
`[fx, fy, cx, cy]`. Resize and crop operations transform that row before the
frame enters GeNT. The keyframe buffer preserves the transformed calibration
for every accepted frame, so frontend tracking, backend temporal aggregation,
nonlocal measurements, and projection-distance filtering all use the actual
source and target cameras. A fixed `[4]` calibration in the inference config is
expanded once to `[F, 4]` as a convenience for fixed-camera sequences.

`SLAMOutput.intrinsics` is keyframe-aligned with shape `[N, 4]`, while
`SLAMOutput.frame_intrinsics` is aligned with every processed frame and has
shape `[F, 4]`. Both are restored to the original input image coordinates.

## Edge Convention

Pose graph measurements are directed and source-scale-normalized:

```text
translation(T_j * inv(T_i)) / scale_i ~= edge_relative_pose_ij[:3]
```

`edge_relative_log_scale` is the GeNT measurement for
`log(scale_j) - log(scale_i)` and is an active PGO residual.

Saved pose artifacts keep the stable NPZ keys:

- `edge_ii`
- `edge_jj`
- `edge_relative_pose`
- `edge_relative_log_scale`
- `edge_confidence`

The Python runtime object names this store `SLAMOutput.pose_edges`.

## Camera Prior Reuse

Accepted keyframes retain their frontend tracking group until temporal
finalization. The group contains the neighbor IDs, flow information, predicted
relative poses, and predicted relative log scales. Backend aggregation:

1. Uses the source tracking group directly for past-neighbor pose priors and
   flow.
2. Inverts each future neighbor's direct tracking pose. If
   `l = log(scale_source) - log(scale_future)`, the inverse translation is
   multiplied by `exp(-l)` to enter the source normalized-depth gauge.
3. Computes the required source-to-future flow, builds the backend patch tokens,
   and runs one CameraEnc-conditioned pose-depth solver pass.

Future flow is directional and is therefore not obtained by inverting the
frontend flow. CameraEnc output directly replaces the learned camera tokens;
the two token types are never mixed. Frontend tracking and nonlocal pair
measurements continue to use one learned-token pass. A runtime checkpoint must
therefore contain the trained `cam_enc.*` weights.

## Key Configuration

```yaml
local_edge_max_rotation_deg: 30.0
local_edge_max_translation: 5.0
keyframe_motion_thresh: 20.0
local_mapping_window: 25
local_mapping_radius: 2
local_mapping_nms: 1
local_mapping_thresh: 16.0
local_pgo_every: 10
local_edge_outlier_trans_conf_thresh: 0.1
pgo_mode: se3_scale
pgo_backend: cuda_eigen
pgo_moge_mode_nis: true
pgo_moge_mode_count: 8
pgo_moge_mode_nis_cutoff: 0.01
```
