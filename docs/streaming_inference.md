# GeoNT Streaming Local Mapping

The default SLAM runtime is a streaming local mapping pipeline centered on
GeoNT depth and local pose consistency. It no longer runs the previous one-pass
initializer, offline frontend sweep, or full-sequence backend proximity pass.

## Runtime Flow

For each incoming frame, `SLAMSystem` builds an uncommitted
`KeyframeCandidate` containing GeoNT features, MoGe-normalized depth, the fixed
raw MoGe scale, a valid-depth mask, and flow bases.

The first frame is accepted directly. Every later frame runs GeoNT tracking
aggregation against up to `tracking_multiview_neighbors` previous keyframes.
The resulting `current -> last` relative pose seeds the frame pose. Coarse flow
magnitude only decides whether the frame becomes a keyframe; tracking-stage
edges are never inserted into the pose graph.

An accepted keyframe immediately publishes its tracking-refined depth. Once it
has `local_mapping_radius` later keyframes, the backend performs its single
temporal finalization: it runs one GeoNT multi-view aggregation, replaces the
published depth once, and inserts the finalized temporal pose and relative
log-scale measurements. Nonlocal proximity measurements use one-view pose-only
inference and the same graph edge store.

Every 10 finalized keyframes, marginalized local PGO optimizes the latest 25
finalized pose-scale nodes. A dense float64 Schur prior carries GeoNT edge
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

## Edge Convention

Pose graph measurements are directed and source-scale-normalized:

```text
translation(T_j * inv(T_i)) / scale_i ~= edge_relative_pose_ij[:3]
```

`edge_relative_log_scale` is the GeoNT measurement for
`log(scale_j) - log(scale_i)` and is an active PGO residual.

Saved pose artifacts keep the stable NPZ keys:

- `edge_ii`
- `edge_jj`
- `edge_relative_pose`
- `edge_relative_log_scale`
- `edge_confidence`

The Python runtime object names this store `SLAMOutput.pose_edges`.

## Camera Token Prior

GeoNT accepts an optional `camera_prior` that is already encoded as backbone
camera tokens with shape `[num_edges, 1, embed_dim]`. The model does not encode
raw relative poses and the camera decoder still predicts pose from network
features. This interface is reserved for future training or finetuning that
initializes camera tokens from local pose priors.

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
tracking_multiview_neighbors: 2
pgo_mode: se3_scale
pgo_backend: cuda_eigen
pgo_moge_mode_nis: true
pgo_moge_mode_count: 8
pgo_moge_mode_nis_cutoff: 0.01
```
