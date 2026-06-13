# GeoNT Streaming Local Mapping

The default SLAM runtime is a streaming local mapping pipeline centered on
GeoNT depth and local pose consistency. It no longer runs the previous one-pass
initializer, offline frontend sweep, or full-sequence backend proximity pass.

## Runtime Flow

For each incoming frame, `SLAMSystem` first builds a `KeyframeCandidate` with
GeoNT feature maps, normalized MoGe depth, non-sky mask, source scale, and flow
bases. The keyframe candidate is not written into `GraphBuffer` until tracking
accepts it.

The first frame is accepted directly with monocular depth. Later frames first
run the same coarse optical-flow motion check used by the tracking seed:
`flow_init` predicts dense flow from the last accepted keyframe to the keyframe
candidate, and the mean flow magnitude is compared against
`keyframe_motion_thresh`. Low-motion frames are skipped before multi-view
bootstrap unless the final frame is forced.

Frames that pass the coarse-flow motion check are bootstrapped before commit.
`SLAMBackend` treats the candidate as the source keyframe and runs GeoNT
multi-view aggregation against the newest available accepted keyframes, up to
`depth_bootstrap_neighbors`. The second keyframe uses one previous keyframe;
the third and later keyframes use two by default. Bootstrap writes refined
candidate depth in decoder output scale and updates the candidate source scale
by `source_depth_mean / refined_depth_mean`.

After bootstrap, the keyframe candidate is committed to `GraphBuffer`.
`SLAMBackend` inserts bootstrap pose edges `current -> previous`. The
`current -> last` edge bypasses local pose/confidence gates and seeds the new
keyframe pose from the previous keyframe. Additional bootstrap edges, such as
`current -> last-1`, use the normal local-edge gates.

After a keyframe is accepted, `SLAMBackend` asks `FactorGraph` for local
proximity candidates, runs GeoNT one-view pose-only inference for those
candidates, filters local pose edges by pose magnitude and local-edge confidence,
and inserts accepted measurements directly into the pose graph. There is no
queued edge state and no marginalization pass.

Sources touched by newly inserted local outgoing edges are then refined with
multi-view depth aggregation over their explicit local neighbor set. During
SLAM inference, the raw pre-softmax depth-head gate logits are averaged per
neighbor and the top `depth_refine_observability_topk` neighbors are used for
depth aggregation. Depth refinement writes refined depth in decoder output
scale, updates the source scale by `source_depth_mean / refined_depth_mean`,
sets `depth_status=REFINED`, and marks `depth_dirty=True`. It does not add or
update pose graph edges.

Local PGO runs on the recent window using the already inserted pose edges. PGO
updates local poses and `depths_sens_scale`; stored depth tensors are not
rescaled by PGO. Consumers should interpret metric depth as:

```text
metric_depth = stored_depth * depths_sens_scale
```

When PGO changes a node scale, the corresponding depth is marked dirty without
rescaling the stored depth map.

## Edge Convention

Pose graph measurements are directed and source-scale-normalized:

```text
translation(T_j * inv(T_i)) / scale_i ~= edge_relative_pose_ij[:3]
```

`edge_relative_scale` stores the source keyframe scale at edge insertion time as
a diagnostic/artifact field. PGO optimizes node scales through
`depths_sens_scale`; edge scale is not a hard residual target.

Saved pose artifacts keep the stable NPZ keys:

- `edge_ii`
- `edge_jj`
- `edge_relative_pose`
- `edge_relative_scale`
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
keyframe_motion_thresh: 2.5
local_mapping_window: 8
local_mapping_radius: 2
local_mapping_nms: 1
local_mapping_thresh: 16.0
local_pgo_every: 1
local_refine_depth: true
local_edge_outlier_trans_conf_thresh: 0.1
depth_bootstrap_neighbors: 2
depth_refine_observability_topk: 4
pgo_mode: se3_scale
pgo_backend: cuda_eigen
```
