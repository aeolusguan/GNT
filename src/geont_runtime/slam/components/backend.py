import torch

from .buffer import DEPTH_STATUS_REFINED, GraphBuffer
from .edge_filters import (
    filter_local_edge_measurement,
    make_local_edge_outlier_info,
    record_local_edge_outlier_info,
)
from .factor_graph import FactorGraph
from .measurements import GeoNTMeasurements


class SLAMBackend:
    """Local graph construction, depth refinement, and pose-graph optimization."""

    def __init__(
        self,
        measurements: GeoNTMeasurements,
        buffer: GraphBuffer,
        graph: FactorGraph,
        args,
        device: torch.device,
    ):
        self.measurements = measurements
        self.buffer = buffer
        self.device = device
        self.graph = graph

        self.local_edge_outlier_trans_conf_thresh = float(args.local_edge_outlier_trans_conf_thresh)
        self.local_edge_max_rotation_deg = float(args.get("local_edge_max_rotation_deg", 30.0))
        self.local_edge_max_translation = float(args.get("local_edge_max_translation", 5.0))
        self.local_mapping_window = int(args.get("local_mapping_window", 25))
        self.local_mapping_radius = int(args.get("local_mapping_radius", 2))
        self.local_mapping_nms = int(args.get("local_mapping_nms", 1))
        self.local_mapping_thresh = float(args.get("local_mapping_thresh", 16.0))
        self.local_pgo_every = int(args.get("local_pgo_every", 1))
        self.pgo_iters = args.pgo_iters
        self.pgo_damping = args.pgo_damping
        self.pgo_lm_max_attempts = args.pgo_lm_max_attempts
        self.pgo_huber_delta = args.pgo_huber_delta
        self.pgo_scale_conf = args.pgo_scale_conf
        self.pgo_mode = args.pgo_mode
        self.pgo_rotation_only = args.pgo_rotation_only
        self.pgo_backend = args.pgo_backend

        self.local_edge_outlier_info = make_local_edge_outlier_info(self.local_edge_outlier_trans_conf_thresh)
        self.backend_info = {
            "local_mapping_radius": self.local_mapping_radius,
            "pose_edges_added": 0,
            "temporal_finalize_runs": 0,
            "temporal_edges_added": 0,
            "proximity_edges_added": 0,
            "local_pgo_runs": 0,
            "global_pgo_runs": 0,
        }
        self._temporal_finalized: set[int] = set()

    def _record_local_edge_outlier_info(self, info: dict, duplicate_filtered_after_gate: int) -> None:
        record_local_edge_outlier_info(self.local_edge_outlier_info, info, duplicate_filtered_after_gate)
        self.graph.edges.pgo_info.update(self.local_edge_outlier_info)

    def _temporal_neighbors(self, source: int) -> torch.Tensor:
        start = max(0, int(source) - self.local_mapping_radius)
        end = min(self.buffer.n_frames, int(source) + self.local_mapping_radius + 1)
        neighbors = torch.arange(start, end, device=self.device, dtype=torch.long)
        return neighbors[neighbors != int(source)]

    def _finalize_temporal_keyframe(self, source: int) -> None:
        source = int(source)
        if source in self._temporal_finalized:
            return

        neighbors = self._temporal_neighbors(source)
        if neighbors.numel() == 0:
            return

        result = self.measurements.measure_multiview_pose_depth(source, neighbors)
        refined_depth = result["refined_depth"].float()
        mask = self.buffer.non_sky_masks[source, 0]
        if mask.any():
            source_mean = self.buffer.depths_sens_normed[source, 0].float()[mask].mean().clamp_min(1e-6)
            refined_mean = refined_depth[mask].mean().clamp_min(1e-6)
            new_base_scale = self.buffer.depths_sens_scale[source, 0] * (source_mean / refined_mean)
            self.buffer.depths_sens_scale[source, 0] = new_base_scale
            self.buffer.pgo_base_scale[source, 0] = new_base_scale
        self.buffer.depths[source, 0] = refined_depth.to(dtype=self.buffer.depths.dtype)
        self.buffer.depth_status[source] = DEPTH_STATUS_REFINED
        self.buffer.depth_dirty[source] = True
        added = self.graph.add_measurement_result(result)
        n_added = int(added["n_added"])
        self.backend_info["temporal_finalize_runs"] += 1
        self.backend_info["temporal_edges_added"] += n_added
        self.backend_info["pose_edges_added"] += n_added
        self._temporal_finalized.add(source)

    def _add_nonlocal_proximity_edges(self, current_keyframe: int, finalized_source_end: int) -> None:
        window_start = max(0, current_keyframe + 1 - self.local_mapping_window)
        if finalized_source_end <= window_start:
            return

        ii, jj = self.graph.select_nonlocal_proximity_edges(
            window_start,
            window_start,
            int(finalized_source_end),
            radius=self.local_mapping_radius,
            nms=self.local_mapping_nms,
            thresh=self.local_mapping_thresh,
        )
        if ii.numel() == 0:
            return

        result = self.measurements.predict_pose_edges(ii, jj)
        result = filter_local_edge_measurement(
            result,
            max_rotation_deg=self.local_edge_max_rotation_deg,
            max_translation=self.local_edge_max_translation,
            trans_conf_thresh=self.local_edge_outlier_trans_conf_thresh,
        )
        result = self.graph.add_measurement_result(result)
        self._record_local_edge_outlier_info(
            result.get(
                "local_edge_outlier_info",
                make_local_edge_outlier_info(self.local_edge_outlier_trans_conf_thresh),
            ),
            int(result["n_duplicates"]),
        )
        n_added = int(result["n_added"])
        self.backend_info["proximity_edges_added"] += n_added
        self.backend_info["pose_edges_added"] += n_added

    def update_local_graph(self, current_keyframe: int) -> None:
        if current_keyframe <= 0:
            return

        mature_source = int(current_keyframe) - self.local_mapping_radius
        if mature_source < 0:
            return

        self._finalize_temporal_keyframe(mature_source)
        self._add_nonlocal_proximity_edges(current_keyframe, finalized_source_end=mature_source + 1)

    def finalize_pending_keyframes(self) -> None:
        for source in range(self.buffer.n_frames):
            self._finalize_temporal_keyframe(source)
        self._add_nonlocal_proximity_edges(self.buffer.n_frames - 1, finalized_source_end=self.buffer.n_frames)

    def optimize_local_window(self, current_keyframe: int) -> None:
        if self.local_pgo_every <= 0:
            return
        if (current_keyframe + 1) % self.local_pgo_every != 0:
            return

        window_start = max(0, current_keyframe + 1 - self.local_mapping_window)
        window_end = current_keyframe + 1
        old_scales = self.buffer.depths_sens_scale[window_start:window_end, 0].clone()
        result = self.graph.optimize_pose_graph_window(
            window_start,
            window_end,
            n_iters=self.pgo_iters,
            damping=self.pgo_damping,
            lm_max_attempts=self.pgo_lm_max_attempts,
            huber_delta=self.pgo_huber_delta,
            scale_conf=10 * self.pgo_scale_conf,
            mode="rotation_only" if self.pgo_rotation_only else self.pgo_mode,
            backend=self.pgo_backend,
        )
        if result is None:
            return

        self.backend_info["local_pgo_runs"] += 1
        new_scales = self.buffer.depths_sens_scale[window_start:window_end, 0]
        changed = (torch.log(new_scales.clamp_min(1e-6)) - torch.log(old_scales.clamp_min(1e-6))).abs() > 1e-5
        if changed.any():
            indices = torch.arange(window_start, window_end, device=self.buffer.device)[changed]
            self.buffer.depth_dirty[indices] = True
        self.graph.edges.pgo_info.update(self.backend_info)
        self.graph.edges.pgo_info.update(self.local_edge_outlier_info)

    def optimize_full_graph(self) -> None:
        old_scales = self.buffer.depths_sens_scale[: self.buffer.n_frames, 0].clone()
        result = self.graph.optimize_pose_graph(
            anchor=0,
            n_iters=self.pgo_iters,
            damping=self.pgo_damping,
            lm_max_attempts=self.pgo_lm_max_attempts,
            huber_delta=self.pgo_huber_delta,
            scale_conf=self.pgo_scale_conf,
            mode="rotation_only" if self.pgo_rotation_only else self.pgo_mode,
            backend=self.pgo_backend,
            extra_info={"scope": "streaming_final_full_graph"},
        )
        if result is None:
            return

        self.backend_info["global_pgo_runs"] += 1
        new_scales = self.buffer.depths_sens_scale[: self.buffer.n_frames, 0]
        changed = (torch.log(new_scales.clamp_min(1e-6)) - torch.log(old_scales.clamp_min(1e-6))).abs() > 1e-5
        if changed.any():
            indices = torch.arange(self.buffer.n_frames, device=self.buffer.device)[changed]
            self.buffer.depth_dirty[indices] = True
        self.graph.edges.pgo_info.update(self.backend_info)
        self.graph.edges.pgo_info.update(self.local_edge_outlier_info)
