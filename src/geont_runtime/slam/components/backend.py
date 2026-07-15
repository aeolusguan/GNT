import torch

from .buffer import DEPTH_STATUS_REFINED, GraphBuffer
from .edge_filters import filter_local_edge_measurement
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
        self.local_edge_max_rotation_deg = float(args.local_edge_max_rotation_deg)
        self.local_edge_max_translation = float(args.local_edge_max_translation)
        self.local_mapping_window = int(args.local_mapping_window)
        self.local_mapping_radius = int(args.local_mapping_radius)
        self.local_mapping_nms = int(args.local_mapping_nms)
        self.local_mapping_thresh = float(args.local_mapping_thresh)
        self.local_pgo_every = int(args.local_pgo_every)
        self.pgo_iters = args.pgo_iters
        self.pgo_damping = args.pgo_damping
        self.pgo_lm_max_attempts = args.pgo_lm_max_attempts
        self.pgo_huber_delta = args.pgo_huber_delta
        self.pgo_mode = args.pgo_mode
        self.pgo_backend = args.pgo_backend
        self.pgo_moge_mode_nis = bool(args.pgo_moge_mode_nis)
        self.pgo_moge_mode_count = int(args.pgo_moge_mode_count)
        self.pgo_moge_mode_nis_cutoff = float(args.pgo_moge_mode_nis_cutoff)

        self._temporal_finalized: set[int] = set()
        self.local_pgo_runs = 0
        self.local_pgo_runtime_sec = 0.0
        self.local_pgo_last_alpha = 0.0

    def _temporal_neighbors(self, source: int) -> torch.Tensor:
        start = max(0, int(source) - self.local_mapping_radius)
        end = min(self.buffer.n_frames, int(source) + self.local_mapping_radius + 1)
        neighbors = torch.arange(start, end, device=self.device, dtype=torch.long)
        return neighbors[neighbors != int(source)]

    def _finalize_temporal_keyframe(self, source: int) -> None:
        if source in self._temporal_finalized:
            return

        neighbors = self._temporal_neighbors(source)
        if neighbors.numel() == 0:
            return

        result = self.measurements.predict_pose_depth(source, neighbors)
        refined_depth = result.refined_depth.float()
        self.buffer.depths[source, 0] = refined_depth.to(dtype=self.buffer.depths.dtype)
        self.buffer.depth_status[source] = DEPTH_STATUS_REFINED
        self.buffer.depth_dirty[source] = True
        self.graph.add_measurement_result(result)
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
        self.graph.add_measurement_result(result)

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

    def optimize_local_pgo(self, finalized_end: int) -> None:
        if self.local_pgo_every <= 0:
            return
        if finalized_end % self.local_pgo_every != 0:
            return

        window_start = max(0, finalized_end - self.local_mapping_window)
        old_scales = self.buffer.scale[window_start:finalized_end, 0].clone()
        result = self.graph.optimize_local_pgo(
            finalized_end,
            self.local_mapping_window,
            n_iters=self.pgo_iters,
            damping=self.pgo_damping,
            lm_max_attempts=self.pgo_lm_max_attempts,
            huber_delta=self.pgo_huber_delta,
            moge_mode_nis_cutoff=self.pgo_moge_mode_nis_cutoff,
        )
        if result is None:
            return

        self.local_pgo_runs += 1
        self.local_pgo_runtime_sec += float(result.info["runtime_sec"])
        self.local_pgo_last_alpha = float(result.info["moge_mode_alpha"])
        new_scales = self.buffer.scale[window_start:finalized_end, 0]
        changed = (torch.log(new_scales.clamp_min(1e-6)) - torch.log(old_scales.clamp_min(1e-6))).abs() > 1e-5
        if changed.any():
            indices = torch.arange(window_start, finalized_end, device=self.buffer.device)[changed]
            self.buffer.depth_dirty[indices] = True

    def optimize_full_graph(self) -> None:
        old_scales = self.buffer.scale[: self.buffer.n_frames, 0].clone()
        result = self.graph.optimize_pose_graph(
            anchor=0,
            n_iters=self.pgo_iters,
            damping=self.pgo_damping,
            lm_max_attempts=self.pgo_lm_max_attempts,
            huber_delta=self.pgo_huber_delta,
            mode=self.pgo_mode,
            backend=self.pgo_backend,
            moge_mode_nis=self.pgo_moge_mode_nis,
            moge_mode_count=self.pgo_moge_mode_count,
            moge_mode_nis_cutoff=self.pgo_moge_mode_nis_cutoff,
        )
        if result is None:
            return

        result.info.update(
            local_pgo_runs=self.local_pgo_runs,
            local_pgo_runtime_sec=self.local_pgo_runtime_sec,
            local_pgo_last_moge_mode_alpha=self.local_pgo_last_alpha,
        )
        self.graph.edges.pgo_info = result.info

        new_scales = self.buffer.scale[: self.buffer.n_frames, 0]
        changed = (torch.log(new_scales.clamp_min(1e-6)) - torch.log(old_scales.clamp_min(1e-6))).abs() > 1e-5
        if changed.any():
            indices = torch.arange(self.buffer.n_frames, device=self.buffer.device)[changed]
            self.buffer.depth_dirty[indices] = True
