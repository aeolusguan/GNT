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
        self.local_mapping_window = int(args.get("local_mapping_window", 8))
        self.local_mapping_radius = int(args.get("local_mapping_radius", 2))
        self.local_mapping_nms = int(args.get("local_mapping_nms", 1))
        self.local_mapping_thresh = float(args.get("local_mapping_thresh", 16.0))
        self.local_pgo_every = int(args.get("local_pgo_every", 1))
        self.local_refine_depth = bool(args.get("local_refine_depth", True))
        self.depth_refine_observability_topk = int(args.get("depth_refine_observability_topk", 4))
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
            "depth_refine_observability_topk": self.depth_refine_observability_topk,
            "depth_observability_selected_neighbor_count": 0,
            "depth_observability_selected_neighbor_count_min": 0,
            "depth_observability_selected_neighbor_count_mean": 0.0,
            "depth_observability_selected_neighbor_count_max": 0,
            "pose_edges_added": 0,
            "depth_refine_runs": 0,
            "local_pgo_runs": 0,
            "global_pgo_runs": 0,
        }
        self._selected_neighbor_count_sum = 0
        self._selected_neighbor_count_min: int | None = None
        self._selected_neighbor_count_max = 0

    def _record_selected_neighbor_count(self, count: int) -> None:
        count = int(count)
        self.backend_info["depth_observability_selected_neighbor_count"] += 1
        self._selected_neighbor_count_sum += count
        self._selected_neighbor_count_min = count if self._selected_neighbor_count_min is None else min(
            self._selected_neighbor_count_min,
            count,
        )
        self._selected_neighbor_count_max = max(self._selected_neighbor_count_max, count)
        n = self.backend_info["depth_observability_selected_neighbor_count"]
        self.backend_info["depth_observability_selected_neighbor_count_min"] = int(self._selected_neighbor_count_min)
        self.backend_info["depth_observability_selected_neighbor_count_mean"] = float(
            self._selected_neighbor_count_sum / n
        )
        self.backend_info["depth_observability_selected_neighbor_count_max"] = int(self._selected_neighbor_count_max)

    def _record_local_edge_outlier_info(self, info: dict, duplicate_filtered_after_gate: int) -> None:
        record_local_edge_outlier_info(self.local_edge_outlier_info, info, duplicate_filtered_after_gate)
        self.graph.edges.pgo_info.update(self.local_edge_outlier_info)

    def update_local_graph(self, current_keyframe: int) -> torch.Tensor:
        if current_keyframe <= 0:
            return torch.as_tensor([], dtype=torch.long, device=self.graph.device)

        window_start = max(0, current_keyframe + 1 - self.local_mapping_window)
        ii, jj = self.graph.select_proximity_edges(
            window_start,
            window_start,
            current_keyframe + 1,
            radius=self.local_mapping_radius,
            nms=self.local_mapping_nms,
            thresh=self.local_mapping_thresh,
        )
        if ii.numel() == 0:
            return ii

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
        self.backend_info["pose_edges_added"] += int(result["n_added"])
        return result["changed_sources"]

    def _commit_refined_depth(self, keyframe: int, refined_depth: torch.Tensor) -> None:
        refined_depth = refined_depth.float()
        mask = self.buffer.non_sky_masks[keyframe, 0]
        depth_scale = torch.ones((), device=refined_depth.device, dtype=refined_depth.dtype)
        if mask.any():
            source_mean = self.buffer.depths[keyframe, 0].float()[mask].mean().clamp_min(1e-6)
            refined_mean = refined_depth[mask].mean().clamp_min(1e-6)
            depth_scale = source_mean / refined_mean
        self.buffer.depths[keyframe, 0] = (refined_depth * depth_scale).to(dtype=self.buffer.depths.dtype)
        self.buffer.depth_status[keyframe] = DEPTH_STATUS_REFINED
        self.buffer.depth_dirty[keyframe] = True

    def refine_changed_sources(self, sources: torch.Tensor, current_keyframe: int) -> None:
        if not self.local_refine_depth or sources.numel() == 0:
            return

        window_start = max(0, current_keyframe + 1 - self.local_mapping_window)
        window_end = current_keyframe + 1
        for source in torch.unique(sources).cpu().tolist():
            neighbors = self.graph.edge_neighbors(int(source), window_start=window_start, window_end=window_end)
            selected_neighbors = self.graph.edges.select_depth_observability_topk(
                int(source),
                neighbors,
                self.depth_refine_observability_topk,
            )
            refined = self.measurements.refine_depth(
                int(source),
                selected_neighbors,
            )
            if refined is None:
                continue
            self.graph.edges.update_depth_observability(
                torch.full_like(selected_neighbors, int(source)),
                selected_neighbors,
                refined["observability_logits"],
                rank=2,
            )
            self._commit_refined_depth(int(source), refined["refined_depth"])
            self._record_selected_neighbor_count(int(selected_neighbors.numel()))
            self.backend_info["depth_refine_runs"] += 1

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
            scale_conf=self.pgo_scale_conf,
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
