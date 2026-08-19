import torch
from lietorch import SE3

from .buffer import DEPTH_STATUS_REFINED, GraphBuffer
from .edge_filters import filter_local_edge_measurement
from .factor_graph import FactorGraph
from .frontend import TrackingGroup
from .measurements import GeNTMeasurements


class SLAMBackend:
    """Local graph construction, depth refinement, and pose-graph optimization."""

    def __init__(
        self,
        measurements: GeNTMeasurements,
        buffer: GraphBuffer,
        graph: FactorGraph,
        tracking_groups: dict[int, TrackingGroup],
        args,
        device: torch.device,
    ):
        self.measurements = measurements
        self.buffer = buffer
        self.device = device
        self.graph = graph
        self.tracking_groups = tracking_groups

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

    @torch.no_grad()
    def _temporal_inputs(
        self,
        source: int,
        neighbors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        past_count = int((neighbors < source).sum().item())
        past_neighbors = neighbors[:past_count]
        future_neighbors = neighbors[past_count:]

        if source == 0:
            source_group = None
        else:
            source_group = self.tracking_groups.pop(source)
            assert torch.equal(source_group.neighbors, past_neighbors.flip(0))

        future_pose = []
        future_log_scale = []
        for future in future_neighbors.tolist():
            future_group = self.tracking_groups[future]
            edge_index = future - source - 1
            assert int(future_group.neighbors[edge_index].item()) == source
            future_pose.append(future_group.relative_pose[edge_index])
            future_log_scale.append(future_group.relative_log_scale[edge_index])

        if future_neighbors.numel() == 0:
            return (
                source_group.flow.flip(0),
                source_group.info.flip(0),
                source_group.relative_pose.flip(0),
            )

        future_flow, future_info = self.measurements.predict_flow(
            self.buffer.fmaps[source : source + 1, 0],
            self.buffer.fmaps[future_neighbors, 0],
            self.buffer.bases[future_neighbors, 0],
        )
        future_pose = SE3(torch.stack(future_pose)).inv().data
        future_log_scale = torch.stack(future_log_scale)
        future_pose[:, :3] *= torch.exp(-future_log_scale)[:, None]
        if source == 0:
            return future_flow, future_info, future_pose

        return (
            torch.cat((source_group.flow.flip(0), future_flow), dim=0),
            torch.cat((source_group.info.flip(0), future_info), dim=0),
            torch.cat((source_group.relative_pose.flip(0), future_pose), dim=0),
        )

    def _finalize_temporal_keyframe(self, source: int) -> None:
        if source in self._temporal_finalized:
            return

        neighbors = self._temporal_neighbors(source)
        if neighbors.numel() == 0:
            return

        flow, info, relative_pose_prior = self._temporal_inputs(source, neighbors)
        ii = torch.full_like(neighbors, int(source))
        source_intrinsics = self.buffer.intrinsics[source, 0]
        target_intrinsics = self.buffer.intrinsics[neighbors, 0]
        result = self.measurements.predict_multiview(
            ii,
            neighbors,
            flow=flow,
            info=info,
            source_depth=self.buffer.depths_sens_normed[source, 0].float(),
            source_mask=self.buffer.non_sky_masks[source, 0],
            source_depth_token=self.buffer.depth_tokens[source, 0].float(),
            target_depth_token=self.buffer.depth_tokens[neighbors, 0].float(),
            source_intrinsics=source_intrinsics,
            target_intrinsics=target_intrinsics,
            decode_depth=True,
            relative_pose_prior=relative_pose_prior,
        )
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

        flow, info = self.measurements.predict_flow(
            self.buffer.fmaps[ii, 0],
            self.buffer.fmaps[jj, 0],
            self.buffer.bases[jj, 0],
        )
        result = self.measurements.predict_pairwise_pose(
            ii,
            jj,
            flow=flow,
            info=info,
            source_depth_token=self.buffer.depth_tokens[ii, 0].float(),
            target_depth_token=self.buffer.depth_tokens[jj, 0].float(),
            source_intrinsics=self.buffer.intrinsics[ii, 0],
            target_intrinsics=self.buffer.intrinsics[jj, 0],
            image_shape=(self.buffer.height, self.buffer.width),
        )
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
