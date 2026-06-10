import torch

from geont.models import GeoNTWrapper
from geont_runtime.utils.logging import pbar
from .buffer import GraphBuffer
from .factor_graph import InitializationFactorGraph, PoseGraphEdges
from geont_runtime.slam.pgo import optimize_sim3_pose_graph
from geont_runtime.slam.pgo.replay import make_pgo_replay_graph


class OnePassInitializer:
    def __init__(self, net: GeoNTWrapper, video: GraphBuffer, args, device: torch.device):
        self.video = video
        self.graph = InitializationFactorGraph(net, video, device)
        self.edges = PoseGraphEdges(device)
        assert args.warmup >= 2
        self.warmup = args.warmup
        self.frontend_radius = args.frontend_radius
        self.frontend_thresh = args.frontend_thresh
        self.seq_init = args.seq_init
        self.pgo_iters = args.pgo_iters
        self.pgo_damping = args.pgo_damping
        self.pgo_lm_max_attempts = args.pgo_lm_max_attempts
        self.pgo_huber_delta = args.pgo_huber_delta
        self.pgo_scale_conf = args.pgo_scale_conf
        self.pgo_mode = args.pgo_mode
        self.pgo_rotation_only = args.pgo_rotation_only
        self.pgo_backend = args.pgo_backend
        self.use_fp16 = args.use_fp16

    def run(self):
        """Add initialization edges for the most recent keyframe window."""
        if self.video.n_frames < self.warmup:
            return

        self.graph.add_neighborhood_factors(
            max(self.video.n_frames - self.warmup, 0),
            self.video.n_frames,
            r=1 if self.seq_init else self.frontend_radius,
            thresh=self.frontend_thresh,
        )

    def marginalize_oldest_keyframe(self):
        keyframe = self.graph.oldest_keyframe()
        if keyframe is None:
            return None

        result = self.graph.marginalize_keyframe(keyframe, use_fp16=self.use_fp16)
        if result is not None:
            self.edges.add(
                result["ii"],
                result["jj"],
                result["relative_pose"],
                result["relative_scale"],
                result["confidence"],
            )
        return result

    def finalize(self):
        """Finalize depth, relative poses, and global keyframe poses."""
        n_to_marginalize = self.graph.active_keyframe_count()
        for _ in pbar(range(n_to_marginalize), desc="Finalizing initializer"):
            if self.marginalize_oldest_keyframe() is None:
                break

        if self.edges.ii.numel() == 0:
            return None

        initial_log_scales = torch.log(self.video.depths_sens_scale[: self.video.n_frames, 0].clamp_min(1e-6))
        result = optimize_sim3_pose_graph(
            n_nodes=self.video.n_frames,
            ii=self.edges.ii,
            jj=self.edges.jj,
            rel_poses=self.edges.relative_pose,
            rel_scales=self.edges.relative_scale,
            edge_conf=self.edges.confidence,
            initial_log_scales=initial_log_scales,
            anchor=0,
            n_iters=self.pgo_iters,
            damping=self.pgo_damping,
            lm_max_attempts=self.pgo_lm_max_attempts,
            huber_delta=self.pgo_huber_delta,
            scale_conf=self.pgo_scale_conf,
            mode="rotation_only" if self.pgo_rotation_only else self.pgo_mode,
            backend=self.pgo_backend,
        )
        assert result.initial_poses is not None
        self.edges.pgo_replay = make_pgo_replay_graph(
            n_nodes=self.video.n_frames,
            anchor=0,
            ii=self.edges.ii,
            jj=self.edges.jj,
            relative_pose=self.edges.relative_pose,
            relative_scale=self.edges.relative_scale,
            confidence=self.edges.confidence,
            initial_poses=result.initial_poses,
            initial_log_scales=initial_log_scales,
        )
        self.video.poses[: self.video.n_frames] = result.poses.to(dtype=self.video.poses.dtype)
        new_scales = torch.exp(result.log_scales).to(dtype=self.video.depths_sens_scale.dtype)
        self.video.depths_sens_scale[: self.video.n_frames, 0] = new_scales
        self.edges.pgo_info = result.info
        return result
