from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from .factor_graph import FactorGraph


def _natural_sort_key(path: Path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


@dataclass(frozen=True)
class LoopCandidate:
    ii: int
    jj: int
    score: float = 1.0


@dataclass(frozen=True)
class LoopMeasurement:
    ii: int
    jj: int
    relative_pose: torch.Tensor
    n_matches: int
    score: float = 1.0


def resolve_keyframe_image_paths(scene: Path, image_subdir: str, timestamps: Sequence[int]) -> list[Path]:
    image_dir = scene / image_subdir
    if not image_dir.exists():
        raise FileNotFoundError(f"loop-closure image directory not found: {image_dir}")

    image_paths: list[Path] = []
    for suffix in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif"):
        image_paths.extend(image_dir.glob(suffix))
        image_paths.extend(image_dir.glob(suffix.upper()))
    image_paths = sorted(set(image_paths), key=_natural_sort_key)
    if not image_paths:
        raise FileNotFoundError(f"no loop-closure images found under {image_dir}")

    out: list[Path] = []
    for timestamp in timestamps:
        idx = int(timestamp)
        if idx < 0 or idx >= len(image_paths):
            raise IndexError(f"timestamp {idx} is outside image range [0, {len(image_paths)}) for {image_dir}")
        out.append(image_paths[idx])
    return out


def select_retrieval_candidates(
    retrieval_hits: Mapping[int, Iterable[int]],
    *,
    temporal_exclusion: int,
    max_pairs: int,
) -> list[LoopCandidate]:
    selected: list[LoopCandidate] = []
    seen: set[tuple[int, int]] = set()
    for query_idx in sorted(int(k) for k in retrieval_hits.keys()):
        for hit_idx in retrieval_hits[query_idx]:
            i, j = int(query_idx), int(hit_idx)
            if i == j or abs(i - j) <= int(temporal_exclusion):
                continue
            pair = (max(i, j), min(i, j))
            if pair in seen:
                continue
            seen.add(pair)
            selected.append(LoopCandidate(ii=pair[0], jj=pair[1]))
            if max_pairs > 0 and len(selected) >= int(max_pairs):
                return selected
    return selected


def _normalize_quaternion(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def _quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    q = _normalize_quaternion(q)
    x, y, z, w = q.unbind(dim=-1)
    two = 2.0
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    row0 = torch.stack((1 - two * (yy + zz), two * (xy - wz), two * (xz + wy)), dim=-1)
    row1 = torch.stack((two * (xy + wz), 1 - two * (xx + zz), two * (yz - wx)), dim=-1)
    row2 = torch.stack((two * (xz - wy), two * (yz + wx), 1 - two * (xx + yy)), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def _matrix_to_quaternion(matrix: np.ndarray, *, device: torch.device) -> torch.Tensor:
    m = torch.as_tensor(matrix, dtype=torch.float32, device=device)
    q = torch.empty(4, dtype=torch.float32, device=device)
    trace = float((m[0, 0] + m[1, 1] + m[2, 2]).cpu())
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q[3] = 0.25 * s
        q[0] = (m[2, 1] - m[1, 2]) / s
        q[1] = (m[0, 2] - m[2, 0]) / s
        q[2] = (m[1, 0] - m[0, 1]) / s
    elif float(m[0, 0].cpu()) > float(m[1, 1].cpu()) and float(m[0, 0].cpu()) > float(m[2, 2].cpu()):
        s = math.sqrt(max(float((1.0 + m[0, 0] - m[1, 1] - m[2, 2]).cpu()), 1e-12)) * 2.0
        q[3] = (m[2, 1] - m[1, 2]) / s
        q[0] = 0.25 * s
        q[1] = (m[0, 1] + m[1, 0]) / s
        q[2] = (m[0, 2] + m[2, 0]) / s
    elif float(m[1, 1].cpu()) > float(m[2, 2].cpu()):
        s = math.sqrt(max(float((1.0 + m[1, 1] - m[0, 0] - m[2, 2]).cpu()), 1e-12)) * 2.0
        q[3] = (m[0, 2] - m[2, 0]) / s
        q[0] = (m[0, 1] + m[1, 0]) / s
        q[1] = 0.25 * s
        q[2] = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(max(float((1.0 + m[2, 2] - m[0, 0] - m[1, 1]).cpu()), 1e-12)) * 2.0
        q[3] = (m[1, 0] - m[0, 1]) / s
        q[0] = (m[0, 2] + m[2, 0]) / s
        q[1] = (m[1, 2] + m[2, 1]) / s
        q[2] = 0.25 * s
    return _normalize_quaternion(q)


def _invert_pose(pose: torch.Tensor) -> torch.Tensor:
    t = pose[:3]
    q = _normalize_quaternion(pose[3:7])
    q_inv = torch.stack((-q[0], -q[1], -q[2], q[3]))
    rot_inv = _quaternion_to_matrix(q_inv)
    t_inv = -(rot_inv @ t)
    return torch.cat((t_inv, q_inv), dim=0)


def add_loop_closure_edges(
    graph: FactorGraph,
    measurements: Sequence[LoopMeasurement],
    *,
    edge_confidence: Sequence[float],
) -> int:
    if not measurements:
        return 0

    device = graph.device
    confidence = torch.as_tensor(edge_confidence, dtype=torch.float32, device=device)
    if confidence.numel() != 2:
        raise ValueError("loop_closure.edge_confidence must contain [translation_conf, rotation_conf]")

    ii_values: list[int] = []
    jj_values: list[int] = []
    pose_values: list[torch.Tensor] = []
    scale_values: list[torch.Tensor] = []
    confidence_values: list[torch.Tensor] = []
    for measurement in measurements:
        pose_ij = measurement.relative_pose.to(device=device, dtype=torch.float32).clone()
        pose_ij[3:7] = _normalize_quaternion(pose_ij[3:7])
        scale_i = graph.buffer.depths_sens_scale[int(measurement.ii), 0].float()
        scale_j = graph.buffer.depths_sens_scale[int(measurement.jj), 0].float()
        pose_ij_metric = pose_ij.clone()
        pose_ij_metric[:3] = pose_ij_metric[:3] * scale_i
        pose_ji = _invert_pose(pose_ij_metric)
        pose_ji[:3] = pose_ji[:3] / scale_j.clamp_min(1e-6)
        for src, dst, pose, scale in (
            (measurement.ii, measurement.jj, pose_ij, scale_i),
            (measurement.jj, measurement.ii, pose_ji, scale_j),
        ):
            ii_values.append(int(src))
            jj_values.append(int(dst))
            pose_values.append(pose)
            scale_values.append(scale)
            confidence_values.append(confidence)

    ii = torch.as_tensor(ii_values, dtype=torch.long, device=device)
    jj = torch.as_tensor(jj_values, dtype=torch.long, device=device)
    pose = torch.stack(pose_values, dim=0)
    scale = torch.stack(scale_values, dim=0)
    edge_conf = torch.stack(confidence_values, dim=0)
    observability_score = torch.as_tensor(
        [float(measurement.n_matches) for measurement in measurements for _ in range(2)],
        dtype=torch.float32,
        device=device,
    )
    observability_rank = torch.full((ii.numel(),), 1, dtype=torch.uint8, device=device)
    keep = graph.edges.add(ii, jj, pose, scale, edge_conf, observability_score, observability_rank)
    return int(keep.sum().item())


def loop_closure_info(
    *,
    candidates: Sequence[LoopCandidate],
    measurements: Sequence[LoopMeasurement],
    added_edges: int,
    config_summary: dict,
) -> dict:
    match_counts = [int(measurement.n_matches) for measurement in measurements]
    info = {
        "loop_closure_candidates": int(len(candidates)),
        "loop_closure_accepted_pairs": int(len(measurements)),
        "loop_closure_added_edges": int(added_edges),
        "loop_closure_config": dict(config_summary),
    }
    if match_counts:
        info.update(
            {
                "loop_closure_match_count_min": int(min(match_counts)),
                "loop_closure_match_count_mean": float(sum(match_counts) / len(match_counts)),
                "loop_closure_match_count_max": int(max(match_counts)),
            }
        )
    else:
        info.update(
            {
                "loop_closure_match_count_min": 0,
                "loop_closure_match_count_mean": 0.0,
                "loop_closure_match_count_max": 0,
            }
        )
    return info


class Mast3RLoopClosureAdapter:
    def __init__(
        self,
        *,
        checkpoint: str,
        retriever_checkpoint: str | None,
        vendor_path: Path,
        device: torch.device,
        topk: int,
        temporal_exclusion: int,
        retrieval_score_thresh: float,
        min_matches: int,
        max_pairs: int,
    ) -> None:
        self.device = device
        self.topk = int(topk)
        self.temporal_exclusion = int(temporal_exclusion)
        self.retrieval_score_thresh = float(retrieval_score_thresh)
        self.min_matches = int(min_matches)
        self.max_pairs = int(max_pairs)
        self.vendor_path = Path(vendor_path)
        self._add_vendor_paths()

        import lietorch
        from mast3r_slam.frame import create_frame
        from mast3r_slam.mast3r_utils import load_mast3r, load_retriever, mast3r_match_asymmetric

        self._lietorch = lietorch
        self._create_frame = create_frame
        self._match_pair = mast3r_match_asymmetric
        self.model = load_mast3r(checkpoint, device=str(device)).eval()
        self.retriever = load_retriever(
            self.model,
            retriever_path=None if retriever_checkpoint is None or str(retriever_checkpoint) == "" else retriever_checkpoint,
            device=str(device),
        )

    def _add_vendor_paths(self) -> None:
        paths = [
            self.vendor_path,
            self.vendor_path / "thirdparty" / "mast3r",
            self.vendor_path / "thirdparty" / "mast3r" / "dust3r",
            self.vendor_path / "thirdparty" / "mast3r" / "asmk",
            self.vendor_path / "thirdparty" / "in3d",
        ]
        for path in paths:
            if path.exists() and str(path) not in sys.path:
                sys.path.insert(0, str(path))

    def _make_frames(self, images: Sequence[torch.Tensor]):
        frames = []
        identity = self._lietorch.Sim3.Identity(1, device=str(self.device))
        for idx, image in enumerate(images):
            image_np = image.detach().cpu().numpy()
            frame = self._create_frame(idx, image_np, identity, img_size=512, device=str(self.device))
            frames.append(frame)
        return frames

    def _retrieve_candidates(self, frames) -> list[LoopCandidate]:
        hits: dict[int, list[int]] = {}
        for idx, frame in enumerate(frames):
            if frame.feat is None:
                frame.feat, frame.pos, _ = self.model._encode_image(frame.img, frame.img_true_shape)
            top_indices = self.retriever.update(
                frame,
                add_after_query=True,
                k=self.topk,
                min_thresh=self.retrieval_score_thresh,
            )
            hits[idx] = [int(value) for value in top_indices]
        return select_retrieval_candidates(
            hits,
            temporal_exclusion=self.temporal_exclusion,
            max_pairs=self.max_pairs,
        )

    def _relative_pose_from_matches(
        self,
        candidate: LoopCandidate,
        frame_i,
        frame_j,
        graph: FactorGraph,
    ) -> LoopMeasurement | None:
        import cv2

        idx_i2j, valid_match_j, *_ = self._match_pair(self.model, frame_i, frame_j)
        idx_i2j = idx_i2j[0].detach().cpu().numpy().astype(np.int64)
        valid = valid_match_j[0].detach().cpu().numpy().astype(bool)
        n_matches = int(valid.sum())
        if n_matches < self.min_matches:
            return None

        h, w = [int(v) for v in frame_i.img_shape.flatten().detach().cpu().tolist()]
        src_idx = np.arange(idx_i2j.shape[0], dtype=np.int64)[valid]
        dst_idx = idx_i2j[valid]
        pts_i = np.stack((src_idx % w, src_idx // w), axis=-1).astype(np.float32)
        pts_j = np.stack((dst_idx % w, dst_idx // w), axis=-1).astype(np.float32)

        center = np.asarray([(w - 1) * 0.5, (h - 1) * 0.5], dtype=np.float32)
        focal = float(max(h, w))
        pts_i_norm = (pts_i - center) / focal
        pts_j_norm = (pts_j - center) / focal

        essential, inliers = cv2.findEssentialMat(
            pts_i_norm,
            pts_j_norm,
            focal=1.0,
            pp=(0.0, 0.0),
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0e-3,
        )
        if essential is None:
            return None
        if essential.shape[0] > 3:
            essential = essential[:3]
        _, rotation, translation, recovered = cv2.recoverPose(
            essential,
            pts_i_norm,
            pts_j_norm,
            focal=1.0,
            pp=(0.0, 0.0),
            mask=inliers,
        )
        if int(recovered) < self.min_matches:
            return None

        device = graph.device
        direction = torch.as_tensor(translation[:, 0], dtype=torch.float32, device=device)
        direction = direction / direction.norm().clamp_min(1e-6)
        source_scale = graph.buffer.depths_sens_scale[candidate.ii, 0].float().clamp_min(1e-6)
        current_baseline = (graph.buffer.poses[candidate.jj, :3] - graph.buffer.poses[candidate.ii, :3]).norm()
        translation_norm = current_baseline / source_scale
        quat = _matrix_to_quaternion(rotation, device=device)
        relative_pose = torch.cat((direction * translation_norm, quat), dim=0)
        return LoopMeasurement(
            ii=candidate.ii,
            jj=candidate.jj,
            relative_pose=relative_pose,
            n_matches=int(recovered),
            score=candidate.score,
        )

    def estimate(self, graph: FactorGraph, images: Sequence[torch.Tensor]) -> tuple[list[LoopCandidate], list[LoopMeasurement]]:
        frames = self._make_frames(images)
        candidates = self._retrieve_candidates(frames)
        measurements: list[LoopMeasurement] = []
        for candidate in candidates:
            measurement = self._relative_pose_from_matches(
                candidate,
                frames[candidate.ii],
                frames[candidate.jj],
                graph,
            )
            if measurement is not None:
                measurements.append(measurement)
        return candidates, measurements


def run_mast3r_loop_closure(
    graph: FactorGraph,
    images: Sequence[torch.Tensor],
    cfg,
    *,
    adapter: Mast3RLoopClosureAdapter | None = None,
) -> dict:
    config_summary = {
        "topk": int(cfg.topk),
        "temporal_exclusion": int(cfg.temporal_exclusion),
        "retrieval_score_thresh": float(cfg.retrieval_score_thresh),
        "min_matches": int(cfg.min_matches),
        "max_pairs": int(cfg.max_pairs),
        "edge_confidence": [float(v) for v in cfg.edge_confidence],
    }
    if adapter is None:
        vendor_path = Path(str(cfg.vendor_path) if "vendor_path" in cfg else "third_party/mast3r_slam/upstream")
        retriever_checkpoint = None
        if "retriever_checkpoint" in cfg and cfg.retriever_checkpoint is not None:
            retriever_checkpoint = str(cfg.retriever_checkpoint)
        adapter = Mast3RLoopClosureAdapter(
            checkpoint=str(cfg.checkpoint),
            retriever_checkpoint=retriever_checkpoint,
            vendor_path=vendor_path,
            device=graph.device,
            topk=int(cfg.topk),
            temporal_exclusion=int(cfg.temporal_exclusion),
            retrieval_score_thresh=float(cfg.retrieval_score_thresh),
            min_matches=int(cfg.min_matches),
            max_pairs=int(cfg.max_pairs),
        )

    candidates, measurements = adapter.estimate(graph, images)
    added_edges = add_loop_closure_edges(
        graph,
        measurements,
        edge_confidence=[float(v) for v in cfg.edge_confidence],
    )
    info = loop_closure_info(
        candidates=candidates,
        measurements=measurements,
        added_edges=added_edges,
        config_summary=config_summary,
    )
    graph.edges.pgo_info.update(info)
    return info
