#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np


def _stats(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    return float(values.mean()), float(values.max())


def _edge_confidence(data: np.lib.npyio.NpzFile, n_edges: int) -> np.ndarray:
    if "edge_confidence" not in data:
        return np.ones(n_edges, dtype=np.float32)
    conf = np.asarray(data["edge_confidence"], dtype=np.float32)
    assert conf.shape[0] == n_edges
    if conf.ndim == 2:
        conf = conf.mean(axis=1)
    return np.clip(conf, 0.0, 1.0)


def _load_graph(path: Path) -> dict:
    data = np.load(path)
    ii = np.asarray(data["edge_ii"], dtype=np.int64)
    jj = np.asarray(data["edge_jj"], dtype=np.int64)
    n_edges = int(ii.shape[0])
    assert jj.shape[0] == n_edges

    if "trajectory" in data:
        n_nodes = int(data["trajectory"].shape[0])
    elif n_edges:
        n_nodes = int(max(ii.max(), jj.max()) + 1)
    else:
        n_nodes = 0

    if "timestamps" in data:
        timestamps = np.asarray(data["timestamps"], dtype=np.int64)
    else:
        timestamps = np.arange(n_nodes, dtype=np.int64)
    assert timestamps.shape[0] == n_nodes

    return {
        "ii": ii,
        "jj": jj,
        "timestamps": timestamps,
        "confidence": _edge_confidence(data, n_edges),
        "scales": np.asarray(data["scales"], dtype=np.float32) if "scales" in data else None,
    }


def _node_positions(n_nodes: int, width: int, margin: int) -> np.ndarray:
    if n_nodes <= 1:
        return np.full(n_nodes, width / 2.0)
    return np.linspace(margin, width - margin, n_nodes)


def _label_stride(n_nodes: int, requested: int) -> int:
    if requested > 0:
        return requested
    return max(1, math.ceil(n_nodes / 24))


def _path_for_edge(x0: float, x1: float, base_y: float, span: int) -> tuple[str, bool]:
    forward = x1 > x0
    arc = min(210.0, 28.0 + 15.0 * span)
    ctrl_y = base_y - arc if forward else base_y + arc
    return f"M {x0:.2f} {base_y:.2f} C {x0:.2f} {ctrl_y:.2f}, {x1:.2f} {ctrl_y:.2f}, {x1:.2f} {base_y:.2f}", forward


def render_svg(graph: dict, title: str, width: int, height: int, label_stride: int) -> str:
    ii = graph["ii"]
    jj = graph["jj"]
    timestamps = graph["timestamps"]
    confidence = graph["confidence"]
    n_nodes = int(timestamps.shape[0])
    n_edges = int(ii.shape[0])
    margin = 70
    base_y = height * 0.55
    xs = _node_positions(n_nodes, width, margin)

    out_degree = np.zeros(n_nodes, dtype=np.float32)
    in_degree = np.zeros(n_nodes, dtype=np.float32)
    for i, j in zip(ii, jj):
        out_degree[int(i)] += 1
        in_degree[int(j)] += 1
    max_degree = max(float((out_degree + in_degree).max()), 1.0) if n_nodes else 1.0

    spans = np.abs(ii - jj) if n_edges else np.asarray([], dtype=np.int64)
    mean_span, max_span = _stats(spans.astype(np.float32))
    forward = int(np.count_nonzero(jj > ii))
    backward = int(np.count_nonzero(jj < ii))
    mean_conf, _ = _stats(confidence.astype(np.float32))
    graph_label = "one-pass factors"

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<marker id="arrow-blue" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth">',
        '<path d="M 0 0 L 8 4 L 0 8 z" fill="#2563eb"/>',
        "</marker>",
        '<marker id="arrow-orange" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth">',
        '<path d="M 0 0 L 8 4 L 0 8 z" fill="#d97706"/>',
        "</marker>",
        "</defs>",
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        f'<text x="{margin}" y="42" font-family="Menlo, Consolas, monospace" font-size="22" font-weight="700" fill="#17202a">{escape(title)}</text>',
        f'<text x="{margin}" y="70" font-family="Menlo, Consolas, monospace" font-size="13" fill="#4b5563">'
        f"{graph_label}: keyframes={n_nodes} edges={n_edges} forward={forward} backward={backward} "
        f"mean_span={mean_span:.2f} max_span={max_span:.0f} mean_conf={mean_conf:.2f}</text>",
        f'<line x1="{margin}" y1="{base_y:.2f}" x2="{width - margin}" y2="{base_y:.2f}" stroke="#cbd5e1" stroke-width="2"/>',
    ]

    for k, (i, j) in enumerate(zip(ii, jj)):
        i = int(i)
        j = int(j)
        path, is_forward = _path_for_edge(xs[i], xs[j], base_y, abs(i - j))
        conf = float(confidence[k])
        opacity = 0.22 + 0.65 * conf
        stroke_width = 1.0 + 2.2 * conf
        color = "#2563eb" if is_forward else "#d97706"
        marker = "arrow-blue" if is_forward else "arrow-orange"
        edge_title = escape(f"edge {i}->{j}, span {abs(i - j)}, confidence {conf:.3f}")
        elements.append(
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{stroke_width:.2f}" '
            f'stroke-opacity="{opacity:.3f}" marker-end="url(#{marker})">'
            f"<title>{edge_title}</title></path>"
        )

    stride = _label_stride(n_nodes, label_stride)
    for idx in range(n_nodes):
        degree = float((out_degree[idx] + in_degree[idx]) / max_degree)
        radius = 5.0 + 8.0 * degree
        fill = "#111827" if idx == 0 else "#475569"
        elements.append(
            f'<circle cx="{xs[idx]:.2f}" cy="{base_y:.2f}" r="{radius:.2f}" fill="{fill}" stroke="#fbfaf7" stroke-width="2">'
            f"<title>keyframe {idx}, timestamp {int(timestamps[idx])}, out {int(out_degree[idx])}, in {int(in_degree[idx])}</title></circle>"
        )
        if idx % stride == 0 or idx == n_nodes - 1:
            elements.append(
                f'<text x="{xs[idx]:.2f}" y="{base_y + 30:.2f}" text-anchor="middle" '
                f'font-family="Menlo, Consolas, monospace" font-size="11" fill="#334155">{idx}</text>'
            )
            elements.append(
                f'<text x="{xs[idx]:.2f}" y="{base_y + 45:.2f}" text-anchor="middle" '
                f'font-family="Menlo, Consolas, monospace" font-size="10" fill="#64748b">t={int(timestamps[idx])}</text>'
            )

    legend_y = height - 62
    elements.extend(
        [
            f'<path d="M {margin} {legend_y} C {margin + 24} {legend_y - 22}, {margin + 72} {legend_y - 22}, {margin + 96} {legend_y}" '
            'fill="none" stroke="#2563eb" stroke-width="3" marker-end="url(#arrow-blue)"/>',
            f'<text x="{margin + 112}" y="{legend_y + 4}" font-family="Menlo, Consolas, monospace" font-size="12" fill="#334155">forward edge j &gt; i</text>',
            f'<path d="M {margin + 300} {legend_y} C {margin + 324} {legend_y + 22}, {margin + 372} {legend_y + 22}, {margin + 396} {legend_y}" '
            'fill="none" stroke="#d97706" stroke-width="3" marker-end="url(#arrow-orange)"/>',
            f'<text x="{margin + 412}" y="{legend_y + 4}" font-family="Menlo, Consolas, monospace" font-size="12" fill="#334155">backward edge j &lt; i</text>',
            f'<text x="{width - margin}" y="{height - 24}" text-anchor="end" font-family="Menlo, Consolas, monospace" font-size="11" fill="#64748b">node size = incident degree, edge opacity = confidence/tree membership</text>',
        ]
    )
    elements.append("</svg>")
    return "\n".join(elements)


def visualize_pose_npz(
    pose_npz: Path,
    output: Path,
    title: str | None,
    width: int,
    height: int,
    label_stride: int,
) -> dict:
    graph = _load_graph(pose_npz)
    svg = render_svg(
        graph,
        title or f"One-Pass Initialization Factor Graph: {pose_npz.stem}",
        width=width,
        height=height,
        label_stride=label_stride,
    )
    output.write_text(svg)
    ii, jj = graph["ii"], graph["jj"]
    spans = np.abs(ii - jj) if ii.size else np.asarray([], dtype=np.int64)
    mean_span, max_span = _stats(spans.astype(np.float32))
    return {
        "keyframes": int(graph["timestamps"].shape[0]),
        "edges": int(ii.shape[0]),
        "forward_edges": int(np.count_nonzero(jj > ii)),
        "backward_edges": int(np.count_nonzero(jj < ii)),
        "mean_span": mean_span,
        "max_span": max_span,
        "output": str(output),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize one-pass initialization factor graph edges.")
    parser.add_argument("pose_npz", type=Path, help="Pose artifact, e.g. outputs/pose/<sequence>.npz")
    parser.add_argument("-o", "--output", type=Path, help="Output SVG path.")
    parser.add_argument("--title", help="SVG title.")
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=560)
    parser.add_argument("--label-stride", type=int, default=0, help="Show every Nth keyframe label; 0 chooses automatically.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = args.output or args.pose_npz.with_name(f"{args.pose_npz.stem}_init_graph.svg")
    summary = visualize_pose_npz(
        args.pose_npz,
        output,
        title=args.title,
        width=args.width,
        height=args.height,
        label_stride=args.label_stride,
    )
    print(
        "keyframes={keyframes} edges={edges} forward={forward_edges} "
        "backward={backward_edges} mean_span={mean_span:.2f} max_span={max_span:.0f} output={output}".format(**summary)
    )


if __name__ == "__main__":
    main()
