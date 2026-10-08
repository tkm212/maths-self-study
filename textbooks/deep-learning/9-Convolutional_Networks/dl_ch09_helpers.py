"""Shared plotting helpers for Deep Learning Ch. 9 (Convolutional Networks)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from maths_self_study.math.convolutional import (
    DEMO_IMAGE_SIZE,
    KERNEL_PRESETS,
    conv2d,
    demo_image,
    flatten_spatial_params,
    pool2d,
    receptive_field_size,
    shift_image,
    spatial_output_length,
    stack_spatial_sizes,
    tower_demo_layers,
    translation_equivariance_stats,
)
from maths_self_study.viz.graphs import apply_layout, bar_chart, heatmap_chart, line_chart

STRIDE_DEFAULT = 1
PADDING_DEFAULT = 0
POOL_SIZE_DEFAULT = 2
POOL_STRIDE_DEFAULT = 2
TOWER_BLOCKS_DEFAULT = 3
KERNEL_SIZE_DEFAULT = 3
SHIFT_DEFAULT = 2
FILTER_DEFAULT = "sobel_x"


def _axis_labels(n: int) -> list[int]:
    return list(range(n))


def _heatmap_panel(z: np.ndarray, *, title: str, colorscale: str = "Viridis") -> go.Figure:
    h, w = z.shape if z.size else (1, 1)
    fig = heatmap_chart(
        z,
        x=_axis_labels(w),
        y=_axis_labels(h),
        colorscale=colorscale,
        zmin=float(np.min(z)) if z.size else 0,
        zmax=float(np.max(z)) if z.size else 1,
        showscale=True,
        title=title,
        height=360,
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def _three_panel(
    left: np.ndarray,
    mid: np.ndarray,
    right: np.ndarray,
    *,
    titles: tuple[str, str, str],
) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=list(titles),
        horizontal_spacing=0.06,
    )
    for col, arr, scale in zip([1, 2, 3], [left, mid, right], ["Viridis", "RdBu", "Plasma"], strict=True):
        h, w = arr.shape if arr.size else (1, 1)
        heatmap_chart(
            arr,
            x=_axis_labels(w),
            y=_axis_labels(h),
            colorscale=scale,
            showscale=col == 3,
            row=1,
            col=col,
            fig=fig,
        )
        fig.update_yaxes(autorange="reversed", row=1, col=col)
    apply_layout(fig, height=380)
    return fig


def plot_convolution(
    stride: int,
    padding: int,
    filter_name: str,
) -> tuple[go.Figure, go.Figure, dict[str, float]]:
    """Input, kernel, and feature map (§9.1)."""
    img = demo_image(DEMO_IMAGE_SIZE)
    name = filter_name if filter_name in KERNEL_PRESETS else FILTER_DEFAULT
    kernel = KERNEL_PRESETS[name]
    s = max(1, int(stride))
    p = max(0, int(padding))
    out = conv2d(img, kernel, stride=s, padding=p)
    fig_maps = _three_panel(
        img,
        kernel,
        out,
        titles=("Input", f"Kernel ({name})", "Feature map"),
    )
    h_out = spatial_output_length(DEMO_IMAGE_SIZE, kernel.shape[0], stride=s, padding=p)
    w_out = spatial_output_length(DEMO_IMAGE_SIZE, kernel.shape[1], stride=s, padding=p)
    fig_bar = bar_chart(
        ["Input H", "Output H", "Input W", "Output W"],
        [DEMO_IMAGE_SIZE, h_out, DEMO_IMAGE_SIZE, w_out],
        title="Spatial sizes after cross-correlation",
        yaxis_title="pixels",
        color="#60a5fa",
        height=340,
    )
    stats = {
        "output_h": float(h_out),
        "output_w": float(w_out),
        "kernel_norm": float(np.linalg.norm(kernel)),
        "activation_max": float(np.max(out)) if out.size else 0.0,
    }
    return fig_maps, fig_bar, stats


def plot_pooling(
    pool_size: int,
    pool_stride: int,
    mode: str,
) -> tuple[go.Figure, go.Figure, dict[str, float]]:
    """Max vs average pooling on a conv feature map (§9.2)."""
    img = demo_image(DEMO_IMAGE_SIZE)
    feat = conv2d(img, KERNEL_PRESETS["sobel_x"], stride=1, padding=1)
    p = max(1, int(pool_size))
    s = max(1, int(pool_stride))
    pooled = pool2d(feat, p, stride=s, mode=mode if mode in {"max", "avg"} else "max")
    window = np.ones((p, p), dtype=float)
    fig_maps = _three_panel(
        feat,
        window,
        pooled,
        titles=("Conv feature map", f"{mode} pool {p}x{p}", "Pooled map"),
    )

    other = "avg" if mode == "max" else "max"
    alt = pool2d(feat, p, stride=s, mode=other)
    diff = np.abs(pooled - alt) if pooled.shape == alt.shape else pooled
    fig_diff = heatmap_chart(
        diff,
        colorscale="Hot",
        title=f"|{mode} - {other}| at same stride",
        height=360,
    )
    fig_diff.update_yaxes(autorange="reversed")
    stats = {
        "in_h": float(feat.shape[0]),
        "out_h": float(pooled.shape[0]),
        "pool_size": float(p),
        "pool_stride": float(s),
        "compression": float(feat.size / pooled.size) if pooled.size else 1.0,
    }
    return fig_maps, fig_diff, stats


def plot_edge_filters(filter_name: str) -> tuple[go.Figure, go.Figure, dict[str, float]]:
    """Classical edge and blur filters on the demo image (§9.1)."""
    img = demo_image(DEMO_IMAGE_SIZE)
    name = filter_name if filter_name in KERNEL_PRESETS else FILTER_DEFAULT
    kernel = KERNEL_PRESETS[name]
    out = conv2d(img, kernel, stride=1, padding=1)
    fig = _three_panel(img, kernel, out, titles=("Input", name.replace("_", " "), "Response"))
    stacks = []
    labels = []
    for key in ("sobel_x", "sobel_y", "laplacian"):
        resp = conv2d(img, KERNEL_PRESETS[key], stride=1, padding=1)
        stacks.append(float(np.mean(np.abs(resp))))
        labels.append(key)
    fig_bar = bar_chart(
        labels,
        stacks,
        title="Mean |activation| across filter bank",
        yaxis_title="mean |response|",
        color="#2563eb",
        height=340,
    )
    stats = {
        "selected_mean_abs": float(np.mean(np.abs(out))),
        "selected_max": float(np.max(out)) if out.size else 0.0,
    }
    return fig, fig_bar, stats


def plot_receptive_field(
    n_blocks: int,
    kernel_size: int,
) -> tuple[go.Figure, go.Figure, dict[str, float]]:
    """Receptive field growth and parameter comparison (§9.3)."""
    layers = tower_demo_layers(n_blocks, kernel_size=kernel_size, pool_size=2)
    sizes = stack_spatial_sizes(DEMO_IMAGE_SIZE, layers)
    rfs = []
    partial: list = []
    for layer in layers:
        partial.append(layer)
        rfs.append(receptive_field_size(partial))
    depth = np.arange(len(rfs))
    fig_rf = line_chart(
        depth,
        rfs,
        name="receptive field (1D)",
        color="#2563eb",
        mode="lines+markers",
        title="Receptive field vs depth",
        xaxis_title="block index",
        yaxis_title="RF size (pixels)",
        height=400,
    )
    fig_size = line_chart(
        np.arange(len(sizes)),
        sizes,
        name="spatial map size",
        color="#16a34a",
        mode="lines+markers",
        title="Spatial resolution vs depth",
        xaxis_title="layer index",
        yaxis_title="H = W (pixels)",
        height=400,
    )
    fc_params, conv_params = flatten_spatial_params(DEMO_IMAGE_SIZE, kernel_size)
    stats = {
        "receptive_field": float(rfs[-1]) if rfs else 1.0,
        "final_spatial": float(sizes[-1]),
        "fc_params": float(fc_params),
        "conv_params": float(conv_params),
    }
    return fig_rf, fig_size, stats


def plot_translation_equivariance(
    shift: int,
    padding: int,
) -> tuple[go.Figure, go.Figure, dict[str, float]]:
    """Shift commutes with convolution under zero pad (§9.3)."""
    img = demo_image(DEMO_IMAGE_SIZE)
    kernel = KERNEL_PRESETS["sobel_x"]
    dy = int(shift)
    dx = int(shift)
    p = max(0, int(padding))
    base = conv2d(img, kernel, stride=1, padding=p)
    shifted_in = shift_image(img, dy, dx)
    conv_shift = conv2d(shifted_in, kernel, stride=1, padding=p)
    shift_conv = shift_image(base, dy, dx)
    margin = max(abs(dy), abs(dx)) + kernel.shape[0] // 2
    h = min(conv_shift.shape[0], shift_conv.shape[0])
    w = min(conv_shift.shape[1], shift_conv.shape[1])
    if h > 2 * margin and w > 2 * margin:
        cs = conv_shift[margin : h - margin, margin : w - margin]
        sc = shift_conv[margin : h - margin, margin : w - margin]
    else:
        cs = conv_shift[:h, :w]
        sc = shift_conv[:h, :w]
    diff = np.abs(cs - sc)
    fig = _three_panel(
        cs,
        sc,
        diff,
        titles=("conv(shift(x)) interior", "shift(conv(x)) interior", "|difference|"),
    )
    stats_dict = translation_equivariance_stats(
        img,
        kernel,
        shift_y=dy,
        shift_x=dx,
        padding=p,
        stride=1,
    )
    fig_line = bar_chart(
        ["max |diff|", "mean |diff|"],
        [stats_dict["max_abs_diff"], stats_dict["mean_abs_diff"]],
        title="Translation equivariance error (should be ~0 with same padding)",
        yaxis_title="abs error",
        color="#64748b",
        height=340,
    )
    stats = {
        **stats_dict,
        "shift_pixels": float(dy),
        "padding": float(p),
    }
    return fig, fig_line, stats


def plot_cnn_tower(
    n_blocks: int,
    kernel_size: int,
) -> tuple[go.Figure, go.Figure, dict[str, float]]:
    """Spatial shrinkage through conv+pool tower (§9.1-§9.3)."""
    layers = tower_demo_layers(n_blocks, kernel_size=kernel_size, pool_size=2)
    sizes = stack_spatial_sizes(DEMO_IMAGE_SIZE, layers)
    labels = [f"L{i}" for i in range(len(sizes))]
    fig_bar = bar_chart(
        labels,
        sizes,
        title="Feature map side length after each block",
        yaxis_title="H = W",
        color="#60a5fa",
        height=400,
    )
    params_per_block = kernel_size * kernel_size + 1
    cumulative = np.cumsum([params_per_block] * len(layers))
    fig_params = line_chart(
        np.arange(1, len(layers) + 1),
        cumulative,
        name="shared-weight params",
        color="#dc2626",
        mode="lines+markers",
        title="Parameter count (one channel, shared kernel per block)",
        xaxis_title="depth",
        yaxis_title="parameters",
        height=400,
    )
    stats = {
        "input_size": float(DEMO_IMAGE_SIZE),
        "output_size": float(sizes[-1]),
        "n_blocks": float(len(layers)),
        "params_estimate": float(cumulative[-1]) if len(cumulative) else 0.0,
    }
    return fig_bar, fig_params, stats
