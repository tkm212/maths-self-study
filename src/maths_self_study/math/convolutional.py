"""Convolution, pooling, and spatial geometry for Deep Learning Ch. 9 demos."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KERNEL_PRESETS: dict[str, np.ndarray] = {
    "identity": np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
    "box_blur": np.ones((3, 3), dtype=float) / 9.0,
    "sharpen": np.array([[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]]),
    "sobel_x": np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]),
    "sobel_y": np.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]),
    "laplacian": np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]),
}

DEMO_IMAGE_SIZE = 16


def demo_image(size: int = DEMO_IMAGE_SIZE) -> np.ndarray:
    """Synthetic 2D pattern with edges and a bright block (no external assets)."""
    n = max(8, int(size))
    yy, xx = np.mgrid[0:n, 0:n]
    checker = ((xx // 3) + (yy // 3)) % 2
    stripe = (xx + yy) % 5 == 0
    block = (xx >= n // 3) & (xx < 2 * n // 3) & (yy >= n // 4) & (yy < 3 * n // 4)
    img = 0.25 * checker + 0.35 * stripe.astype(float) + 0.55 * block.astype(float)
    return np.clip(img, 0.0, 1.0)


def spatial_output_length(
    input_size: int,
    kernel_size: int,
    *,
    stride: int = 1,
    padding: int = 0,
) -> int:
    """Output length along one spatial axis (valid-style formula with padding)."""
    inp = int(input_size)
    k = int(kernel_size)
    s = max(1, int(stride))
    p = max(0, int(padding))
    if inp + 2 * p < k:
        return 0
    return (inp + 2 * p - k) // s + 1


def pad2d(x: np.ndarray, padding: int) -> np.ndarray:
    p = max(0, int(padding))
    if p == 0:
        return np.asarray(x, dtype=float)
    return np.pad(np.asarray(x, dtype=float), ((p, p), (p, p)), mode="constant")


def conv2d(
    x: np.ndarray,
    kernel: np.ndarray,
    *,
    stride: int = 1,
    padding: int = 0,
) -> np.ndarray:
    """2D cross-correlation (the usual deep learning 'convolution')."""
    img = pad2d(x, padding)
    k = np.asarray(kernel, dtype=float)
    kh, kw = k.shape
    s = max(1, int(stride))
    h, w = img.shape
    out_h = (h - kh) // s + 1
    out_w = (w - kw) // s + 1
    if out_h <= 0 or out_w <= 0:
        return np.zeros((0, 0), dtype=float)
    out = np.zeros((out_h, out_w), dtype=float)
    for i in range(out_h):
        for j in range(out_w):
            patch = img[i * s : i * s + kh, j * s : j * s + kw]
            out[i, j] = float(np.sum(patch * k))
    return out


def pool2d(
    x: np.ndarray,
    pool_size: int,
    *,
    stride: int | None = None,
    mode: str = "max",
) -> np.ndarray:
    """Max or average pooling on a 2D feature map."""
    arr = np.asarray(x, dtype=float)
    p = max(1, int(pool_size))
    s = max(1, int(stride if stride is not None else p))
    h, w = arr.shape
    out_h = (h - p) // s + 1
    out_w = (w - p) // s + 1
    if out_h <= 0 or out_w <= 0:
        return np.zeros((0, 0), dtype=float)
    out = np.zeros((out_h, out_w), dtype=float)
    for i in range(out_h):
        for j in range(out_w):
            patch = arr[i * s : i * s + p, j * s : j * s + p]
            if mode == "avg":
                out[i, j] = float(np.mean(patch))
            else:
                out[i, j] = float(np.max(patch))
    return out


def shift_image(x: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Zero-padded integer shift (positive dy moves content down)."""
    arr = np.asarray(x, dtype=float)
    h, w = arr.shape
    out = np.zeros_like(arr)
    dy, dx = int(dy), int(dx)
    for i in range(h):
        for j in range(w):
            si, sj = i - dy, j - dx
            if 0 <= si < h and 0 <= sj < w:
                out[i, j] = arr[si, sj]
    return out


def _interior_crop(a: np.ndarray, b: np.ndarray, margin: int) -> tuple[np.ndarray, np.ndarray]:
    m = max(0, int(margin))
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    if h <= 2 * m or w <= 2 * m:
        return a[:h, :w], b[:h, :w]
    return a[m : h - m, m : w - m], b[m : h - m, m : w - m]


def translation_equivariance_stats(
    x: np.ndarray,
    kernel: np.ndarray,
    *,
    shift_y: int,
    shift_x: int,
    padding: int = 0,
    stride: int = 1,
) -> dict[str, float]:
    """Compare conv(shift(x)) with shift(conv(x)) on the interior (zero-pad borders differ)."""
    dy, dx = int(shift_y), int(shift_x)
    shifted = shift_image(x, dy, dx)
    conv_then_shift = shift_image(conv2d(x, kernel, stride=stride, padding=padding), dy, dx)
    shift_then_conv = conv2d(shifted, kernel, stride=stride, padding=padding)
    if conv_then_shift.size == 0 or shift_then_conv.size == 0:
        return {"max_abs_diff": float("nan"), "mean_abs_diff": float("nan"), "match_fraction": 0.0}
    k = np.asarray(kernel, dtype=float)
    margin = max(abs(dy), abs(dx)) + k.shape[0] // 2
    a, b = _interior_crop(conv_then_shift, shift_then_conv, margin)
    diff = np.abs(a - b)
    tol = 1e-10
    match = float(np.mean(diff <= tol)) if diff.size else 0.0
    return {
        "max_abs_diff": float(np.max(diff)) if diff.size else 0.0,
        "mean_abs_diff": float(np.mean(diff)) if diff.size else 0.0,
        "match_fraction": match,
    }


@dataclass(frozen=True)
class LayerSpec:
    kernel_size: int
    stride: int
    padding: int
    pool_size: int
    pool_stride: int


def receptive_field_size(layers: list[LayerSpec]) -> int:
    """Receptive field along one axis after a stack of conv+pool blocks (§9.3)."""
    rf = 1
    jump = 1
    for layer in layers:
        k = max(1, int(layer.kernel_size))
        s = max(1, int(layer.stride))
        rf += (k - 1) * jump
        jump *= s
        if layer.pool_size > 1:
            p = max(1, int(layer.pool_size))
            ps = max(1, int(layer.pool_stride))
            rf += (p - 1) * jump
            jump *= ps
    return int(rf)


def stack_spatial_sizes(
    input_size: int,
    layers: list[LayerSpec],
) -> list[int]:
    """Spatial length after each block (conv then pool)."""
    size = int(input_size)
    sizes = [size]
    for layer in layers:
        size = spatial_output_length(
            size,
            layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
        )
        if layer.pool_size > 1:
            size = spatial_output_length(size, layer.pool_size, stride=layer.pool_stride, padding=0)
        sizes.append(size)
    return sizes


def tower_demo_layers(n_blocks: int, *, kernel_size: int = 3, pool_size: int = 2) -> list[LayerSpec]:
    """Build a simple CNN tower spec for receptive-field demos."""
    n = max(1, min(int(n_blocks), 6))
    k = max(1, int(kernel_size))
    p = max(1, int(pool_size))
    return [LayerSpec(kernel_size=k, stride=1, padding=1, pool_size=p, pool_stride=p) for _ in range(n)]


def flatten_spatial_params(input_size: int, kernel_size: int) -> tuple[int, int]:
    """Fully connected vs conv parameter count at one location (§9.1)."""
    inp = int(input_size)
    k = int(kernel_size)
    fc = inp * inp
    conv = k * k
    return fc, conv
