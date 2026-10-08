"""Tests for Ch. 9 convolution utilities."""

from __future__ import annotations

import numpy as np

from maths_self_study.math.convolutional import (
    DEMO_IMAGE_SIZE,
    conv1d,
    conv2d,
    conv2d_dilated,
    demo_image,
    demo_series_1d,
    expand_kernel_dilation,
    gabor_kernel,
    pool2d,
    receptive_field_size,
    spatial_output_length,
    tower_demo_layers,
    translation_equivariance_stats,
)


def test_spatial_output_length_same_as_conv():
    img = demo_image(8)
    k = np.ones((3, 3))
    out = conv2d(img, k, stride=1, padding=1)
    assert out.shape[0] == spatial_output_length(8, 3, stride=1, padding=1)


def test_pool2d_reduces_size():
    x = np.arange(16, dtype=float).reshape(4, 4)
    pooled = pool2d(x, 2, stride=2, mode="max")
    assert pooled.shape == (2, 2)


def test_translation_equivariance_near_zero_with_padding():
    img = demo_image(DEMO_IMAGE_SIZE)
    k = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    stats = translation_equivariance_stats(img, k, shift_y=2, shift_x=2, padding=1, stride=1)
    assert stats["max_abs_diff"] < 1e-8
    assert stats["match_fraction"] == 1.0


def test_receptive_field_grows_with_depth():
    one = tower_demo_layers(1)
    three = tower_demo_layers(3)
    assert receptive_field_size(three) > receptive_field_size(one)


def test_dilated_kernel_expands():
    k = np.ones((3, 3))
    expanded = expand_kernel_dilation(k, 2)
    assert expanded.shape == (5, 5)


def test_conv1d_output_length():
    series = demo_series_1d(32)
    out = conv1d(series, np.array([-1.0, 0.0, 1.0]), padding=1)
    assert len(out) == len(series)


def test_dilated_conv_differs_from_standard():
    img = demo_image(12)
    k = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    a = conv2d(img, k, padding=1)
    b = conv2d_dilated(img, k, dilation=2, padding=2)
    assert not np.allclose(a, b)


def test_gabor_kernel_finite():
    g = gabor_kernel(7, theta=0.5)
    assert g.shape == (7, 7)
    assert np.all(np.isfinite(g))
