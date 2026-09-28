"""Tests for Ch. 9 convolution utilities."""

from __future__ import annotations

import numpy as np

from maths_self_study.math.convolutional import (
    DEMO_IMAGE_SIZE,
    conv2d,
    demo_image,
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
