"""Tests for maths_self_study.dashboards.utils."""

from __future__ import annotations

import numpy as np
import pytest

from maths_self_study.dashboards.utils import (
    as_matrix,
    coerce_matrix_2x2,
    coerce_tensor_3d,
    format_matrix_2x2,
    parse_matrix_2x2,
    renorm,
)


def test_as_matrix():
    m = as_matrix(1.0, 2.0, 3.0, 4.0)
    assert m.shape == (2, 2)
    assert m[0, 1] == 2.0


def test_renorm():
    p = renorm(np.array([1.0, 1.0, 1.0]))
    np.testing.assert_allclose(p.sum(), 1.0)
    np.testing.assert_allclose(p, np.full(3, 1 / 3))


def test_renorm_handles_zero_total():
    p = renorm(np.array([0.0, 0.0]))
    np.testing.assert_allclose(p, np.array([0.5, 0.5]))


def test_format_parse_matrix_2x2_roundtrip():
    m = np.array([[1.0, 2.5], [3.0, 4.0]])
    text = format_matrix_2x2(m)
    parsed = parse_matrix_2x2(text, fallback=np.zeros((2, 2)))
    np.testing.assert_allclose(parsed, m)


def test_parse_matrix_2x2_invalid_falls_back():
    fallback = np.eye(2)
    parsed = parse_matrix_2x2("not a matrix", fallback=fallback)
    np.testing.assert_allclose(parsed, fallback)


def test_parse_matrix_2x2_accepts_commas_and_spaces():
    parsed = parse_matrix_2x2("1, 2\n3, 4", fallback=np.zeros((2, 2)))
    np.testing.assert_allclose(parsed, np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_parse_matrix_2x2_accepts_parentheses():
    text = format_matrix_2x2(np.array([[1.0, -2.0], [3.5, 4.0]]))
    parsed = parse_matrix_2x2(text, fallback=np.zeros((2, 2)))
    np.testing.assert_allclose(parsed, np.array([[1.0, -2.0], [3.5, 4.0]]))
    assert "(" in text and ")" in text


def test_coerce_matrix_2x2_uses_fallback_for_none():
    fallback = np.array([[1.0, 2.0], [3.0, 4.0]])
    m = coerce_matrix_2x2(None, 5.0, None, 6.0, fallback=fallback)
    np.testing.assert_allclose(m, np.array([[1.0, 5.0], [3.0, 6.0]]))


def test_coerce_tensor_3d():
    from maths_self_study.demos.deep_learning import ch2 as helpers

    ni, nj, nk = helpers.TENSOR_SHAPE
    ordered: list[int | float | None] = []
    for k in range(nk):
        for i in range(ni):
            for j in range(nj):
                ordered.append(float(helpers.TENSOR_DEFAULT[i, j, k]))
    tensor = coerce_tensor_3d(ordered, fallback=helpers.TENSOR_DEFAULT, shape=helpers.TENSOR_SHAPE)
    np.testing.assert_allclose(tensor, helpers.TENSOR_DEFAULT)


def test_complement_prob():
    from maths_self_study.dashboards.utils import clamp_prob, complement_prob, redistribute_simplex

    assert complement_prob(0.7) == pytest.approx(0.3)
    assert clamp_prob(None, default=0.6) == 0.6
    out = redistribute_simplex([0.4, 0.3, 0.2, 0.1], 0, 0.5)
    assert sum(out) == pytest.approx(1.0)
    assert out[0] == pytest.approx(0.5)


def test_coerce_float_and_vector():
    from maths_self_study.dashboards.utils import coerce_float, coerce_floats, coerce_vector2

    assert coerce_float(None, default=1.5) == 1.5
    assert coerce_float(2.5, default=0.0) == 2.5
    np.testing.assert_allclose(coerce_vector2(None, 3.0, fallback=np.array([1.0, 2.0])), [1.0, 3.0])
    np.testing.assert_allclose(coerce_floats([None, 2.0], fallback=np.array([5.0, 6.0])), [5.0, 2.0])
