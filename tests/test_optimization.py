"""Tests for numerical computation and optimization utilities."""

from __future__ import annotations

import numpy as np
import pytest

from maths_self_study.optimization import (
    condition_number,
    gradient_descent_quadratic,
    kkt_quadratic_halfspace,
    linear_least_squares,
    log_sum_exp,
    near_singular_system,
    newton_quadratic,
    softmax_naive,
    softmax_stable,
    solve_perturbed,
)


def test_softmax_stable_sums_to_one():
    logits = np.array([1000.0, 1001.0, 1002.0])
    probs = softmax_stable(logits)
    np.testing.assert_allclose(probs.sum(), 1.0)
    assert np.all(probs > 0)


def test_softmax_naive_overflows_on_large_logits():
    logits = np.array([1000.0, 1001.0, 1002.0])
    naive = softmax_naive(logits)
    assert not np.all(np.isfinite(naive))


def test_log_sum_exp_matches_stable_softmax():
    logits = np.array([1.0, 2.0, 3.0])
    lse = log_sum_exp(logits)
    probs = softmax_stable(logits)
    np.testing.assert_allclose(np.log(probs), logits - lse, rtol=1e-10)
    np.testing.assert_allclose(lse, np.log(np.sum(np.exp(logits))), rtol=1e-10)


def test_condition_number_identity():
    assert condition_number(np.eye(2)) == pytest.approx(1.0)


def test_condition_number_diagonal():
    kappa = condition_number(np.diag([10.0, 1.0]))
    assert kappa == pytest.approx(10.0)


def test_gradient_descent_reaches_minimum():
    h = np.diag([2.0, 8.0])
    start = np.array([2.0, 2.0])
    path = gradient_descent_quadratic(h, np.zeros(2), start, learning_rate=0.1, n_steps=50)
    np.testing.assert_allclose(path[-1], np.zeros(2), atol=1e-3)


def test_newton_one_step_on_quadratic():
    h = np.diag([2.0, 8.0])
    start = np.array([2.0, 2.0])
    path = newton_quadratic(h, np.zeros(2), start, n_steps=1)
    np.testing.assert_allclose(path[-1], np.zeros(2), atol=1e-12)


def test_linear_least_squares_line_fit():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    design = np.column_stack([np.ones(4), x])
    targets = 1.0 + 1.5 * x
    weights, residuals = linear_least_squares(design, targets)
    np.testing.assert_allclose(weights, [1.0, 1.5], atol=1e-10)
    np.testing.assert_allclose(residuals, 0.0, atol=1e-10)


def test_near_singular_system_solution_is_one_one():
    matrix, rhs = near_singular_system(1e-4)
    x = np.linalg.solve(matrix, rhs)
    np.testing.assert_allclose(x, [1.0, 1.0], atol=1e-10)


def test_conditioning_amplifies_rhs_perturbation():
    matrix, rhs = near_singular_system(1e-4)
    x, x_pert, _, _, amplification = solve_perturbed(matrix, rhs, delta=1e-4, component=0)
    assert amplification > 100.0
    assert np.linalg.norm(x_pert - x) > 0.5


def test_kkt_inactive_when_origin_is_feasible():
    h = np.diag([1.0, 4.0])
    a = np.array([1.0, 1.0])
    x_star, lagrange, active = kkt_quadratic_halfspace(h, a, lower_bound=-1.0)
    np.testing.assert_allclose(x_star, [0.0, 0.0], atol=1e-12)
    assert lagrange == pytest.approx(0.0)
    assert active is False


def test_kkt_active_on_halfspace_boundary():
    h = np.diag([1.0, 4.0])
    a = np.array([1.0, 1.0])
    bound = 1.0
    x_star, lagrange, active = kkt_quadratic_halfspace(h, a, lower_bound=bound)
    np.testing.assert_allclose(a @ x_star, bound, atol=1e-10)
    np.testing.assert_allclose(h @ x_star, lagrange * a, atol=1e-10)
    assert active is True
    assert lagrange > 0
