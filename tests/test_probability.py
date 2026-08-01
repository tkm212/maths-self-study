"""Unit tests for maths_self_study.probability."""

import numpy as np
import pytest

from maths_self_study.probability import (
    bayes_posterior,
    binary_entropy,
    cross_entropy,
    kl_divergence,
    marginalize,
    monty_hall_posterior,
    shannon_entropy,
)


def test_bayes_posterior_medical_example() -> None:
    # Goodfellow et al. style: rare disease, imperfect test.
    prior = np.array([0.01, 0.99])
    likelihood = np.array([0.95, 0.05])  # P(positive test | disease state)
    post = bayes_posterior(prior, likelihood)
    assert post[0] == pytest.approx(0.01 * 0.95 / (0.01 * 0.95 + 0.99 * 0.05), rel=1e-6)


def test_marginalize_joint() -> None:
    joint = np.array([[0.1, 0.2], [0.3, 0.4]])
    assert marginalize(joint, axis=1) == pytest.approx(np.array([0.3, 0.7]))
    assert marginalize(joint, axis=0) == pytest.approx(np.array([0.4, 0.6]))


def test_binary_entropy_extremes() -> None:
    assert binary_entropy(0.0) == pytest.approx(0.0)
    assert binary_entropy(1.0) == pytest.approx(0.0)
    assert binary_entropy(0.5) == pytest.approx(np.log(2.0))


def test_shannon_entropy_uniform() -> None:
    probs = np.array([0.25, 0.25, 0.25, 0.25])
    assert shannon_entropy(probs) == pytest.approx(np.log(4.0))


def test_kl_cross_entropy_identity() -> None:
    p = np.array([0.2, 0.3, 0.5])
    q = np.array([0.1, 0.4, 0.5])
    assert kl_divergence(p, q) == pytest.approx(cross_entropy(p, q) - shannon_entropy(p))
    assert kl_divergence(p, p) == pytest.approx(0.0)


def test_monty_hall_switch_wins() -> None:
    # Contestant picks door 0; host opens door 1 (goat).
    post = monty_hall_posterior(chosen_door=0, opened_door=1)
    assert post[0] == pytest.approx(1.0 / 3.0)
    assert post[2] == pytest.approx(2.0 / 3.0)


def test_invalid_probabilities_raise() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        shannon_entropy(np.array([-0.1, 1.1]))
