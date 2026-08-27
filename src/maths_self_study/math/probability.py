"""
Probability and information theory (Goodfellow, Bengio & Courville, Ch. 3).

Discrete distributions only; continuous differential entropy is handled in notebooks
via SciPy where needed.
"""

from __future__ import annotations

import numpy as np


def _as_probabilities(probs: np.ndarray) -> np.ndarray:
    p = np.asarray(probs, dtype=float)
    if p.ndim != 1:
        msg = "probabilities must be a one-dimensional array"
        raise ValueError(msg)
    if np.any(p < 0):
        msg = "probabilities must be non-negative"
        raise ValueError(msg)
    total = p.sum()
    if total <= 0:
        msg = "probabilities must sum to a positive value"
        raise ValueError(msg)
    return p / total


def bayes_posterior(
    prior: np.ndarray,
    likelihood: np.ndarray,
) -> np.ndarray:
    """
    Posterior P(x | y) from prior P(x) and likelihood P(y | x) (Bayes' rule, eq. 3.42).

    Both arrays are over the same discrete support for x. The likelihood gives
    P(y | x) for a fixed observation y, one value per x.
    """
    p = _as_probabilities(prior)
    ell = np.asarray(likelihood, dtype=float)
    if ell.shape != p.shape:
        msg = "prior and likelihood must have the same shape"
        raise ValueError(msg)
    if np.any(ell < 0):
        msg = "likelihood values must be non-negative"
        raise ValueError(msg)
    unnorm = p * ell
    return unnorm / unnorm.sum()


def marginalize(joint: np.ndarray, axis: int) -> np.ndarray:
    """Marginal distribution obtained by summing ``joint`` over ``axis``."""
    return np.asarray(joint, dtype=float).sum(axis=axis)


def shannon_entropy(probs: np.ndarray, *, base: float = np.e) -> float:
    """
    Shannon entropy H(P) = -Σ P(x) log P(x) (eq. 3.49), in nats when ``base=e``.

    Zero-probability outcomes contribute 0, following the convention lim_{x→0} x log x = 0.
    """
    p = _as_probabilities(probs)
    positive = p[p > 0]
    return float(-np.sum(positive * np.log(positive)) / np.log(base))


def cross_entropy(
    p: np.ndarray,
    q: np.ndarray,
    *,
    base: float = np.e,
) -> float:
    """
    Cross-entropy H(P, Q) = -Σ P(x) log Q(x).

    Used throughout deep learning as a training objective when P is the data distribution
    and Q is the model.
    """
    p_norm = _as_probabilities(p)
    q_norm = _as_probabilities(q)
    positive = p_norm > 0
    return float(-np.sum(p_norm[positive] * np.log(q_norm[positive])) / np.log(base))


def align_model_to_support(
    p: np.ndarray,
    q: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Floor Q on the support of P so cross-entropy and KL(P || Q) stay finite in demos."""
    p_norm = _as_probabilities(p)
    q_norm = _as_probabilities(q)
    q_safe = q_norm.copy()
    support = p_norm > 0
    q_safe[support] = np.maximum(q_safe[support], eps)
    return q_safe / q_safe.sum()


def kl_divergence(
    p: np.ndarray,
    q: np.ndarray,
    *,
    base: float = np.e,
) -> float:
    """
    Kullback-Leibler divergence D_KL(P || Q) (eq. 3.50).

    Non-negative; zero iff P and Q are identical on the support of P.
    """
    p_norm = _as_probabilities(p)
    q_norm = _as_probabilities(q)
    positive = p_norm > 0
    if np.any(q_norm[positive] <= 0):
        msg = "Q must be positive wherever P is positive"
        raise ValueError(msg)
    ratio = p_norm[positive] / q_norm[positive]
    return float(np.sum(p_norm[positive] * np.log(ratio)) / np.log(base))


def binary_entropy(p: float) -> float:
    """Entropy of Bernoulli(p) in nats — figure 3.5 in the Deep Learning book."""
    if not 0.0 <= p <= 1.0:
        msg = "p must lie in [0, 1]"
        raise ValueError(msg)
    if p in (0.0, 1.0):
        return 0.0
    return float(-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)))


def monty_hall_posterior(
    *,
    chosen_door: int,
    opened_door: int,
    n_doors: int = 3,
) -> np.ndarray:
    """
    Posterior probability the car is behind each door after the host opens a goat door.

    Returns a length-``n_doors`` vector. ``chosen_door`` is the contestant's pick (0-indexed);
    ``opened_door`` is the goat door revealed by the host.
    """
    if n_doors < 3:
        msg = "Monty Hall requires at least three doors"
        raise ValueError(msg)
    if not (0 <= chosen_door < n_doors and 0 <= opened_door < n_doors):
        msg = "door indices must lie in [0, n_doors)"
        raise ValueError(msg)
    if chosen_door == opened_door:
        msg = "host cannot open the chosen door"
        raise ValueError(msg)

    prior = np.full(n_doors, 1.0 / n_doors)
    likelihood = np.zeros(n_doors)
    for car_door in range(n_doors):
        if car_door == chosen_door:
            # Host chooses uniformly among the other goat doors.
            goat_doors = [d for d in range(n_doors) if d not in (car_door, chosen_door)]
            likelihood[car_door] = 1.0 / len(goat_doors) if opened_door in goat_doors else 0.0
        else:
            # Car not behind chosen door: host must open the remaining goat door.
            likelihood[car_door] = 1.0 if opened_door != car_door and opened_door != chosen_door else 0.0
    return bayes_posterior(prior, likelihood)
