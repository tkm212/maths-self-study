"""Key LaTeX formulas for Deep Learning Ch. 7 (Regularization)."""

from __future__ import annotations

REGULARIZED_OBJECTIVE = r"\tilde{J}(\theta; X, y) = J(\theta; X, y) + \alpha \Omega(\theta)"
L2_PENALTY = r"\Omega(w) = \frac{1}{2}\lVert w \rVert_2^2"
WEIGHT_DECAY_UPDATE = r"w \leftarrow (1 - \eta\lambda) w - \eta \nabla_w J"
EARLY_STOPPING_RULE = r"\text{stop when validation error has not improved for } k \text{ epochs}"
DROPOUT_MASK = r"\tilde{h} = m \odot h / (1-p), \quad m_i \sim \mathrm{Bernoulli}(1-p)"
INPUT_NOISE = r"\tilde{x} = x + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)"
