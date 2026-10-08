"""Key LaTeX formulas for Deep Learning Ch. 7 (Regularization)."""

from __future__ import annotations

REGULARIZED_OBJECTIVE = r"\tilde{J}(\theta; X, y) = J(\theta; X, y) + \alpha \Omega(\theta)"
L2_PENALTY = r"\Omega(w) = \frac{1}{2}\lVert w \rVert_2^2"
WEIGHT_DECAY_UPDATE = r"w \leftarrow (1 - \eta\lambda) w - \eta \nabla_w J"
EARLY_STOPPING_RULE = r"\text{stop when validation error has not improved for } k \text{ epochs}"
DROPOUT_MASK = r"\tilde{h} = m \odot h / (1-p), \quad m_i \sim \mathrm{Bernoulli}(1-p)"
INPUT_NOISE = r"\tilde{x} = x + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)"
SEMI_SUPERVISED_OBJECTIVE = (
    r"\tilde{J} = J_{\mathrm{labeled}} + \alpha \,\mathbb{E}_{x}\lVert f(x) - f(\tilde{x}) \rVert^2"
)
MULTITASK_OBJECTIVE = r"J = J_1(\theta_{\mathrm{shared}}, \theta_1) + J_2(\theta_{\mathrm{shared}}, \theta_2)"
CONV1D = r"(x * k)[i] = \sum_{j} x[i+j]\, k[j]"
BAGGING_PRED = r"\hat{y} = \frac{1}{M}\sum_{m=1}^{M} f_m(x)"
ADVERSARIAL_PERTURB = r"x_{\mathrm{adv}} = x + \epsilon \,\mathrm{sign}(\nabla_x J(\theta, x, y))"
TANGENT_DISTANCE = r"d_{\mathrm{tangent}}(x, x') = \min_{\alpha}\lVert x - \mathcal{T}_\alpha(x') \rVert"
