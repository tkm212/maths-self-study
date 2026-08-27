"""Key LaTeX formulas for Deep Learning Ch. 5 (Machine Learning Basics)."""

from __future__ import annotations

# §5.2 — Capacity and overfitting
EMPIRICAL_RISK = (
    r"\hat{R}_S(f) = \frac{1}{m} \sum_{i=1}^{m} L\big(f(x^{(i)}), y^{(i)}\big)"
)
POLYNOMIAL_MODEL = r"f(x; \theta) = \sum_{j=0}^{d} \theta_j x^j"

# §5.3 — Validation and generalization
GENERALIZATION_GAP = (
    r"\text{gap} = \hat{R}_{\text{train}}(f) - \hat{R}_{\text{val}}(f)"
)

# §5.4 — Bias-variance decomposition
BIAS_VARIANCE_DECOMP = (
    r"\mathbb{E}_{x,y}\big[(y - \hat{f}(x))^2\big]"
    r" = \underbrace{\big(\mathbb{E}_x[\hat{f}(x)] - f(x)\big)^2}_{\text{bias}^2}"
    r" + \underbrace{\mathbb{E}_x\big[(\hat{f}(x) - \mathbb{E}_x[\hat{f}(x)])^2\big]}_{\text{variance}}"
    r" + \underbrace{\sigma^2}_{\text{noise}}"
)

# §5.5 — Maximum likelihood
LOG_LIKELIHOOD = (
    r"\ell(\theta) = \sum_{i=1}^{m} \log p_{\text{model}}(x^{(i)}; \theta)"
)
GAUSSIAN_PDF = (
    r"p(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}"
    r"\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)"
)
GAUSSIAN_MLE = (
    r"\hat{\mu} = \frac{1}{m}\sum_{i=1}^{m} x^{(i)}, \quad"
    r"\hat{\sigma}^2 = \frac{1}{m}\sum_{i=1}^{m}\big(x^{(i)} - \hat{\mu}\big)^2"
)

# §5.7 — Regularization (validation page uses ridge)
RIDGE_OBJECTIVE = (
    r"J(\theta) = \frac{1}{m}\sum_{i=1}^{m}\big(f(x^{(i)};\theta) - y^{(i)}\big)^2"
    r" + \lambda \|\theta\|_2^2"
)

# §5.9 — Stochastic gradient descent
GD_UPDATE = (
    r"\theta \leftarrow \theta - \epsilon \nabla_\theta"
    r" \frac{1}{m}\sum_{i=1}^{m} L\big(f(x^{(i)};\theta), y^{(i)}\big)"
)
MINIBATCH_UPDATE = (
    r"\theta \leftarrow \theta - \epsilon \nabla_\theta"
    r" \frac{1}{|B|}\sum_{i \in B} L\big(f(x^{(i)};\theta), y^{(i)}\big)"
)
