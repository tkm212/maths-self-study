"""Definitions for Deep Learning Ch. 4 (Numerical Computation) dashboard pages."""

from __future__ import annotations

STABILITY = [
    (
        "Numerical overflow",
        r"Floating-point exponents have finite range; $\exp(z)$ overflows when $z$ is large, producing $\infty$ and breaking downstream normalisation.",
    ),
    (
        "Log-sum-exp trick",
        r"Rewriting $\log \sum_i \exp(z_i)$ as $\max(z) + \log \sum_i \exp(z_i - \max(z))$ keeps exponentials bounded and softmax numerically stable.",
    ),
]

CONDITIONING = [
    (
        "Condition number",
        r"$\kappa(A) = \sigma_{\max}/\sigma_{\min}$ ratio of largest to smallest singular value. "
        r"Large $\kappa(A)$ means $Ax = b$ is sensitive to tiny changes in $b$ or rounding error in $A$.",
    ),
    (
        "Ill-conditioned system",
        r"When $\kappa(A)$ is huge, nearly parallel rows make different inputs look alike while solutions can differ enormously.",
    ),
]

GRADIENT_DESCENT = [
    (
        "Gradient",
        r"$\nabla f(x)$ points in the direction of steepest ascent. "
        r"First-order methods move opposite to the gradient to decrease $f$.",
    ),
    (
        "Learning rate",
        r"Step size $\eta$ in $x \leftarrow x - \eta \nabla f(x)$. Too large causes oscillation; too small slows convergence.",
    ),
]

NEWTON = [
    (
        "Newton's method",
        r"Uses second-order curvature: $x \leftarrow x - H^{-1}\nabla f(x)$ where $H$ is the Hessian. "
        r"Converges fast near a minimum but costs $O(n^3)$ per step and needs $H$ invertible.",
    ),
    (
        "Hessian",
        r"Matrix of second partial derivatives $H_{ij} = \partial^2 f / \partial x_i \partial x_j$. "
        r"Its eigenvalues describe local curvature along each direction.",
    ),
]

LEAST_SQUARES = [
    (
        "Linear least squares",
        r"Find $w$ minimising $\|Xw - y\|_2^2$ — the best linear fit when equations $Xw = y$ are overdetermined or noisy.",
    ),
    (
        "Normal equations",
        r"Critical points satisfy $X^\top X w = X^\top y$. When $X^\top X$ is invertible, $w = (X^\top X)^{-1} X^\top y$ is the unique minimiser.",
    ),
]

KKT = [
    (
        "Lagrangian",
        r"For inequality constraint $g(x) \le 0$, $\mathcal{L}(x, \lambda) = f(x) + \lambda g(x)$ with multiplier $\lambda \ge 0$. "
        r"Stationarity, primal/dual feasibility, and complementary slackness characterise optima.",
    ),
    (
        "Complementary slackness",
        r"At a KKT point, $\lambda_i g_i(x^*) = 0$ for each inequality: either the constraint is active ($g_i = 0$) or $\lambda_i = 0$.",
    ),
]
