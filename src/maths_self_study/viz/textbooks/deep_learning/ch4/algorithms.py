"""Algorithms for Deep Learning Ch. 4 dashboard pages."""

from __future__ import annotations

GRADIENT_DESCENT = (
    "Gradient descent",
    [
        r"Choose learning rate $\eta > 0$ and initialise $x^{(0)}$.",
        r"Repeat until convergence: compute gradient $g^{(t)} = \nabla f(x^{(t)})$.",
        r"Update $x^{(t+1)} = x^{(t)} - \eta g^{(t)}$ (§4.3).",
        r"Stop when $\|g^{(t)}\|$ is small or progress in $f$ falls below tolerance.",
    ],
)

NEWTON = (
    "Newton's method",
    [
        r"Initialise $x^{(0)}$ near a critical point of $f$.",
        r"At step $t$, compute gradient $g^{(t)} = \nabla f(x^{(t)})$ and Hessian $H^{(t)} = \nabla^2 f(x^{(t)})$.",
        r"Solve $H^{(t)} d^{(t)} = -g^{(t)}$ and set $x^{(t+1)} = x^{(t)} + d^{(t)}$ (§4.4).",
        r"Requires $H^{(t)}$ to be positive definite near a minimum; otherwise use damped or quasi-Newton variants.",
    ],
)

STABLE_SOFTMAX = (
    "Stable softmax and log-sum-exp",
    [
        r"Given logits $z$, set $m = \max_i z_i$.",
        r"Compute $P(y=i) = \exp(z_i - m) / \sum_j \exp(z_j - m)$ — turns logits into a probability vector.",
        r"Naive $\exp(z_i)$ overflows when $m$ is large and underflows when $z_i \ll m$; max-subtraction keeps exponents in $(-1, 0]$.",
        r"Log-sum-exp: $\log\sum_i \exp(z_i) = m + \log\sum_i \exp(z_i - m)$ (§4.1).",
    ],
)

KKT = (
    "KKT conditions for inequality constraints",
    [
        r"Form Lagrangian $\mathcal{L}(x, \lambda) = f(x) + \lambda g(x)$ with $\lambda \ge 0$ for $g(x) \le 0$.",
        r"Stationarity: $\nabla f(x^*) + \lambda^* \nabla g(x^*) = 0$.",
        r"Primal feasibility: $g(x^*) \le 0$; dual feasibility: $\lambda^* \ge 0$.",
        r"Complementary slackness: $\lambda^* g(x^*) = 0$ — active constraints have positive multipliers.",
    ],
)

LEAST_SQUARES = (
    "Linear least squares via normal equations",
    [
        r"Overdetermined system $Aw \approx b$ — minimise $\|Aw - b\|_2^2$.",
        r"Critical points satisfy normal equations $A^\top A w^* = A^\top b$.",
        r"When $A^\top A$ is invertible, $w^* = (A^\top A)^{-1} A^\top b$; otherwise use pseudoinverse $A^+$ (Ch. 2).",
    ],
)
