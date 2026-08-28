"""Theorems for Deep Learning Ch. 4 (Numerical Computation) dashboard pages."""

from __future__ import annotations

STABILITY = [
    (
        "Log-sum-exp identity",
        r"For any $z_1, \ldots, z_n$, $\log \sum_i \exp(z_i) = \max_i z_i + \log \sum_i \exp(z_i - \max_j z_j)$, keeping all exponentials finite.",
    ),
]

CONDITIONING = [
    (
        "Condition-number perturbation bound",
        r"For $Ax = b$ and perturbed system $(A + \Delta A)x' = b + \Delta b$, relative error in $x$ is bounded by $\kappa(A)$ times relative input error, up to first order.",
    ),
]

LEAST_SQUARES = [
    (
        "Normal-equation solution",
        r"If $X^\top X$ is invertible, $w^* = (X^\top X)^{-1} X^\top y$ is the unique minimiser of $\|Xw - y\|_2^2$.",
    ),
]

KKT = [
    (
        "Karush-Kuhn-Tucker conditions",
        r"For $\min f(x)$ s.t. $g(x) \le 0$ with smooth $f, g$, a feasible point $x^*$ is optimal iff "
        r"$\nabla f(x^*) + \lambda \nabla g(x^*) = 0$, $g(x^*) \le 0$, $\lambda \ge 0$, and $\lambda g(x^*) = 0$.",
    ),
]
