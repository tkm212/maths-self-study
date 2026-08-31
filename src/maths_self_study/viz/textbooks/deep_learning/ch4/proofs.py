"""Proofs for Deep Learning Ch. 4 (Numerical Computation) dashboard pages."""

from __future__ import annotations

LEAST_SQUARES = (
    "Normal-equation solution",
    [
        r"Minimise $L(w) = \|Xw - y\|_2^2 = (Xw - y)^\top (Xw - y)$ over $w \in \mathbb{R}^p$.",
        r"Expand: $L(w) = w^\top X^\top X w - 2 y^\top X w + y^\top y$.",
        r"Set the gradient to zero: $\nabla_w L = 2 X^\top X w - 2 X^\top y = 0$.",
        r"If $X^\top X$ is invertible, $w^* = (X^\top X)^{-1} X^\top y$ is the unique critical point.",
        r"The Hessian $2 X^\top X \succ 0$, so this critical point is the global minimum (Goodfellow et al. §4.5).",
    ],
)

LOG_SUM_EXP = (
    "Log-sum-exp stabilisation",
    [
        r"Let $M = \max_i z_i$. For any shift $c$, "
        r"$\log \sum_i e^{z_i} = \log \sum_i e^{z_i - c} + c$ because $e^{z_i} = e^{c} e^{z_i - c}$.",
        r"Choose $c = M$: then $z_i - M \le 0$ for all $i$, so every exponent $e^{z_i - M} \in (0, 1]$ is finite.",
        r"Therefore $\log \sum_i e^{z_i} = M + \log \sum_i e^{z_i - M}$ avoids overflow from large positive $z_i$.",
        r"The softmax ratio $\exp(z_i)/\sum_j \exp(z_j)$ is unchanged by subtracting $M$ from all logits "
        r"(Goodfellow et al. §4.1).",
    ],
)
