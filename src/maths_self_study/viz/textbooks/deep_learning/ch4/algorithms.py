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
