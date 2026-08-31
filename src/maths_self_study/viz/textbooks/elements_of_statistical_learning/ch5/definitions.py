"""Definitions for ESL Ch. 5 (Basis Expansions and Regularization) dashboard pages."""

from __future__ import annotations

SPLINES = [
    (
        "Basis expansion",
        r"Replace a single input $x$ with derived features $h(x) = (h_1(x), \ldots, h_M(x))$ and fit a linear model "
        r"$f(x) = \sum_{m=1}^M \theta_m h_m(x)$ in the expanded space (ESL §5.1).",
    ),
    (
        "Spline",
        r"A **piecewise polynomial** joined at **knots** $\xi_1 < \cdots < \xi_K$ with continuity constraints at each knot. "
        r"A **cubic spline** is piecewise cubic and $C^2$ — $f$, $f'$, and $f''$ are continuous at every knot (ESL §5.2).",
    ),
    (
        "Natural cubic spline",
        r"A cubic spline that is **linear beyond the outermost knots** ($f'' = 0$ for $x < \xi_1$ and $x > \xi_K$). "
        r"Reduces extrapolation wildness and lowers effective degrees of freedom (ESL §5.2.1).",
    ),
    (
        "Spline degrees of freedom",
        r"A natural cubic spline with $K$ interior knots has $K + 4$ free parameters — $K$ cubic pieces subject to "
        r"$3(K-1)$ continuity constraints plus two natural boundary conditions (ESL §5.2).",
    ),
]

SMOOTHING_SPLINES = [
    (
        "Smoothing spline",
        r"Minimises $\sum_{i=1}^n (y_i - f(x_i))^2 + \lambda \int [f''(t)]^2 \, dt$ over an appropriate function space. "
        r"The solution is a natural cubic spline with a knot at every distinct $x_i$ (ESL §5.4).",
    ),
    (
        "Effective degrees of freedom",
        r"For a linear smoother $\hat{y} = S_\lambda y$, $\text{df}_\lambda = \text{tr}(S_\lambda)$ summarises model flexibility. "
        r"As penalty $\lambda \uparrow$, df decreases from $n$ toward 2 (a global line).",
    ),
]
