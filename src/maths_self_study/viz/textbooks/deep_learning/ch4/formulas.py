"""Key LaTeX formulas for Deep Learning Ch. 4 (Numerical Computation)."""

from __future__ import annotations

# §4.4 — Constrained optimization and KKT
LAGRANGIAN = r"\mathcal{L}(x, \lambda) = f(x) + \lambda\, g(x)"
KKT_STATIONARITY = r"\nabla_x \mathcal{L}(x, \lambda) = \nabla f(x) + \lambda \nabla g(x) = 0"
KKT_COMPLEMENTARITY = r"\lambda\, g(x) = 0, \quad \lambda \geq 0, \quad g(x) \leq 0"
