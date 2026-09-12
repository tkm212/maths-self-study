"""Key LaTeX formulas for Deep Learning Ch. 4 (Numerical Computation)."""

from __future__ import annotations

# §4.1 — Overflow, underflow, and stable softmax
SOFTMAX = r"P(y=i) = \frac{\exp(z_i)}{\sum_j \exp(z_j)}"
STABLE_SOFTMAX = r"P(y=i) = \frac{\exp(z_i - m)}{\sum_j \exp(z_j - m)}, \quad m = \max_i z_i"
LOG_SUM_EXP = r"\log\sum_i \exp(z_i) = m + \log\sum_i \exp(z_i - m), \quad m = \max_i z_i"

# §4.2 — Conditioning
CONDITION_NUMBER = r"\kappa(A) = \frac{\sigma_{\max}}{\sigma_{\min}}"
ERROR_AMPLIFICATION = (
    r"\frac{\|\delta x\|}{\|x\|} \lesssim \kappa(A)\,\frac{\|\delta b\|}{\|b\|}"
    r"\quad\text{when solving } Ax = b"
)

# §4.4 — Constrained optimization and KKT
LAGRANGIAN = r"\mathcal{L}(x, \lambda) = f(x) + \lambda\, g(x)"
KKT_STATIONARITY = r"\nabla_x \mathcal{L}(x, \lambda) = \nabla f(x) + \lambda \nabla g(x) = 0"
KKT_COMPLEMENTARITY = r"\lambda\, g(x) = 0, \quad \lambda \geq 0, \quad g(x) \leq 0"

# §4.5 — Least squares
LEAST_SQUARES_OBJECTIVE = r"\min_w \|Aw - b\|_2^2"
NORMAL_EQUATIONS = r"A^\top A w^* = A^\top b"
OLS_SOLUTION = r"w^* = (A^\top A)^{-1} A^\top b \quad\text{(when } A^\top A \text{ is invertible)}"
