"""Key LaTeX formulas for Deep Learning Ch. 4 (Numerical Computation)."""

from __future__ import annotations

# §4.1 — Overflow and underflow
SOFTMAX = r"\mathrm{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}"
LOG_SUM_EXP = r"\log \sum_i e^{z_i} = \max_i z_i + \log \sum_i e^{z_i - \max_j z_j}"

# §4.2 — Conditioning
CONDITION_NUMBER = r"\kappa(A) = \frac{\sigma_{\max}}{\sigma_{\min}}"
CONDITION_BOUND = r"\frac{\|\delta x\|}{\|x\|} \lesssim \kappa(A)\, \frac{\|\delta b\|}{\|b\|}"

# §4.3 — Gradient descent
GD_UPDATE = r"x \leftarrow x - \eta \nabla f(x)"

# §4.3.1 — Newton's method
NEWTON_UPDATE = r"x \leftarrow x - H^{-1} \nabla f(x)"

# §4.5 — Linear least squares
NORMAL_EQUATIONS = r"(A^\top A)\, w = A^\top y"
