"""Theorems for ESL Ch. 6 (Kernel Smoothing Methods) dashboard pages."""

from __future__ import annotations

KERNEL_SMOOTHERS = [
    (
        "LOO-CV shortcut for linear smoothers",
        r"If $\hat{y} = S y$, leave-one-out predictions satisfy "
        r"$\text{CV}(\lambda) = \frac{1}{n}\sum_i \left(\frac{y_i - \hat{f}_\lambda(x_i)}{1 - S_{ii}}\right)^2$ "
        r"without refitting $n$ times (ESL §6.2).",
    ),
]
