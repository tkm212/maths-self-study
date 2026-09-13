"""Key LaTeX formulas for Deep Learning Ch. 6 (Deep Feedforward Networks)."""

from __future__ import annotations

# §6.1 — XOR example
MLP_COMPOSITION = r"f(x; W, b) = f^{(3)}\big(f^{(2)}(f^{(1)}(x))\big)"
XOR_TARGET = r"y = x_1 \oplus x_2"

# §6.2.2 — Output units
SIGMOID_OUTPUT = r"\hat{y} = \sigma(w^\top h + b) = \frac{1}{1 + e^{-(w^\top h + b)}}"
SOFTMAX_OUTPUT = r"\hat{y} = \mathrm{softmax}(z), \quad z = W h + b"
CROSS_ENTROPY = r"L = -\sum_i y_i \log \hat{y}_i"

# §6.3 — Hidden activations
RELU = r"\mathrm{ReLU}(z) = \max(0, z)"
LOGISTIC_SIGMOID = r"\sigma(z) = \frac{1}{1 + e^{-z}}"
TANH = r"\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}"

# §6.4.1 — Universal approximation
UNIVERSAL_APPROX = (
    r"\forall f^* \text{ continuous on compact } K,\; \exists \hat{f} \text{ (MLP)} "
    r"\text{ s.t. } \sup_{x \in K} |f^*(x) - \hat{f}(x)| < \varepsilon"
)
