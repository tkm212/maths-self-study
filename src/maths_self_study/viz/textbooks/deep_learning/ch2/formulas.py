"""Key LaTeX formulas for Deep Learning Ch. 2 (Linear Algebra)."""

from __future__ import annotations

# §2.1–2.2 — Vectors and matrices
MATRIX_MAP = r"A \in \mathbb{R}^{m \times n} \text{ is a linear map } x \mapsto Ax"
INNER_PRODUCT = r"x^\top y = \sum_i x_i y_i = \|x\|_2 \|y\|_2 \cos\theta"

# §2.5 — Norms
LP_NORM = r"\|x\|_p = \left(\sum_i |x_i|^p\right)^{1/p}"
COSINE_SIMILARITY = r"\cos\theta = \frac{x^\top y}{\|x\|_2 \|y\|_2}"

# §2.7 — Eigendecomposition
EIGENPAIR = r"Av = \lambda v, \quad v \neq 0"
SPECTRAL_DECOMPOSITION = r"A = Q \Lambda Q^\top \quad\text{(symmetric } A\text{)}"

# §2.8–2.9 — SVD
SVD = r"A = U \Sigma V^\top, \quad \sigma_1 \ge \sigma_2 \ge \cdots \ge 0"
SINGULAR_VALUES = r"\sigma_i = \sqrt{\lambda_i(A^\top A)}"
PSEUDOINVERSE_LS = r"x = A^+ b \text{ minimises } \|Ax - b\|_2"

# §2.12 — PCA
SAMPLE_COVARIANCE = r"\Sigma = \frac{X_c^\top X_c}{n-1}, \quad X_c = X - \mu"
PCA_PROJECTION = r"Z = X_c W^\top, \quad W = \text{top } k \text{ eigenvectors of } \Sigma"
