"""Key LaTeX formulas for Deep Learning Ch. 2 (Linear Algebra)."""

from __future__ import annotations

# §2.1-2.3 — Vectors and matrices
MATRIX_VECTOR = r"y = Ax"
INNER_PRODUCT = r"x^\top y = \|x\|_2 \|y\|_2 \cos\theta"

# §2.3 — Norms
LP_NORM = r"\|x\|_p = \left(\sum_i |x_i|^p\right)^{1/p}"
COSINE_SIMILARITY = r"\cos(\theta) = \frac{x^\top y}{\|x\|_2 \|y\|_2}"

# §2.7 — Eigendecomposition
EIGENDECOMPOSITION = r"A = Q \Lambda Q^\top, \quad A q_i = \lambda_i q_i"

# §2.8 — Singular value decomposition
SVD_DECOMPOSITION = r"A = U \Sigma V^\top"
PSEUDOINVERSE = r"A^+ = V \Sigma^+ U^\top, \quad x = A^+ b"

# §2.12 — PCA
COVARIANCE = r"\Sigma = \frac{1}{n-1} X_c^\top X_c"
PCA_PROJECTION = r"Z = X_c W^\top, \quad \hat{X} = Z W + \mu"

# §2.3 — Tensors
FROBENIUS_NORM = r"\|T\|_F = \sqrt{\sum_{i,j,k} T_{ijk}^2}"
