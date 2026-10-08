"""Key LaTeX formulas for Deep Learning Ch. 9."""

from __future__ import annotations

CONV2D = r"(I * K)[i,j] = \sum_{u,v} I[i+u,\, j+v]\, K[u,v]"
OUTPUT_SIZE = r"H_{\mathrm{out}} = \left\lfloor \frac{H_{\mathrm{in}} + 2P - K}{S} \right\rfloor + 1"
MAX_POOL = r"y_{i,j} = \max_{(u,v) \in \mathcal{R}_{i,j}} x_{u,v}"
AVG_POOL = r"y_{i,j} = \frac{1}{|\mathcal{R}_{i,j}|}\sum_{(u,v) \in \mathcal{R}_{i,j}} x_{u,v}"
RECEPTIVE_FIELD = r"r_\ell = r_{\ell-1} + (k_\ell - 1)\, j_{\ell-1}, \quad j_\ell = j_{\ell-1}\, s_\ell"
