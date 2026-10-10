"""Key LaTeX formulas for Deep Learning Ch. 9."""

from __future__ import annotations

CONV2D = r"(I * K)[i,j] = \sum_{u,v} I[i+u,\, j+v]\, K[u,v]"
OUTPUT_SIZE = r"H_{\mathrm{out}} = \left\lfloor \frac{H_{\mathrm{in}} + 2P - K}{S} \right\rfloor + 1"
MAX_POOL = r"y_{i,j} = \max_{(u,v) \in \mathcal{R}_{i,j}} x_{u,v}"
AVG_POOL = r"y_{i,j} = \frac{1}{|\mathcal{R}_{i,j}|}\sum_{(u,v) \in \mathcal{R}_{i,j}} x_{u,v}"
RECEPTIVE_FIELD = r"r_\ell = r_{\ell-1} + (k_\ell - 1)\, j_{\ell-1}, \quad j_\ell = j_{\ell-1}\, s_\ell"
EQUIVARIANCE = r"f(g(x)) = g(f(x)) \;\text{for translation } g \text{ (conv layers, §9.2)}"
DILATED_CONV = r"K'_{i,j} = K_{i/d,\, j/d} \text{ when } i,j \equiv 0 \pmod{d} \text{ (zeros between taps)}"
POINTWISE_CONV = r"y_{i,j,c'} = \sum_c W_{c',c}\, x_{i,j,c} \quad \text{(1x1 mix, §9.5)}"
CONV1D = r"(x * k)[t] = \sum_{\tau} x[t+\tau]\, k[\tau]"
GABOR_FORM = (
    r"g(x,y) = \exp\!\left(-\frac{x_\theta^2 + \gamma^2 y_\theta^2}{2\sigma^2}\right)"
    r"\cos\!\left(\frac{2\pi x_\theta}{\lambda} + \psi\right)"
)
