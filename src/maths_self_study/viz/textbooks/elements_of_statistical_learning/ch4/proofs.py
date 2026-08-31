"""Proofs for ESL Ch. 4 (Linear Methods for Classification) dashboard pages."""

from __future__ import annotations

LDA = (
    "LDA boundary from the Bayes rule",
    [
        r"Class $k$ has prior $\pi_k$ and density $f_k(x) = \mathcal{N}(\mu_k, \Sigma)$ with shared $\Sigma$.",
        r"The Bayes classifier assigns $x$ to $\arg\max_k \pi_k f_k(x)$.",
        r"Take logs (monotone): compare $\log \pi_k + \log f_k(x)$ across $k$.",
        r"The quadratic $-\tfrac{1}{2}(x-\mu_k)^\top \Sigma^{-1}(x-\mu_k)$ expands to "
        r"$-\tfrac{1}{2} x^\top \Sigma^{-1} x + \mu_k^\top \Sigma^{-1} x - \tfrac{1}{2}\mu_k^\top \Sigma^{-1}\mu_k$.",
        r"The term $-\tfrac{1}{2} x^\top \Sigma^{-1} x$ is common to all classes, so the decision depends on "
        r"$\delta_k(x) = x^\top \Sigma^{-1}\mu_k - \tfrac{1}{2}\mu_k^\top \Sigma^{-1}\mu_k + \log \pi_k$ — a linear function of $x$ "
        r"(ESL §4.3).",
    ],
)

PERCEPTRON = (
    "Perceptron convergence (separable data)",
    [
        r"Suppose $\|x_i\| \le R$ and labels $y_i \in \{-1,+1\}$ with margin $\gamma > 0$: "
        r"$\exists w^*$, $\|w^*\|=1$, such that $y_i (w^{*\top} x_i) \ge \gamma$ for all $i$.",
        r"Perceptron updates only on mistakes: if $y_i (w^\top x_i) \le 0$, set $w \leftarrow w + y_i x_i$.",
        r"After an update, $(w + y_i x_i)^\top w^* \ge w^\top w^* + \gamma$; summing over $T$ mistakes gives "
        r"$w^\top w^* \ge T \gamma$.",
        r"Also $\|w\|^2$ increases by at most $R^2$ per mistake, so $\|w\|^2 \le T R^2$.",
        r"Cauchy-Schwarz: $T \gamma \le w^\top w^* \le \|w\| \|w^*\| \le R\sqrt{T}$, hence $T \le R^2/\gamma^2$ — "
        r"finite mistakes and convergence to some separating hyperplane (ESL §4.5.1).",
    ],
)
