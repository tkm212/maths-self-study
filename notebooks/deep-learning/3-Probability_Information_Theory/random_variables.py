import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Random Variables and Probability — Deep Learning Ch. 3 §3.2-3.8

    *Goodfellow, Bengio & Courville (2016). [Deep Learning](https://www.deeplearningbook.org/).*

    Probability theory gives a language for **uncertainty**. A random variable $X$ takes values $x$ with probabilities given by a distribution $P(X)$.

    For discrete variables the **probability mass function** satisfies $P(x) \in [0,1]$ and $\sum_x P(x) = 1$. The **marginal** over a subset of variables sums (or integrates) out the rest; **conditional** probability is

    $$P(x \mid y) = \frac{P(x, y)}{P(y)} \qquad \text{when } P(y) > 0.$$

    Two variables are **independent** when $P(x, y) = P(x)P(y)$, or equivalently $P(x \mid y) = P(x)$.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))

    import ch3_helpers

    ch3_helpers.init_paths()
    return (ch3_helpers,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Joint, marginal, and conditional probabilities

    Consider a tiny joint over two binary variables $A$ (rain) and $B$ (traffic). From the joint table we can compute marginals and conditionals using `maths_self_study.probability.marginalize`.
    """)
    return


@app.cell
def _(ch3_helpers):
    import numpy as np

    from maths_self_study.probability import marginalize

    joint = np.array([
        [0.10, 0.15],  # A=0
        [0.25, 0.50],  # A=1
    ])
    p_a = marginalize(joint, axis=1)
    p_b = marginalize(joint, axis=0)
    p_a_given_b1 = joint[:, 1] / p_b[1]

    ch3_helpers.plot_discrete_distribution(np.array([0, 1]), p_a, title="Marginal P(A)").show()

    print("P(A):", p_a)
    print("P(B):", p_b)
    print("P(A | B=1):", p_a_given_b1)
    return joint, marginalize, np, p_a, p_a_given_b1, p_b


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Expectation and variance

    For discrete $X$, the expectation is $\mathbb{E}[X] = \sum_x x P(x)$ and $\mathrm{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2]$.
    """)
    return


@app.cell
def _(np):
    support = np.array([0, 1, 2, 3])
    probs = np.array([0.1, 0.2, 0.3, 0.4])
    mean = np.sum(support * probs)
    variance = np.sum((support - mean) ** 2 * probs)
    print(f"E[X] = {mean:.2f}, Var(X) = {variance:.2f}")
    return mean, probs, support, variance


if __name__ == "__main__":
    app.run()
