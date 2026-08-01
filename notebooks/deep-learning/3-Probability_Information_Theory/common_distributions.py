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
    # Common Probability Distributions — Deep Learning Ch. 3 §3.9

    *Goodfellow, Bengio & Courville (2016). [Deep Learning](https://www.deeplearningbook.org/contents/prob.html).*

    Deep learning reuses a small set of distributions repeatedly:

    - **Bernoulli** — single binary outcome
    - **Categorical / Multinoulli** — one of $k$ classes
    - **Gaussian (normal)** — continuous modelling workhorse
    - **Exponential, Laplace, Beta** — conjugate priors and regularisation links

    This notebook plots a few of the most common forms from §3.9.
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
    ## Bernoulli and binary entropy

    If $X \sim \mathrm{Bernoulli}(p)$ then $P(X=1)=p$. Figure 3.5 plots $H(X) = -(1-p)\log(1-p) - p\log p$.
    """)
    return


@app.cell
def _(ch3_helpers):
    ch3_helpers.plot_binary_entropy_curve().show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gaussian (normal) density

    $$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

    Changing $\mu$ shifts the bell; changing $\sigma$ controls spread.
    """)
    return


@app.cell
def _(ch3_helpers):
    ch3_helpers.plot_gaussian_pdf(0.0, 1.0, title="Standard normal N(0, 1)").show()
    ch3_helpers.plot_gaussian_pdf(2.0, 0.5, title="N(2, 0.25)").show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Categorical distribution

    A **multinoulli** (single categorical draw) assigns probabilities $p_i$ to $k$ outcomes with $\sum_i p_i = 1$.
    """)
    return


@app.cell
def _(ch3_helpers):
    import numpy as np

    labels = np.array(["A", "B", "C", "D"])
    probs = np.array([0.05, 0.15, 0.30, 0.50])
    ch3_helpers.plot_discrete_distribution(labels, probs, title="Categorical example").show()
    return labels, np, probs


if __name__ == "__main__":
    app.run()
