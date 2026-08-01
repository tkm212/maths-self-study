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
    # Information Theory — Deep Learning Ch. 3 §3.13

    *Goodfellow, Bengio & Courville (2016). [Deep Learning](https://www.deeplearningbook.org/contents/prob.html).*

    **Self-information** of outcome $x$ is $I(x) = -\log P(x)$. The **Shannon entropy**

    $$H(P) = \mathbb{E}_{x \sim P}[-\log P(x)]$$

    measures average uncertainty. **Cross-entropy** and **KL divergence** underpin classification losses and variational inference.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    import ch3_helpers as helpers

    return (helpers,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Binary entropy (figure 3.5)
    """)
    return


@app.cell
def _(helpers, mo):
    helpers.display(helpers.plot_binary_entropy_curve(), mo)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cross-entropy and KL divergence

    Compare two discrete distributions over four symbols. Note $D_{\mathrm{KL}}(P \parallel Q) \neq D_{\mathrm{KL}}(Q \parallel P)$ (figure 3.6).
    """)
    return


@app.cell
def _(helpers, mo):
    import numpy as np

    p = np.array([0.40, 0.30, 0.20, 0.10])
    q = np.array([0.25, 0.25, 0.25, 0.25])
    measures = helpers.summarize_information_measures(p, q)
    for name, value in measures.items():
        print(f"{name:16s} = {value:.4f} nats")

    xs = np.arange(len(p))
    fig = helpers.plot_kl_asymmetric(xs, p, q)
    helpers.display(fig, mo)
    return fig, measures, np, p, q, xs


if __name__ == "__main__":
    app.run()
