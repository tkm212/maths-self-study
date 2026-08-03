import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from maths_self_study.deep_learning import ch2_helpers as helpers

    return (helpers,)


@app.cell
def _(mo):
    mo.md(r"""
    # Norms as geometry — Deep Learning Ch. 2 §2.5

    *Goodfellow, Bengio & Courville (2016). [Deep Learning](https://www.deeplearningbook.org/contents/linear_algebra.html).*

    $$\|x\|_p = \left(\sum_i |x_i|^p\right)^{1/p}$$

    Unit balls: $L^2$ circle, $L^1$ diamond, $L^\infty$ square.
    $L^1$ corners on axes → sparsity; $L^2$ shrinks uniformly.
    """)
    return


@app.cell
def _(helpers, mo):
    _fig = helpers.plot_lp_unit_balls()
    balls = helpers.display(_fig, mo)
    return (balls,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Cosine similarity

    $\cos\theta = \dfrac{x^\top y}{\|x\|_2 \|y\|_2}$ — angle without magnitude.
    """)
    return


@app.cell
def _(helpers):
    print(helpers.norm_summary())
    return


if __name__ == "__main__":
    app.run()
