import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from maths_self_study.deep_learning import ch3_helpers as helpers

    return (helpers,)


@app.cell
def _(mo):
    mo.md(r"""
    # The distributions deep learning lives on — Deep Learning Ch. 3 §3.9

    *Goodfellow, Bengio & Courville (2016). [Deep Learning](https://www.deeplearningbook.org/contents/prob.html).*

    **Bernoulli** — one bit. **Categorical** — one of $k$ classes. **Gaussian** — continuous workhorse.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""## Bernoulli — maximal uncertainty at $p = \tfrac{1}{2}$""")
    return


@app.cell
def _(helpers, mo):
    _fig = helpers.plot_binary_entropy_curve()
    bernoulli = helpers.display(_fig, mo)
    return (bernoulli,)


@app.cell
def _(mo):
    mo.md(r"""## Gaussian — elliptical level sets from the covariance""")
    return


@app.cell
def _(helpers, mo):
    gaussian = helpers.show_all(mo, *helpers.gaussian_demo_figures())
    return (gaussian,)


@app.cell
def _(mo):
    mo.md(r"""## Categorical — finite support""")
    return


@app.cell
def _(helpers, mo):
    _fig = helpers.plot_discrete_distribution(
        helpers.CATEGORICAL_LABELS,
        helpers.CATEGORICAL_PROBS,
        title="Softmax target distribution",
    )
    categorical = helpers.display(_fig, mo)
    return (categorical,)


if __name__ == "__main__":
    app.run()
