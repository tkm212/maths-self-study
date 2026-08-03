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
    # Probability as bookkeeping — Deep Learning Ch. 3 §3.2-3.8

    *Goodfellow, Bengio & Courville (2016). [Deep Learning](https://www.deeplearningbook.org/contents/prob.html).*

    Joint → marginals (sum out) → conditionals (slice and renormalise):

    $$P(x \mid y) = \frac{P(x, y)}{P(y)}.$$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""## Joint → marginal → conditional""")
    return


@app.cell
def _(helpers, mo):
    _fig = helpers.plot_joint_with_marginals(
        helpers.RAIN_TRAFFIC_JOINT,
        row_labels=helpers.RAIN_TRAFFIC_ROW_LABELS,
        col_labels=helpers.RAIN_TRAFFIC_COL_LABELS,
        title="Joint table with marginals",
    )
    joint = helpers.display(_fig, mo)
    print("P(rain | heavy traffic) =", helpers.rain_traffic_conditional())
    return (joint,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Expectation and variance

    $\mathbb{E}[X]$ = centre of mass; $\mathrm{Var}(X)$ = spread about the mean.
    """)
    return


@app.cell
def _(helpers, mo):
    _, _, title = helpers.discrete_moments()
    _fig = helpers.plot_discrete_distribution([0, 1, 2, 3], [0.1, 0.2, 0.3, 0.4], title=title)
    moments = helpers.display(_fig, mo)
    return (moments,)


if __name__ == "__main__":
    app.run()
