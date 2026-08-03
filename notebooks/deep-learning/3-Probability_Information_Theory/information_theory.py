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
    # Information and surprise — Deep Learning Ch. 3 §3.13

    *Goodfellow, Bengio & Courville (2016). [Deep Learning](https://www.deeplearningbook.org/contents/prob.html).*

    $I(x) = -\log P(x)$ — rare events are surprising.
    $H(P)$ averages surprise; $H(P,Q)$ is the classification loss; KL is asymmetric.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""## Self-information: $-\log P(x)$""")
    return


@app.cell
def _(helpers, mo):
    _fig = helpers.plot_self_information(helpers.INFO_P, labels=helpers.INFO_LABELS)
    self_info = helpers.display(_fig, mo)
    return (self_info,)


@app.cell
def _(mo):
    mo.md(r"""## Cross-entropy and KL — direction matters""")
    return


@app.cell
def _(helpers, mo):
    import numpy as np

    measures = helpers.summarize_information_measures(helpers.INFO_P, helpers.INFO_Q)
    print(helpers.format_measures(measures))
    _fig = helpers.plot_kl_asymmetric(np.arange(len(helpers.INFO_P)), helpers.INFO_P, helpers.INFO_Q)
    kl = helpers.display(_fig, mo)
    return (kl,)


if __name__ == "__main__":
    app.run()
