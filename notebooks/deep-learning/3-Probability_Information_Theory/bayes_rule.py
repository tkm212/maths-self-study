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
    # Bayes' rule — invert conditioning — Deep Learning Ch. 3 §3.11

    *Goodfellow, Bengio & Courville (2016). [Deep Learning](https://www.deeplearningbook.org/contents/prob.html).*

    $$P(x \mid y) = \frac{P(x)\,P(y \mid x)}{P(y)}$$

    Prior x likelihood → posterior.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""## Rare disease, positive test — base rate dominates""")
    return


@app.cell
def _(helpers, mo):
    medical_s = helpers.medical_test_scenario()
    _fig = helpers.plot_bayes_update(medical_s.states, medical_s.prior, medical_s.likelihood, medical_s.posterior)
    medical = helpers.display(_fig, mo)
    print(medical_s.summary)
    return (medical,)


@app.cell
def _(mo):
    mo.md(r"""## Monty Hall — switching wins with probability $\tfrac{2}{3}$""")
    return


@app.cell
def _(helpers, mo):
    monty_s = helpers.monty_hall_scenario()
    _fig = helpers.plot_monty_hall(monty_s.posterior, chosen=monty_s.chosen, opened=monty_s.opened)
    monty = helpers.display(_fig, mo)
    return (monty,)


if __name__ == "__main__":
    app.run()
