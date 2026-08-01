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
    # Bayes' Rule — Deep Learning Ch. 3 §3.11

    *Goodfellow, Bengio & Courville (2016). [Deep Learning](https://www.deeplearningbook.org/contents/prob.html).*

    When we know the likelihood $P(y \mid x)$ and prior $P(x)$, Bayes' rule inverts conditioning:

    $$P(x \mid y) = \frac{P(x)\,P(y \mid x)}{P(y)}, \qquad P(y) = \sum_x P(y \mid x)P(x).$$

    §3.1 motivates probability with **incomplete observability** — the Monty Hall problem is the classic example.
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
    ## Medical testing (rare disease)

    A disease affects 1% of patients. A test is 95% sensitive and 95% specific. A positive test does **not** imply a 95% chance of disease — the prior matters.
    """)
    return


@app.cell
def _(ch3_helpers):
    import numpy as np

    from maths_self_study.probability import bayes_posterior

    states = np.array(["disease", "healthy"])
    prior = np.array([0.01, 0.99])
    likelihood = np.array([0.95, 0.05])  # P(positive | state)
    posterior = bayes_posterior(prior, likelihood)

    for state, prob in zip(states, posterior, strict=True):
        print(f"P({state} | +) = {prob:.4f}")
    return bayes_posterior, likelihood, np, posterior, prior, states


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Monty Hall problem

    Three doors: one hides a car, two hide goats. You pick a door; the host opens a goat door. Should you switch?

    Using `monty_hall_posterior`, the unopened non-chosen door has probability $2/3$ of hiding the car.
    """)
    return


@app.cell
def _():
    from maths_self_study.probability import monty_hall_posterior

    post = monty_hall_posterior(chosen_door=0, opened_door=1)
    for i, p in enumerate(post):
        print(f"P(car behind door {i}) = {p:.4f}")
    return monty_hall_posterior, post


if __name__ == "__main__":
    app.run()
