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
    # Structured models — factor the joint — Deep Learning Ch. 3 §3.14

    *Goodfellow, Bengio & Courville (2016). [Deep Learning](https://www.deeplearningbook.org/contents/prob.html).*

    $$P(x^{(1)}, \ldots, x^{(n)}) = P(x^{(1)}) \prod_{i=2}^{n} P(x^{(i)} \mid x^{(i-1)})$$

    Each edge is a conditional. RNNs / HMMs / autoregressive LMs are this with neural conditionals.
    """)
    return


@app.cell
def _(helpers, mo):
    demo = helpers.markov_chain_demo()
    _fig = helpers.plot_markov_chain(demo.p_x2_given_x1, labels=("X₁", "X₂"))
    chain = helpers.display(_fig, mo)
    print("Joint shape:", demo.joint.shape, "sum =", demo.joint.sum())
    print("P(X₃=1) =", demo.marginal_x3)
    return (chain,)


if __name__ == "__main__":
    app.run()
