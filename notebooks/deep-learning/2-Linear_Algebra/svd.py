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
    # SVD — every matrix has a geometry — Deep Learning Ch. 2 §2.8–2.9

    *Goodfellow, Bengio & Courville (2016). [Deep Learning](https://www.deeplearningbook.org/contents/linear_algebra.html).*

    $A = U\Sigma V^\top$: rotate, scale, rotate.
    Singular values $\sigma_i$ are axis lengths of the unit ball's image.
    The pseudoinverse $A^+$ gives minimum-norm least squares for $Ax \approx b$.
    """)
    return


@app.cell
def _(helpers, mo):
    _fig = helpers.plot_svd_geometry(helpers.SVD_MAP, title="Unit circle → ellipse; σᵢ = axis lengths")
    svd = helpers.display(_fig, mo)
    return (svd,)


@app.cell
def _(mo):
    mo.md(r"""## Least squares via $A^+$""")
    return


@app.cell
def _(helpers):
    print(helpers.least_squares_summary())
    return


if __name__ == "__main__":
    app.run()
