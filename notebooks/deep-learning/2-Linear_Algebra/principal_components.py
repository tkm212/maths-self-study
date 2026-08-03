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
    # PCA — best low-dimensional view — Deep Learning Ch. 2 §2.12

    *Goodfellow, Bengio & Courville (2016). [Deep Learning](https://www.deeplearningbook.org/contents/linear_algebra.html).*

    Orthogonal directions of maximal variance = eigenvectors of the covariance.
    Encode $c = W^\top(x - \mu)$; decode $\hat{x} = Wc + \mu$.
    """)
    return


@app.cell
def _(helpers, mo):
    demo = helpers.pca_demo()
    pca = helpers.show_all(mo, *helpers.pca_figures(demo))
    print("Reconstruction error:", demo.reconstruction_error)
    return (pca,)


if __name__ == "__main__":
    app.run()
