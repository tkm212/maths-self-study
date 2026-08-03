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
    # Eigendecomposition — invariant directions — Deep Learning Ch. 2 §2.7

    *Goodfellow, Bengio & Courville (2016). [Deep Learning](https://www.deeplearningbook.org/contents/linear_algebra.html).*

    $Av = \lambda v$ — $A$ only scales $v$, never rotates it off its line.
    Symmetric $A$: $A = Q\Lambda Q^\top$ with orthogonal eigenvectors.
    """)
    return


@app.cell
def _(helpers, mo):
    import numpy as np

    values, _, _fig = helpers.eigendecomposition_demo()
    eigen = helpers.display(_fig, mo)
    print("Eigenvalues:", np.round(values, 3))
    return (eigen,)


@app.cell
def _(mo):
    mo.md(r"""## Spectral theorem — reconstruct $A = Q\Lambda Q^\top$""")
    return


@app.cell
def _(helpers):
    err = helpers.spectral_reconstruction_error(helpers.COV_2X2)
    print(f"Reconstruction error ‖A - QΛQᵀ‖: {err}")
    return


if __name__ == "__main__":
    app.run()
