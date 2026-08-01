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
    # Eigendecomposition — Deep Learning Ch. 2 §2.7

    *Goodfellow, Bengio & Courville (2016). [Deep Learning](https://www.deeplearningbook.org/contents/linear_algebra.html).*

    For a square matrix $A$, an **eigenvector** $v \neq 0$ and **eigenvalue** $\lambda$ satisfy

    $$Av = \lambda v.$$

    If $A$ is symmetric, eigenvectors form an orthonormal basis and $A = Q \Lambda Q^\top$.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    import ch2_helpers as helpers

    return (helpers,)


@app.cell
def _(helpers, mo):
    import numpy as np

    from maths_self_study.linalg import symmetric_eigendecomposition

    cov = np.array([[2.0, 0.8], [0.8, 1.0]])
    values, vectors = symmetric_eigendecomposition(cov)

    print("Eigenvalues:", values)
    print("Eigenvectors (columns):\n", vectors)

    fig = helpers.plot_vectors_2d(
        np.zeros(2),
        [vectors[:, 0], vectors[:, 1]],
        labels=["v₁", "v₂"],
        title="Eigenvectors of a 2x2 covariance matrix",
    )
    helpers.display(fig, mo)
    return cov, fig, np, symmetric_eigendecomposition, values, vectors


if __name__ == "__main__":
    app.run()
