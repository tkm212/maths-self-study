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
    # Scalars, Vectors, Matrices — Deep Learning Ch. 2 §2.1-2.2

    *Goodfellow, Bengio & Courville (2016). [Deep Learning](https://www.deeplearningbook.org/contents/linear_algebra.html).*

    A **scalar** is a single number. A **vector** is an array of numbers; we treat it as a column vector $x \in \mathbb{R}^n$. A **matrix** $A \in \mathbb{R}^{m \times n}$ maps vectors via matrix-vector product

    $$y = Ax, \qquad y_i = \sum_j A_{ij} x_j.$$

    Matrix multiplication is associative and distributes over addition, but is **not** commutative in general.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))

    import ch2_helpers

    ch2_helpers.init_paths()
    return (ch2_helpers,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Matrix-vector multiplication

    Apply a $2 \times 2$ rotation-like matrix to basis vectors $e_1 = (1,0)^\top$ and $e_2 = (0,1)^\top$.
    """)
    return


@app.cell
def _(ch2_helpers):
    import numpy as np

    a = np.array([[0.8, -0.6], [0.6, 0.8]])
    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.0, 1.0])
    y1 = a @ e1
    y2 = a @ e2

    ch2_helpers.plot_vectors_2d(
        np.zeros(2),
        [e1, e2, y1, y2],
        labels=["e1", "e2", "A e1", "A e2"],
        title="Matrix action on basis vectors",
    ).show()
    return a, e1, e2, np, y1, y2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Transpose and inner product

    The dot product $x^\top y = \sum_i x_i y_i$ is the matrix product of a row with a column.
    """)
    return


@app.cell
def _(np):
    x = np.array([1.0, 2.0, -1.0])
    y = np.array([2.0, -1.0, 3.0])
    print("x · y =", float(x @ y))
    print("x^T y =", float(x.T @ y))
    return x, y


if __name__ == "__main__":
    app.run()
