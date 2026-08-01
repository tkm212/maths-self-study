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
    # SVD and the Moore-Penrose Pseudoinverse — Deep Learning Ch. 2 §2.8-2.9

    *Goodfellow, Bengio & Courville (2016). [Deep Learning](https://www.deeplearningbook.org/contents/linear_algebra.html).*

    Every matrix has a **singular value decomposition** $A = U \Sigma V^\top$. The **Moore-Penrose pseudoinverse** $A^+$ generalises matrix inversion to non-square or rank-deficient matrices and gives the minimum-norm least-squares solution to $Ax = b$.
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


@app.cell
def _():
    import numpy as np

    from maths_self_study.linalg import moore_penrose_pseudoinverse

    a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    b = np.array([1.0, 2.0, 3.0])
    pinv = moore_penrose_pseudoinverse(a)
    x = pinv @ b
    residual = a @ x - b

    print("Least-squares solution x:", x)
    print("||Ax - b||_2 =", float(np.linalg.norm(residual)))

    u, s, vt = np.linalg.svd(a, full_matrices=False)
    print("Singular values:", s)
    return a, b, moore_penrose_pseudoinverse, np, pinv, residual, s, u, vt, x


if __name__ == "__main__":
    app.run()
