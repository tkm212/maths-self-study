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
    # Norms — Deep Learning Ch. 2 §2.5

    *Goodfellow, Bengio & Courville (2016). [Deep Learning](https://www.deeplearningbook.org/contents/linear_algebra.html).*

    The $L^p$ norm of a vector is

    $$\|x\|_p = \left(\sum_i |x_i|^p\right)^{1/p}.$$

    The Euclidean norm $\|x\|_2$ appears in weight decay and distance metrics; $\|x\|_1$ drives sparsity in Lasso-style regularisation.
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

    from maths_self_study.linalg import cosine_similarity, lp_norm

    x = np.array([3.0, -4.0])
    print(f"||x||_1 = {lp_norm(x, 1):.2f}")
    print(f"||x||_2 = {lp_norm(x, 2):.2f}")

    a = np.array([1.0, 0.0])
    b = np.array([1.0, 1.0])
    print(f"cosine(a, b) = {cosine_similarity(a, b):.4f}")
    return a, b, cosine_similarity, lp_norm, np, x


if __name__ == "__main__":
    app.run()
