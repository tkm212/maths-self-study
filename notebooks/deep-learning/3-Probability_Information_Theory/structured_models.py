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
    # Structured Probabilistic Models — Deep Learning Ch. 3 §3.14

    *Goodfellow, Bengio & Courville (2016). [Deep Learning](https://www.deeplearningbook.org/contents/prob.html).*

    **Structured models** factor the joint into local terms — for example a **chain**:

    $$P(x^{(1)}, \ldots, x^{(n)}) = P(x^{(1)}) \prod_{i=2}^{n} P(x^{(i)} \mid x^{(i-1)}).$$
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))

    return


@app.cell
def _():
    import numpy as np

    p_x1 = np.array([0.6, 0.4])
    p_x2_given_x1 = np.array([
        [0.7, 0.3],
        [0.2, 0.8],
    ])
    p_x3_given_x2 = np.array([
        [0.9, 0.1],
        [0.4, 0.6],
    ])

    joint = np.zeros((2, 2, 2))
    for x1 in (0, 1):
        for x2 in (0, 1):
            for x3 in (0, 1):
                joint[x1, x2, x3] = p_x1[x1] * p_x2_given_x1[x1, x2] * p_x3_given_x2[x2, x3]

    print("Joint table shape:", joint.shape)
    print("Sum of probabilities:", joint.sum())
    print("P(X3=1) =", joint[:, :, 1].sum())
    return joint, np, p_x1, p_x2_given_x1, p_x3_given_x2


if __name__ == "__main__":
    app.run()
