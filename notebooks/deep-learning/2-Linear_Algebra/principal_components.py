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
    # Principal Components Analysis — Deep Learning Ch. 2 §2.12

    *Goodfellow, Bengio & Courville (2016). [Deep Learning](https://www.deeplearningbook.org/contents/linear_algebra.html).*

    PCA finds an orthogonal basis aligned with the directions of greatest variance. Given centred data matrix $\tilde{X}$, the covariance $C = \frac{1}{m-1}\tilde{X}^\top \tilde{X}$ has eigenvectors that are the principal components.

    Encoding: $c = W^\top (x - \mu)$; decoding: $\hat{x} = W c + \mu$.
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

    from maths_self_study.linalg import pca_fit, pca_inverse_transform, pca_transform

    rng = np.random.default_rng(42)
    z = rng.normal(size=(300, 2))
    transform = np.array([[3.0, 1.0], [0.0, 0.5]])
    data = z @ transform + np.array([2.0, -1.0])

    model = pca_fit(data, n_components=2)
    codes = pca_transform(model, data)
    reconstructed = pca_inverse_transform(model, codes)

    print("Explained variance:", model.explained_variance)
    print("Reconstruction error (Frobenius):", float(np.linalg.norm(reconstructed - data)))

    fig_data = helpers.plot_pca_scatter(data, title="Original feature space")
    fig_codes = helpers.plot_pca_codes(codes, title="PCA code space")
    fig_var = helpers.plot_explained_variance(model.explained_variance)

    mo.vstack([
        helpers.display(fig_data, mo),
        helpers.display(fig_codes, mo),
        helpers.display(fig_var, mo),
    ])
    return (
        codes,
        data,
        fig_codes,
        fig_data,
        fig_var,
        model,
        np,
        pca_fit,
        pca_inverse_transform,
        pca_transform,
        reconstructed,
        rng,
        transform,
        z,
    )


if __name__ == "__main__":
    app.run()
