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
    # Linear maps as geometry — Deep Learning Ch. 2 §2.1-2.2

    *Goodfellow, Bengio & Courville (2016). [Deep Learning](https://www.deeplearningbook.org/contents/linear_algebra.html).*

    A matrix $A$ is a **linear map** $x \mapsto Ax$. Columns of $A$ are where the basis goes.
    Composition is associative but **not commutative**.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""## The grid picture""")
    return


@app.cell
def _(helpers, mo):
    _fig = helpers.plot_transformed_grid(helpers.GRID_MAP, title="A deforms the plane, but keeps it flat")
    grid = helpers.display(_fig, mo)
    return (grid,)


@app.cell
def _(mo):
    mo.md(r"""## Non-commutativity — rotation then shear ≠ shear then rotation""")
    return


@app.cell
def _(helpers, mo):
    _fig = helpers.plot_transformed_grid(
        helpers.rotation_2d(30) @ helpers.shear_2d(),
        title="R ∘ S (rotate after shear)",
        grid_range=1.2,
    )
    commute = helpers.display(_fig, mo)
    return (commute,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Inner product

    $x^\top y = \|x\|_2 \|y\|_2 \cos\theta$ — projection length.
    """)
    return


@app.cell
def _(helpers):
    print(helpers.inner_product_45deg())
    return


if __name__ == "__main__":
    app.run()
