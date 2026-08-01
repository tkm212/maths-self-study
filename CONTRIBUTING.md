# Contributing to `maths-self-study`

Contributions are welcome — bug reports, notebook fixes, new textbook chapters, and documentation improvements.

## Where to start

- Browse [open issues](https://github.com/tkm212/maths-self-study/issues)
- Read the [notebooks overview](docs/notebooks.md) to see how chapters are organised
- Each ESL chapter folder has a `ch{N}_helpers.py` module; AFML notebooks import from `maths_self_study`

## Bug reports

Report bugs at https://github.com/tkm212/maths-self-study/issues

Include:

- Python version (`uv run python --version`)
- Steps to reproduce
- Expected vs actual behaviour

## Documentation

`maths-self-study` could always use more documentation, whether as part of the official docs, in docstrings, or even on the web in blog posts, articles, and such.

## Feedback

The best way to send feedback is to file an issue at https://github.com/tkm212/maths-self-study/issues.

## Development setup

Ready to contribute? Here's how to set up `maths-self-study` for local development.

1. Fork the `maths-self-study` repo on GitHub.
2. Clone your fork locally:

```bash
git clone git@github.com:YOUR_NAME/maths-self-study.git
```

3. Install dependencies and pre-commit hooks:

```bash
cd maths-self-study
make install
```

4. Create a branch for your changes:

```bash
git checkout -b my-feature
```

5. Run checks before committing:

```bash
make check
make test
```

6. Push to your fork and open a pull request.

## Notebook conventions

- ESL notebooks live under `notebooks/elements-of-statistical-learning/`
- AFML notebooks live under `notebooks/financial-machine-learning/`
- Use Marimo (`.py` files), not `.ipynb`
- Run with `uv run marimo run path/to/notebook.py`; edit with `uv run marimo edit ...`
- Shared plotting and data loading goes in `ch{N}_helpers.py` for that chapter

## Pull request checklist

- [ ] `make check` passes
- [ ] `make test` passes (≥ 80% coverage on `maths_self_study`)
- [ ] New notebooks follow existing Marimo + helper patterns
- [ ] Docs updated if you add public API or new notebook sections
