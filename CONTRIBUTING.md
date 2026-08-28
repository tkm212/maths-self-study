# Contributing to `maths-self-study`

Contributions are welcome — bug reports, notebook fixes, new textbook chapters, and documentation improvements.

## Where to start

- Browse [open issues](https://github.com/tkm212/maths-self-study/issues)
- Read the [notebooks overview](docs/notebooks.md) to see how chapters are organised
- Each ESL chapter folder has a `ch{N}_helpers.py` module; AFML chapters use Dash dashboards with helpers in `maths_self_study.demos.financial_machine_learning`

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

- ESL notebooks live under `textbooks/elements-of-statistical-learning/` (Marimo `.py` files)
- AFML chapters live under `textbooks/financial-machine-learning/` (Dash `dashboard.py` + `ch{N}_pages/`)
- Deep Learning chapters also use Dash dashboards (same page layout as AFML)
- ESL: run with `uv run marimo run path/to/notebook.py`; shared code in `ch{N}_helpers.py`
- AFML / Deep Learning: run with `uv run python path/to/dashboard.py`; plot helpers in `maths_self_study.demos.*`

## Pull request checklist

- [ ] `make check` passes
- [ ] `make test` passes (≥ 80% coverage on `maths_self_study`)
- [ ] New notebooks follow existing Marimo + helper patterns
- [ ] Docs updated if you add public API or new notebook sections
