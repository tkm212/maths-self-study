# Migration guide

## `financial-machine-learning` → `maths-self-study` (0.1.0)

Version **0.1.0** renames the project and Python package. If you installed or imported the old name, update as follows.

### Package import

```python
# before (≤ 0.0.1)
from financial_machine_learning.bars import dollar_bars
from financial_machine_learning.filters import cusum_filter

# after (≥ 0.1.0)
from maths_self_study.bars import dollar_bars
from maths_self_study.filters import cusum_filter
```

### Install

```bash
# before
pip install financial-machine-learning
uv add financial-machine-learning

# after
pip install maths-self-study
uv add maths-self-study
```

### Dataset loaders module

The loaders module was renamed for use across multiple textbooks (not ESL-only):

```python
# before
from maths_self_study.esl_loaders import load_tmdb_revenue_regression

# after
from maths_self_study.loaders import load_tmdb_revenue_regression
```

### Repository clone URL

After the GitHub repository is renamed to `maths-self-study`:

```bash
git remote set-url origin git@github.com:tkm212/maths-self-study.git
```

### What did not change

- Notebook paths under `notebooks/financial-machine-learning/` still refer to the AFML textbook folder name.
- AFML module names (`bars`, `filters`, `labeling`, `weights`) and their APIs are unchanged.
- Marimo notebook filenames and chapter layout are unchanged.

### GitHub repository settings (maintainers)

After merging the restructure, update on GitHub:

1. **Settings → General → Repository name:** `maths-self-study`
2. **About:** *Marimo notebooks and Python code for self-directed study in maths, statistics, and machine learning — chapter-by-chapter from classic textbooks.*
3. **Topics:** `self-study`, `statistics`, `machine-learning`, `marimo`, `textbooks`, `quantitative-finance`
4. Verify **Codecov** and **GitHub Pages** still point at the new repo name (`tkm212.github.io/maths-self-study`).
