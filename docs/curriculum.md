# Curriculum

Textbook-driven self-study tracks. Each row links a book to its notebook folder and current coverage.

| Textbook | Authors | Status | Notebook path |
|----------|---------|--------|---------------|
| [Advances in Financial Machine Learning](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086) | López de Prado (2018) | Ch. 2–4 | [`textbooks/financial-machine-learning/`](../textbooks/financial-machine-learning/) |
| [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/) | Hastie, Tibshirani & Friedman (2nd ed.) | Ch. 2–18 | [`textbooks/elements-of-statistical-learning/`](../textbooks/elements-of-statistical-learning/) |
| [Deep Learning](https://www.deeplearningbook.org/) | Goodfellow, Bengio & Courville (2016) | Ch. 2–5 | [`textbooks/deep-learning/`](../textbooks/deep-learning/) |

## Advances in Financial Machine Learning

Quantitative finance track: alternative bar types, event-driven sampling, triple-barrier labeling, and sample weighting for overlapping labels.

| Chapter | Topic |
|---------|-------|
| 2 | Financial data structures — bars, CUSUM filter, PCA weights |
| 3 | Labeling — triple-barrier method |
| 4 | Sample weights — concurrency, uniqueness, time decay |

## Elements of Statistical Learning

Core statistics and machine learning from linear models through high-dimensional methods.

| Chapters | Topics |
|----------|--------|
| 2–3 | Supervised learning, linear methods |
| 4 | Linear classification (LDA, logistic regression, SVMs) |
| 5–6 | Basis expansions, kernel smoothing |
| 7–8 | Model assessment, bootstrap, bagging |
| 9–10 | Additive models, trees, boosting |
| 11–12 | Neural networks, SVMs, flexible discriminants |
| 13 | Prototype methods, KNN |
| 14 | Unsupervised learning — clustering, PCA, NMF |
| 15 | Random forests |
| 16–18 | Ensemble learning, graphical models, high-dimensional problems |

## Deep Learning

Foundations for the deep learning track: linear algebra, probability, information theory, and structured models (Goodfellow et al., Part I).

| Chapter | Topic |
|---------|-------|
| 2 | Linear algebra — vectors, norms, eigendecomposition, SVD, PCA |
| 3 | Probability and information theory — random variables, Bayes' rule, entropy, KL divergence, graphical models |
| 4 | Numerical computation — stable softmax, conditioning, gradient descent, Newton's method, least squares |
| 5 | Machine learning basics — capacity, validation, bias-variance, MLE, SGD |

## Planned tracks

Future chapters from *Deep Learning* (deep feedforward networks and beyond) and other textbooks — pure maths (real analysis, linear algebra), additional applied stats/ML books. New tracks get a folder under `textbooks/` named after the book or a short slug, plus shared utilities in `maths_self_study` where they apply across books.
