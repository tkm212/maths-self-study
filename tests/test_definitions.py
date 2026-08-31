"""Tests for textbook-style definition and theorem boxes."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import (
    algorithm_box,
    algorithm_group,
    definition_box,
    definition_group,
    observation_box,
    observation_group,
    proof_box,
    proof_group,
    theorem_box,
    theorem_group,
)
from maths_self_study.dashboards.layout import page_shell
from maths_self_study.viz.textbooks.deep_learning.ch2 import definitions as ch2
from maths_self_study.viz.textbooks.deep_learning.ch2 import theorems as th2
from maths_self_study.viz.textbooks.deep_learning.ch3 import definitions as ch3
from maths_self_study.viz.textbooks.deep_learning.ch3 import proofs as pr3
from maths_self_study.viz.textbooks.deep_learning.ch3 import theorems as th3
from maths_self_study.viz.textbooks.deep_learning.ch4 import definitions as ch4
from maths_self_study.viz.textbooks.deep_learning.ch4 import proofs as pr4
from maths_self_study.viz.textbooks.deep_learning.ch4 import theorems as th4
from maths_self_study.viz.textbooks.deep_learning.ch5 import definitions as ch5
from maths_self_study.viz.textbooks.deep_learning.ch5 import proofs as pr5
from maths_self_study.viz.textbooks.deep_learning.ch5 import theorems as th5
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch2 import definitions as esl2
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch2 import theorems as esl_th2
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch3 import definitions as esl3
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch4 import definitions as esl4
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch4 import proofs as esl_pr4
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch4 import theorems as esl_th4
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch5 import definitions as esl5
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch5 import theorems as esl_th5
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch6 import definitions as esl6
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch6 import theorems as esl_th6
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch7 import definitions as esl7
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch7 import proofs as esl_pr7
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch7 import theorems as esl_th7
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch8 import definitions as esl8
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch8 import proofs as esl_pr8
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch8 import theorems as esl_th8
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch9 import definitions as esl9
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch9 import theorems as esl_th9
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch10 import definitions as esl10


def test_definition_box_renders_term_and_body():
    box = definition_box("Linear map", r"A function $T$ is linear if $T(ax + by) = aT(x) + bT(y)$.")
    rendered = str(box)
    assert "definition-box" in rendered
    assert "Definition" in rendered
    assert "Linear map" in rendered
    assert "math-latex-source" in rendered


def test_definition_group_stacks_items():
    group = definition_group(
        ("Bias", "Error from an overly rigid model."),
        ("Variance", "Error from fitting noise."),
    )
    assert "definition-group" in str(group)
    assert group.children is not None
    assert len(group.children) == 2


def test_algorithm_box_renders_name_and_steps():
    box = algorithm_box(
        "AdaBoost.M1",
        [r"Initialise $w_i = 1/N$.", r"Update $w_i \leftarrow w_i \exp(\alpha_m)$."],
    )
    rendered = str(box)
    assert "algorithm-box" in rendered
    assert "Algorithm" in rendered
    assert "AdaBoost.M1" in rendered
    assert "math-latex-source" in rendered


def test_algorithm_group_stacks_items():
    group = algorithm_group(
        ("EM", [r"E-step: compute $r_{ik}$.", r"M-step: update $\mu_k$."]),
        ("Backfit", [r"Fit $f_j$ to partial residuals."]),
    )
    assert "algorithm-group" in str(group)
    assert group.children is not None
    assert len(group.children) == 2


def test_page_shell_includes_algorithm():
    shell = page_shell(
        "Title",
        "Caption",
        html.Div("filters"),
        "body-id",
        algorithm=("Gradient descent", [r"Update $x \leftarrow x - \eta \nabla f(x)$."]),
    )
    rendered = str(shell)
    assert "algorithm-group" in rendered
    assert "Gradient descent" in rendered


def test_proof_box_renders_name_and_steps():
    box = proof_box(
        "Bias-variance decomposition",
        [r"Write $y = f(x_0) + \varepsilon$.", r"Expand $\mathbb{E}[(y-\hat{f})^2]$."],
    )
    rendered = str(box)
    assert "proof-box" in rendered
    assert "Proof" in rendered
    assert "Bias-variance decomposition" in rendered
    assert "math-latex-source" in rendered


def test_proof_group_stacks_items():
    group = proof_group(
        ("Lemma A", [r"Step one."]),
        ("Lemma B", [r"Step two."]),
    )
    assert "proof-group" in str(group)
    assert group.children is not None
    assert len(group.children) == 2


def test_page_shell_includes_proof():
    shell = page_shell(
        "Title",
        "Caption",
        html.Div("filters"),
        "body-id",
        proof=("Bias-variance", [r"$\mathbb{E}[(y-\hat{f})^2] = \mathrm{bias}^2 + \mathrm{var} + \sigma^2$."]),
    )
    rendered = str(shell)
    assert "proof-group" in rendered
    assert "Bias-variance" in rendered


def test_page_shell_includes_definitions():
    shell = page_shell(
        "Title",
        "Caption",
        html.Div("filters"),
        "body-id",
        definitions=[("Norm", "A function assigning vector length.")],
        methodology=["Step one"],
    )
    rendered = str(shell)
    assert "definition-group" in rendered
    assert "How it works" in rendered


def test_theorem_box_renders_name_and_statement():
    box = theorem_box("Spectral theorem", r"Every real symmetric $A$ admits $A = Q \Lambda Q^\top$.")
    rendered = str(box)
    assert "theorem-box" in rendered
    assert "Theorem" in rendered
    assert "Spectral theorem" in rendered
    assert "math-latex-source" in rendered


def test_theorem_group_stacks_items():
    group = theorem_group(
        ("Bayes' theorem", "Posterior equals likelihood times prior, normalised."),
        ("Chain rule", "Joint probability factorises as a product of conditionals."),
    )
    assert "theorem-group" in str(group)
    assert group.children is not None
    assert len(group.children) == 2


def test_page_shell_includes_theorems():
    shell = page_shell(
        "Title",
        "Caption",
        html.Div("filters"),
        "body-id",
        theorems=[("Cauchy-Schwarz", "Inner products are bounded by norm products.")],
    )
    rendered = str(shell)
    assert "theorem-group" in rendered
    assert "Cauchy-Schwarz" in rendered


def test_observation_box_renders_name_and_note():
    box = observation_box(
        "Precision-recall tradeoff",
        "Abstaining on low-confidence signals improves precision at the cost of fewer trades.",
    )
    rendered = str(box)
    assert "observation-box" in rendered
    assert "Observation" in rendered
    assert "Precision-recall tradeoff" in rendered


def test_observation_group_stacks_items():
    group = observation_group(
        ("Tradeoff", "Fewer bets when filtering aggressively."),
        ("Sizing", "Scale position with meta-model probability."),
    )
    assert "observation-group" in str(group)
    assert group.children is not None
    assert len(group.children) == 2


def test_page_shell_includes_observations():
    shell = page_shell(
        "Title",
        "Caption",
        html.Div("filters"),
        "body-id",
        observations=[("Precision-recall tradeoff", "Abstaining improves precision.")],
    )
    rendered = str(shell)
    assert "observation-group" in rendered
    assert "Precision-recall tradeoff" in rendered


def test_chapter_definition_modules_are_nonempty():
    modules = (
        (ch2, ["VECTORS", "NORMS", "EIGEN", "SVD", "PCA", "TENSORS"]),
        (ch3, ["RANDOM_VARIABLES", "DISTRIBUTIONS", "BAYES", "INFORMATION", "MARKOV"]),
        (ch4, ["STABILITY", "CONDITIONING", "GRADIENT_DESCENT", "NEWTON", "LEAST_SQUARES", "KKT"]),
        (ch5, ["CAPACITY", "VALIDATION", "BIAS_VARIANCE", "MLE", "MANIFOLD", "SGD"]),
        (esl2, ["K_NEAREST_NEIGHBORS", "LEAST_SQUARES"]),
        (esl3, ["SUBSET_SELECTION", "RIDGE", "LASSO", "PCR_PLS"]),
        (esl4, ["LOGISTIC_REGRESSION", "LDA", "SEPARATING_HYPERPLANES"]),
        (esl5, ["SPLINES", "SMOOTHING_SPLINES"]),
        (esl6, ["KERNEL_SMOOTHERS", "KERNEL_DENSITY"]),
        (esl7, ["BIAS_VARIANCE", "CROSS_VALIDATION"]),
        (esl8, ["EM_ALGORITHM", "BAGGING"]),
        (esl9, ["ADDITIVE_MODELS", "DECISION_TREES"]),
        (esl10, ["BOOSTING", "GRADIENT_BOOSTING"]),
    )
    for module, names in modules:
        for name in names:
            items = getattr(module, name)
            assert len(items) >= 1
            for term, definition in items:
                assert term.strip()
                assert len(definition.strip()) > 10


def test_chapter_theorem_modules_are_nonempty():
    modules = (
        (th2, ["NORMS", "EIGEN", "SVD", "PCA"]),
        (th3, ["RANDOM_VARIABLES", "BAYES", "INFORMATION", "MARKOV"]),
        (th4, ["STABILITY", "CONDITIONING", "LEAST_SQUARES", "KKT"]),
        (th5, ["BIAS_VARIANCE", "MLE"]),
        (esl_th2, ["LEAST_SQUARES"]),
        (esl_th4, ["LDA", "SEPARATING_HYPERPLANES"]),
        (esl_th5, ["SMOOTHING_SPLINES"]),
        (esl_th6, ["KERNEL_SMOOTHERS"]),
        (esl_th7, ["BIAS_VARIANCE"]),
        (esl_th8, ["EM_ALGORITHM", "BAGGING"]),
        (esl_th9, ["DECISION_TREES"]),
    )
    for module, names in modules:
        for name in names:
            items = getattr(module, name)
            assert len(items) >= 1
            for title, statement in items:
                assert title.strip()
                assert len(statement.strip()) > 10


def test_chapter_proof_modules_are_nonempty():
    modules = (
        (pr3, ["BAYES", "GIBBS"]),
        (pr4, ["LEAST_SQUARES", "LOG_SUM_EXP"]),
        (pr5, ["BIAS_VARIANCE", "MLE"]),
        (esl_pr4, ["LDA", "PERCEPTRON"]),
        (esl_pr7, ["BIAS_VARIANCE"]),
        (esl_pr8, ["EM_ALGORITHM", "BAGGING"]),
    )
    for module, names in modules:
        for name in names:
            title, steps = getattr(module, name)
            assert title.strip()
            assert len(steps) >= 3
            for step in steps:
                assert len(step.strip()) > 10
