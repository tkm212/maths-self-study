"""Tests for maths_self_study.dashboards.components and logging."""

from __future__ import annotations

import logging

import numpy as np

from maths_self_study.dashboards.components import (
    filter_bar,
    matrix_callback_inputs,
    matrix_cell_id,
    matrix_input,
    num_input,
    prob_simplex_ids,
    table,
    tensor_callback_inputs,
    tensor_cell_id,
    tensor_grid_input,
    text_box,
)
from maths_self_study.dashboards.logging import LOGGER, configure, configure_for_run


def test_matrix_callback_inputs():
    inputs = matrix_callback_inputs("grid-matrix")
    assert len(inputs) == 4
    assert matrix_cell_id("grid-matrix", 1, 2) == "grid-matrix-12"


def test_matrix_input_component():
    defaults = np.eye(2)
    block = matrix_input("test", "Matrix", defaults)
    assert block is not None


def test_tensor_grid_input_component():
    from maths_self_study.demos.deep_learning import ch2 as helpers

    block = tensor_grid_input("tensor", "T", helpers.TENSOR_DEFAULT, shape=helpers.TENSOR_SHAPE)
    assert block is not None


def test_tensor_callback_inputs():
    inputs = tensor_callback_inputs("tensor-grid", (2, 3, 3))
    assert len(inputs) == 18
    assert tensor_cell_id("tensor-grid", 1, 2, 3) == "tensor-grid-123"


def test_filter_bar_wraps_controls():
    bar = filter_bar(num_input("x", "x", 1.0))
    assert bar is not None


def test_table_component():
    block = table(["A", "B"], [["x", "1"], ["y", "2"]], caption="Demo")
    assert block is not None


def test_text_box_renders_steps():
    block = text_box(steps=["Definition", "Algorithm"], title="How it works")
    assert block is not None


def test_prob_simplex_ids():
    assert prob_simplex_ids("info-p", [0, 1, 2, 3]) == ["info-p0", "info-p1", "info-p2", "info-p3"]


def test_configure_logging():
    configure(level=logging.WARNING, force=True)
    assert logging.getLogger().level == logging.WARNING
    configure_for_run(debug=True)
    assert logging.getLogger().level == logging.INFO
    assert LOGGER.level == logging.DEBUG
    assert logging.getLogger("watchdog").level == logging.WARNING
    configure_for_run(debug=False)
    assert LOGGER.level == logging.INFO
