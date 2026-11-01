"""Tests for Deep Learning Ch. 10 sequence-model demos."""

from __future__ import annotations

import numpy as np

from maths_self_study.math.sequence import (
    architecture_demo,
    chain_depth,
    clip_gradient,
    context_reconstruction_mse,
    echo_state_demo,
    eigenvalue_power,
    gated_demo,
    leaky_unit,
    log10_abs,
    long_term_demo,
    memory_attention,
    rnn_demo,
    sigmoid,
    softmax,
    tanh_compositions,
    tree_depth,
    unfolding_demo,
)


def test_sigmoid_and_softmax_are_finite():
    assert np.isclose(float(sigmoid(0.0)), 0.5)
    probs = softmax(np.array([1.0, 2.0, 0.5]))
    assert probs.shape == (3,)
    assert np.isclose(probs.sum(), 1.0)
    assert int(np.argmax(probs)) == 1


def test_log10_abs_caps_huge_values():
    assert log10_abs(0.0) == -16.0
    assert log10_abs(1e20) == 12.0


def test_unfolding_input_moves_the_driven_state():
    quiet = unfolding_demo(0.8, 0.0, 4)
    driven = unfolding_demo(0.8, 1.5, 4)
    assert np.allclose(quiet.autonomous, quiet.driven)
    assert np.max(np.abs(driven.driven - driven.autonomous)) > 0.2
    clipped = unfolding_demo(0.5, 1.0, 100, steps=6)
    assert clipped.drive[-1] == 1.0


def test_teacher_forcing_beats_free_run_when_the_weight_is_wrong():
    matched = rnn_demo(0.9, 3)
    mismatched = rnn_demo(0.3, 3)
    assert matched.teacher_mse == 0.0
    assert mismatched.free_mse > mismatched.teacher_mse
    held = rnn_demo(1.1, 2)
    forgotten = rnn_demo(0.2, 2)
    assert abs(held.hidden[-1]) > abs(forgotten.hidden[-1])
    assert held.grad_log10[0] > forgotten.grad_log10[0]


def test_context_mse_falls_as_the_register_grows():
    short = context_reconstruction_mse(12, 1)
    wide = context_reconstruction_mse(12, 8)
    assert short > wide
    assert context_reconstruction_mse(8, 8) < 0.05


def test_tree_path_is_shorter_than_the_chain():
    assert chain_depth(1) == 0
    assert tree_depth(1) == 0
    demo = architecture_demo(12, 4, 2)
    assert demo.tree_steps < demo.chain_steps
    assert demo.stacked_steps == demo.chain_steps * 2
    spike = int(np.argmax(demo.signal))
    assert demo.backward[spike - 2] > demo.forward[spike - 2]
    assert demo.reconstruction_mse[0] > demo.reconstruction_mse[-1]


def test_eigenvalue_power_and_leaky_memory():
    assert np.isclose(eigenvalue_power(0.5, 4)[-1], 0.5**3)
    _, fast, _ = leaky_unit(0.1, steps=20, pulse_time=2)
    _, slow, _ = leaky_unit(0.95, steps=20, pulse_time=2)
    assert slow[-1] > fast[-1]
    _, _, pulse = leaky_unit(0.5, steps=5, pulse_time=-3)
    assert pulse[0] == 1.0
    init, snaps = tanh_compositions(1.2)
    assert init.shape == snaps[12].shape
    assert np.max(np.abs(snaps[12])) < np.max(np.abs(init))


def test_echo_state_forgets_long_delays_when_the_radius_is_tiny():
    faded = echo_state_demo(0.05)
    kept = echo_state_demo(0.9)
    assert faded.delay_mse[0] < 0.05
    assert faded.delay_mse[-1] > 0.2
    assert kept.delay_mse[1] < faded.delay_mse[1]


def test_clip_gradient_respects_the_threshold():
    raw = np.array([3.0, -4.0])
    tight = clip_gradient(raw, 1.0)
    loose = clip_gradient(raw, 100.0)
    assert np.isclose(np.linalg.norm(tight), 1.0)
    assert np.allclose(loose, raw)
    demo = long_term_demo(1.2, 0.9, 0.9, 1.5, steps=12)
    assert np.linalg.norm(demo.grad_clipped) <= 1.5 + 1e-8
    assert np.linalg.norm(demo.grad_raw) > 1.5
    assert demo.power.shape == (12,)


def test_lstm_forget_bias_holds_the_cue():
    remembered = gated_demo(5.0, 2, 2)
    forgotten = gated_demo(-3.0, 2, 2)
    assert remembered.lstm_cell[-1] > 0.5
    assert forgotten.lstm_cell[-1] < 0.05
    assert remembered.lstm_grad_log10 > remembered.rnn_grad_log10
    assert int(np.argmax(remembered.attention)) == 2
    weights, values, read_value = memory_attention(0)
    assert int(np.argmax(weights)) == 0
    assert values[0] == 1.0
    assert 1.0 <= read_value <= 2.0
    wrapped = gated_demo(4.0, 0, 9, steps=10)
    assert wrapped.attention.shape == (4,)
