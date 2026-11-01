"""Shared plotting helpers for Deep Learning Ch. 10 (Sequence Modeling)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from maths_self_study.math.sequence import (
    ECHO_DELAYS,
    architecture_demo,
    gated_demo,
    long_term_demo,
    rnn_demo,
    unfolding_demo,
)
from maths_self_study.viz.graphs import bar_chart, line_chart

THETA_DEFAULT = 0.8
INPUT_GAIN_DEFAULT = 1.0
PULSE_DEFAULT = 4
RECURRENCE_DEFAULT = 0.9
CUE_DEFAULT = 3
LENGTH_DEFAULT = 12
CONTEXT_DEFAULT = 3
STACK_DEFAULT = 2
MAGNITUDE_DEFAULT = 1.2
ALPHA_DEFAULT = 0.85
RADIUS_DEFAULT = 0.9
CLIP_DEFAULT = 1.5
FORGET_DEFAULT = 4.0
GATED_CUE_DEFAULT = 2
QUERY_DEFAULT = 1

_BLUE = "#2563eb"
_RED = "#dc2626"
_GREEN = "#16a34a"
_AMBER = "#d97706"
_VIOLET = "#7c3aed"


def _line(
    x,
    y,
    *,
    name: str,
    color: str,
    title: str | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    height: int = 360,
    dash: str | None = None,
    fig: go.Figure | None = None,
) -> go.Figure:
    return line_chart(
        x,
        y,
        name=name,
        color=color,
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=height,
        line_dash=dash,
        fig=fig,
        legend="horizontal",
    )


def plot_unfolding(
    theta: float,
    input_gain: float,
    pulse_time: int,
) -> tuple[go.Figure, go.Figure, dict[str, float]]:
    """Autonomous vs driven state, and the product of tanh Jacobians (§10.1)."""
    demo = unfolding_demo(theta, input_gain, pulse_time)
    fig_state = _line(
        demo.time,
        demo.autonomous,
        name="autonomous",
        color=_BLUE,
        title="State with and without an input pulse",
        xaxis_title="t",
        yaxis_title="s",
    )
    _line(demo.time, demo.driven, name="driven by x", color=_RED, fig=fig_state)
    _line(demo.time, demo.drive, name="input x", color=_AMBER, dash="dot", fig=fig_state)
    fig_grad = _line(
        demo.time,
        demo.grad_log10,
        name="log10 |ds_t / ds_0|",
        color=_VIOLET,
        title="Unfolded derivative back to the initial state",
        xaxis_title="t",
        yaxis_title="log10 |gradient|",
    )
    stats = {
        "final_autonomous": float(demo.autonomous[-1]),
        "final_driven": float(demo.driven[-1]),
        "grad_log10_final": float(demo.grad_log10[-1]),
    }
    return fig_state, fig_grad, stats


def plot_rnn(recurrence: float, cue_time: int) -> tuple[go.Figure, go.Figure, go.Figure, dict[str, float]]:
    """Teacher forcing, free run, and Elman memory with a BPTT gradient (§10.2)."""
    demo = rnn_demo(recurrence, cue_time)
    fig_out = _line(
        demo.time,
        demo.target,
        name="target",
        color="#64748b",
        title="Output recurrence: teacher forcing vs free run",
        xaxis_title="t",
        yaxis_title="y",
    )
    _line(demo.time, demo.teacher, name="teacher forcing", color=_BLUE, fig=fig_out)
    _line(demo.time, demo.free_run, name="free run", color=_RED, fig=fig_out)
    fig_hidden = _line(
        demo.time,
        demo.hidden,
        name="hidden",
        color=_GREEN,
        title="Elman hidden state after a unit cue",
        xaxis_title="t",
        yaxis_title="h",
    )
    fig_grad = _line(
        demo.time,
        demo.grad_log10,
        name="log10 |dh_T / dh_t|",
        color=_VIOLET,
        title="BPTT gradient of the final state",
        xaxis_title="t",
        yaxis_title="log10 |gradient|",
    )
    stats = {
        "teacher_mse": demo.teacher_mse,
        "free_mse": demo.free_mse,
        "hidden_final": float(demo.hidden[-1]),
        "early_grad_log10": float(demo.grad_log10[0]),
    }
    return fig_out, fig_hidden, fig_grad, stats


def plot_architectures(
    length: int,
    context_dim: int,
    stack_depth: int,
) -> tuple[go.Figure, go.Figure, go.Figure, dict[str, float]]:
    """Bidirectional evidence, context bottleneck, and path length (§10.3-§10.6)."""
    demo = architecture_demo(length, context_dim, stack_depth)
    fig_bi = _line(
        demo.time,
        demo.label,
        name="label (needs the future spike)",
        color="#64748b",
        title="Forward state misses a future spike; backward state sees it",
        xaxis_title="t",
        yaxis_title="activation",
        dash="dot",
    )
    _line(demo.time, demo.forward, name="forward", color=_BLUE, fig=fig_bi)
    _line(demo.time, demo.backward, name="backward", color=_RED, fig=fig_bi)
    fig_mse = _line(
        demo.context_dims,
        demo.reconstruction_mse,
        name="reconstruction MSE",
        color=_AMBER,
        title="Encoder context size vs sequence reconstruction MSE",
        xaxis_title="context dimension",
        yaxis_title="MSE",
        height=340,
    )
    fig_depth = bar_chart(
        ["Chain RNN", "Recursive tree", "Deep transition"],
        [demo.chain_steps, demo.tree_steps, demo.stacked_steps],
        title="Gradient path length",
        yaxis_title="steps",
        color=_BLUE,
        height=340,
    )
    k = int(np.clip(context_dim, 1, demo.context_dims[-1]))
    mse_at_k = float(demo.reconstruction_mse[k - 1])
    stats = {
        "mse_at_context": mse_at_k,
        "chain_steps": float(demo.chain_steps),
        "tree_steps": float(demo.tree_steps),
        "stacked_steps": float(demo.stacked_steps),
    }
    return fig_bi, fig_mse, fig_depth, stats


def plot_long_term(
    magnitude: float,
    alpha: float,
    spectral_radius: float,
    clip_threshold: float,
) -> tuple[go.Figure, go.Figure, go.Figure, go.Figure, go.Figure, go.Figure, dict[str, float]]:
    """Eigenvalue powers, leaky units, echo-state recall, and clipping (§10.7-§10.11)."""
    demo = long_term_demo(magnitude, alpha, spectral_radius, clip_threshold)
    power_log = np.log10(np.maximum(demo.power, 1e-16))
    fig_power = _line(
        demo.time,
        power_log,
        name="log10 |λ|^t",
        color=_RED,
        title="Linear recurrence: eigenvalues raised to t",
        xaxis_title="t",
        yaxis_title="log10 |λ|^t",
    )
    _line(demo.time, np.zeros_like(demo.time, dtype=float), name="|λ| = 1", color="#64748b", dash="dot", fig=fig_power)
    compose_steps = (1, 4, 12)
    compose_colors = (_BLUE, _AMBER, _RED)
    fig_comp = _line(
        demo.init,
        demo.composed[compose_steps[0]],
        name=f"after {compose_steps[0]} steps",
        color=compose_colors[0],
        title="Repeated tanh, as in figure 10.15",
        xaxis_title="initial state",
        yaxis_title="composed output",
    )
    for step, color in zip(compose_steps[1:], compose_colors[1:], strict=True):
        _line(
            demo.init,
            demo.composed[step],
            name=f"after {step} steps",
            color=color,
            fig=fig_comp,
        )
    fig_leak = _line(
        demo.leak_time,
        demo.leak_state,
        name="leaky state",
        color=_GREEN,
        title="Leaky unit after a unit pulse",
        xaxis_title="t",
        yaxis_title="h",
    )
    _line(demo.leak_time, demo.leak_input, name="input", color=_AMBER, dash="dot", fig=fig_leak)
    sl = slice(0, 48)
    fig_echo = _line(
        demo.echo_time[sl],
        demo.echo_target[sl],
        name="input delayed 5 steps",
        color="#64748b",
        title="Echo-state readout (reservoir weights fixed)",
        xaxis_title="t",
        yaxis_title="value",
    )
    _line(demo.echo_time[sl], demo.echo_pred[sl], name="linear readout", color=_BLUE, fig=fig_echo)
    labels = [str(d) for d in ECHO_DELAYS]
    fig_delays = bar_chart(
        labels,
        demo.delay_mse,
        title="Recall MSE by delay",
        xaxis_title="delay",
        yaxis_title="MSE",
        color=_VIOLET,
        height=320,
    )
    raw_norm = float(np.linalg.norm(demo.grad_raw))
    clipped_norm = float(np.linalg.norm(demo.grad_clipped))
    idx = [str(i) for i in range(demo.grad_raw.size)]
    fig_clip = bar_chart(
        idx,
        demo.grad_raw,
        name="raw",
        title="Gradient components before and after clipping",
        xaxis_title="component",
        yaxis_title="value",
        color=_RED,
        height=320,
        barmode="group",
        legend="horizontal",
    )
    bar_chart(idx, demo.grad_clipped, name="clipped", color=_BLUE, fig=fig_clip)
    stats = {
        "log10_power_final": float(power_log[-1]),
        "leak_final": float(demo.leak_state[-1]),
        "echo_mse": demo.echo_mse,
        "raw_grad_norm": raw_norm,
        "clipped_grad_norm": clipped_norm,
    }
    return fig_power, fig_comp, fig_leak, fig_echo, fig_delays, fig_clip, stats


def plot_gated(
    forget_bias: float,
    cue_time: int,
    query_index: int,
) -> tuple[go.Figure, go.Figure, go.Figure, dict[str, float]]:
    """LSTM and GRU memory of one cue, plus a soft memory read (§10.10, §10.12)."""
    demo = gated_demo(forget_bias, cue_time, query_index)
    fig_mem = _line(
        demo.time,
        demo.target,
        name="cue target",
        color="#64748b",
        title="Remembering a cue: tanh RNN vs LSTM cell vs GRU",
        xaxis_title="t",
        yaxis_title="state",
        dash="dot",
    )
    _line(demo.time, demo.rnn, name="vanilla RNN", color=_RED, fig=fig_mem)
    _line(demo.time, demo.lstm_cell, name="LSTM cell", color=_BLUE, fig=fig_mem)
    _line(demo.time, demo.gru, name="GRU", color=_GREEN, fig=fig_mem)
    fig_grad = bar_chart(
        ["vanilla RNN", "LSTM cell"],
        [demo.rnn_grad_log10, demo.lstm_grad_log10],
        title="log10 gradient across the gap after the cue",
        yaxis_title="log10 |gradient|",
        color=_VIOLET,
        height=320,
    )
    slots = [f"slot {i} (value {int(v)})" for i, v in enumerate(demo.slot_values)]
    fig_attn = bar_chart(
        slots,
        demo.attention,
        title="Soft address of an explicit memory",
        yaxis_title="attention",
        color=_AMBER,
        height=320,
    )
    stats = {
        "rnn_final": float(demo.rnn[-1]),
        "lstm_final": float(demo.lstm_cell[-1]),
        "gru_final": float(demo.gru[-1]),
        "read_value": demo.read_value,
    }
    return fig_mem, fig_grad, fig_attn, stats
