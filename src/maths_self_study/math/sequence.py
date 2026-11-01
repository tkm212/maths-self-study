"""Sequence-model utilities (Goodfellow, Bengio & Courville, Ch. 10).

NumPy demos for unfolded dynamics, teacher forcing, bidirectional and
encoder-decoder bottlenecks, vanishing gradients, echo-state readouts,
and gated memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TRUE_OUTPUT_COEF = 0.9
OUTPUT_DRIVE = 0.15
VANILLA_RECURRENCE = 0.55
ECHO_UNITS = 24
ECHO_STEPS = 120
ECHO_WASHOUT = 20
ECHO_DELAYS = (1, 5, 10, 20)


def sigmoid(x: np.ndarray | float) -> np.ndarray:
    z = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def softmax(scores: np.ndarray) -> np.ndarray:
    z = np.asarray(scores, dtype=float)
    z = z - np.max(z)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)


def log10_abs(value: float) -> float:
    magnitude = min(abs(float(value)), 1e12)
    return float(np.log10(max(magnitude, 1e-16)))


@dataclass(frozen=True)
class UnfoldingDemo:
    time: np.ndarray
    autonomous: np.ndarray
    driven: np.ndarray
    drive: np.ndarray
    grad_log10: np.ndarray


def unfolding_demo(
    theta: float,
    input_gain: float,
    pulse_time: int,
    *,
    steps: int = 24,
    s0: float = 0.8,
) -> UnfoldingDemo:
    """Autonomous state (eq. 10.1) and the same map driven by an input (eq. 10.4)."""
    n = max(4, int(steps))
    time = np.arange(n)
    drive = np.zeros(n)
    pulse = int(np.clip(pulse_time, 0, n - 1))
    drive[pulse] = 1.0
    autonomous = np.zeros(n)
    driven = np.zeros(n)
    autonomous[0] = s0
    driven[0] = s0
    weight = float(theta)
    gain = float(input_gain)
    for t in range(1, n):
        autonomous[t] = np.tanh(weight * autonomous[t - 1])
        driven[t] = np.tanh(weight * driven[t - 1] + gain * drive[t])
    grad = np.ones(n)
    for t in range(1, n):
        local = weight * (1.0 - autonomous[t] ** 2)
        grad[t] = grad[t - 1] * local
    return UnfoldingDemo(
        time=time,
        autonomous=autonomous,
        driven=driven,
        drive=drive,
        grad_log10=np.array([log10_abs(g) for g in grad]),
    )


@dataclass(frozen=True)
class RnnDemo:
    time: np.ndarray
    target: np.ndarray
    teacher: np.ndarray
    free_run: np.ndarray
    hidden: np.ndarray
    grad_log10: np.ndarray
    teacher_mse: float
    free_mse: float


def rnn_demo(recurrence: float, cue_time: int, *, steps: int = 24) -> RnnDemo:
    """Output recurrence with and without teacher forcing, plus an Elman memory trace.

    The scalar output model uses the same drive as a linear system with coefficient
    0.9 (section 10.2.1). The hidden trace is an Elman unit (section 10.2) asked to
    hold a cue, with BPTT gradient magnitude back to each time (section 10.2.2).
    """
    n = max(6, int(steps))
    weight = float(recurrence)
    time = np.arange(n)
    target = np.zeros(n)
    teacher = np.zeros(n)
    free_run = np.zeros(n)
    for t in range(1, n):
        drive = OUTPUT_DRIVE * np.sin(t / 3.0)
        target[t] = TRUE_OUTPUT_COEF * target[t - 1] + drive
        teacher[t] = weight * target[t - 1] + drive
        free_run[t] = weight * free_run[t - 1] + drive

    cue = int(np.clip(cue_time, 1, n - 2))
    inputs = np.zeros(n)
    inputs[cue] = 1.0
    hidden = np.zeros(n)
    for t in range(1, n):
        hidden[t] = np.tanh(weight * hidden[t - 1] + inputs[t])
    grad = np.ones(n)
    for t in range(n - 2, -1, -1):
        local = weight * (1.0 - hidden[t + 1] ** 2)
        grad[t] = grad[t + 1] * local
    return RnnDemo(
        time=time,
        target=target,
        teacher=teacher,
        free_run=free_run,
        hidden=hidden,
        grad_log10=np.array([log10_abs(g) for g in grad]),
        teacher_mse=float(np.mean((teacher - target) ** 2)),
        free_mse=float(np.mean((free_run - target) ** 2)),
    )


def context_reconstruction_mse(
    length: int,
    context_dim: int,
    *,
    n_seq: int = 256,
    seed: int = 0,
) -> float:
    """MSE of reading a full +/-1 sequence back from a length-k shift-register state.

    The register is a linear RNN whose hidden size is the encoder context (§10.4).
    Symbols that have shifted out of the state cannot be recovered.
    """
    n = max(2, int(length))
    k = max(1, int(context_dim))
    rng = np.random.default_rng(seed)
    seqs = rng.choice(np.array([-1.0, 1.0]), size=(n_seq, n))
    contexts = np.zeros((n_seq, k))
    for i, seq in enumerate(seqs):
        state = np.zeros(k)
        for value in seq:
            if k > 1:
                state[1:] = state[:-1]
            state[0] = value
        contexts[i] = state
    gram = contexts.T @ contexts + 1e-6 * np.eye(k)
    mse = 0.0
    for t in range(n):
        coef = np.linalg.solve(gram, contexts.T @ seqs[:, t])
        pred = contexts @ coef
        mse += float(np.mean((pred - seqs[:, t]) ** 2))
    return mse / n


def chain_depth(length: int) -> int:
    """Unfolded RNN path length for a sequence of ``length`` steps (§10.1, §10.6)."""
    return max(0, int(length) - 1)


def tree_depth(length: int) -> int:
    """Depth of a balanced binary tree over ``length`` leaves (§10.6)."""
    n = int(length)
    if n <= 1:
        return 0
    return int(np.ceil(np.log2(n)))


@dataclass(frozen=True)
class ArchitectureDemo:
    time: np.ndarray
    signal: np.ndarray
    label: np.ndarray
    forward: np.ndarray
    backward: np.ndarray
    context_dims: np.ndarray
    reconstruction_mse: np.ndarray
    chain_steps: int
    tree_steps: int
    stacked_steps: int


def architecture_demo(
    length: int,
    context_dim: int,
    stack_depth: int,
    *,
    max_context: int = 8,
) -> ArchitectureDemo:
    """Bidirectional evidence, encoder bottleneck, and chain vs tree path length.

    Sections 10.3-10.6. The label fires just before a future spike, so only the
    backward state can see it. Stacked depth multiplies the chain path (§10.5).
    """
    n = max(8, int(length))
    signal = np.zeros(n)
    spike = max(1, round(0.75 * (n - 1)))
    signal[spike] = 1.0
    horizon = 3
    label = np.zeros(n)
    label[max(0, spike - horizon) : spike + 1] = 1.0
    alpha = 0.55
    forward = np.zeros(n)
    for t in range(n):
        prev = forward[t - 1] if t else 0.0
        forward[t] = (1.0 - alpha) * prev + alpha * signal[t]
    backward = np.zeros(n)
    for t in range(n - 1, -1, -1):
        nxt = backward[t + 1] if t + 1 < n else 0.0
        backward[t] = (1.0 - alpha) * nxt + alpha * signal[t]
    dims = np.arange(1, max_context + 1)
    mse = np.array([context_reconstruction_mse(n, int(k)) for k in dims])
    chain = chain_depth(n)
    layers = max(1, int(stack_depth))
    return ArchitectureDemo(
        time=np.arange(n),
        signal=signal,
        label=label,
        forward=forward,
        backward=backward,
        context_dims=dims,
        reconstruction_mse=mse,
        chain_steps=chain,
        tree_steps=tree_depth(n),
        stacked_steps=chain * layers,
    )


@dataclass(frozen=True)
class LongTermDemo:
    time: np.ndarray
    power: np.ndarray
    unit_power: np.ndarray
    init: np.ndarray
    composed: dict[int, np.ndarray]
    leak_time: np.ndarray
    leak_state: np.ndarray
    leak_input: np.ndarray
    echo_time: np.ndarray
    echo_target: np.ndarray
    echo_pred: np.ndarray
    echo_mse: float
    delay_mse: np.ndarray
    grad_raw: np.ndarray
    grad_clipped: np.ndarray


def eigenvalue_power(magnitude: float, steps: int) -> np.ndarray:
    """|lambda|^t for the linear recurrence h^(t) = lambda^t h^(0) (§10.7)."""
    n = max(2, int(steps))
    return np.abs(float(magnitude)) ** np.arange(n)


def tanh_compositions(gain: float, *, steps: tuple[int, ...] = (1, 4, 12)) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """1D slice of repeated tanh, in the spirit of figure 10.15."""
    init = np.linspace(-2.5, 2.5, 81)
    state = init.copy()
    wanted = set(steps)
    snapshots: dict[int, np.ndarray] = {}
    last = max(wanted)
    for step in range(1, last + 1):
        state = np.tanh(float(gain) * state)
        if step in wanted:
            snapshots[step] = state.copy()
    return init, snapshots


def leaky_unit(alpha: float, *, steps: int = 30, pulse_time: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """h^(t) = alpha h^(t-1) + (1-alpha) x^(t), with alpha near 1 integrating slowly (§10.9.2)."""
    n = max(4, int(steps))
    inputs = np.zeros(n)
    inputs[int(np.clip(pulse_time, 0, n - 1))] = 1.0
    state = np.zeros(n)
    leak = float(np.clip(alpha, 0.0, 0.999))
    for t in range(1, n):
        state[t] = leak * state[t - 1] + (1.0 - leak) * inputs[t]
    return np.arange(n), state, inputs


def clip_gradient(gradient: np.ndarray, threshold: float) -> np.ndarray:
    """Rescale g when its L2 norm exceeds the cap (§10.11.1)."""
    grad = np.asarray(gradient, dtype=float).copy()
    thr = max(float(threshold), 1e-8)
    norm = float(np.linalg.norm(grad))
    if norm > thr:
        grad *= thr / norm
    return grad


_CLIP_EXAMPLE = np.array([2.4, -1.8, 0.7, 3.3, -2.1, 1.1])


@dataclass(frozen=True)
class EchoStateDemo:
    time: np.ndarray
    target: np.ndarray
    prediction: np.ndarray
    mse: float
    delay_mse: np.ndarray


def _fit_delayed_readout(
    hidden: np.ndarray,
    signal: np.ndarray,
    delay: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    lag = max(1, int(delay))
    target = np.zeros(signal.shape[0])
    target[lag:] = signal[:-lag]
    start = max(ECHO_WASHOUT, lag)
    features = hidden[start:]
    y = target[start:]
    n_units = features.shape[1]
    gram = features.T @ features + 1e-2 * np.eye(n_units)
    coef = np.linalg.solve(gram, features.T @ y)
    pred = features @ coef
    mse = float(np.mean((pred - y) ** 2))
    return np.arange(start, signal.shape[0]), y, pred, mse


def echo_state_demo(
    spectral_radius: float,
    *,
    highlight_delay: int = 5,
    n_units: int = ECHO_UNITS,
    steps: int = ECHO_STEPS,
    seed: int = 0,
) -> EchoStateDemo:
    """Train only linear readouts on one fixed random reservoir (§10.8).

    The input is white noise. Each readout recalls that input at a fixed delay.
    A larger spectral radius keeps the delay visible; a tiny radius forgets it.
    """
    radius = max(float(spectral_radius), 1e-3)
    rng = np.random.default_rng(seed)
    recurrent = rng.normal(scale=1.0, size=(n_units, n_units))
    scale = np.max(np.abs(np.linalg.eigvals(recurrent)))
    recurrent *= radius / max(float(scale), 1e-8)
    input_weights = rng.normal(scale=0.5, size=n_units)
    signal = rng.uniform(-1.0, 1.0, size=steps)
    state = np.zeros(n_units)
    hidden = np.zeros((steps, n_units))
    for t in range(steps):
        state = np.tanh(recurrent @ state + input_weights * signal[t])
        hidden[t] = state
    highlight = int(highlight_delay)
    time, target, pred, mse = _fit_delayed_readout(hidden, signal, highlight)
    delay_mse = np.zeros(len(ECHO_DELAYS))
    for i, delay in enumerate(ECHO_DELAYS):
        if delay == highlight:
            delay_mse[i] = mse
        else:
            delay_mse[i] = _fit_delayed_readout(hidden, signal, delay)[3]
    return EchoStateDemo(time=time, target=target, prediction=pred, mse=mse, delay_mse=delay_mse)


def long_term_demo(
    magnitude: float,
    alpha: float,
    spectral_radius: float,
    clip_threshold: float,
    *,
    steps: int = 30,
) -> LongTermDemo:
    """Vanishing powers, leaky memory, echo-state recall, and gradient clipping."""
    n = max(8, int(steps))
    power = eigenvalue_power(magnitude, n)
    init, composed = tanh_compositions(magnitude)
    leak_time, leak_state, leak_input = leaky_unit(alpha, steps=n)
    echo = echo_state_demo(spectral_radius)
    raw = _CLIP_EXAMPLE.copy()
    return LongTermDemo(
        time=np.arange(n),
        power=power,
        unit_power=np.ones(n),
        init=init,
        composed=composed,
        leak_time=leak_time,
        leak_state=leak_state,
        leak_input=leak_input,
        echo_time=echo.time,
        echo_target=echo.target,
        echo_pred=echo.prediction,
        echo_mse=echo.mse,
        delay_mse=echo.delay_mse,
        grad_raw=raw,
        grad_clipped=clip_gradient(raw, clip_threshold),
    )


@dataclass(frozen=True)
class GatedDemo:
    time: np.ndarray
    target: np.ndarray
    rnn: np.ndarray
    lstm_cell: np.ndarray
    gru: np.ndarray
    rnn_grad_log10: float
    lstm_grad_log10: float
    attention: np.ndarray
    slot_values: np.ndarray
    read_value: float


def memory_attention(
    query_index: int, *, n_slots: int = 4, temperature: float = 0.35
) -> tuple[np.ndarray, np.ndarray, float]:
    """Soft content addressing of a fixed memory (§10.12)."""
    n = max(2, int(n_slots))
    idx = int(np.clip(query_index, 0, n - 1))
    keys = np.eye(n) + 0.2
    values = np.arange(1, n + 1, dtype=float)
    weights = softmax(keys @ keys[idx] / max(float(temperature), 1e-3))
    return weights, values, float(weights @ values)


def gated_demo(forget_bias: float, cue_time: int, query_index: int, *, steps: int = 30) -> GatedDemo:
    """Vanilla RNN vs a hand-gated LSTM cell and GRU on a single cue (§10.10).

    The forget gate is held at sigmoid(bias) so the slider is the self-loop that
    section 10.10 uses to keep the cell state alive. The input gate opens on the cue.
    """
    n = max(8, int(steps))
    cue = int(np.clip(cue_time, 1, n - 2))
    inputs = np.zeros(n)
    inputs[cue] = 1.0
    target = np.zeros(n)
    target[cue:] = 1.0
    rnn = np.zeros(n)
    cell = np.zeros(n)
    gru = np.zeros(n)
    forget = float(sigmoid(forget_bias))
    for t in range(1, n):
        rnn[t] = np.tanh(VANILLA_RECURRENCE * rnn[t - 1] + inputs[t])
        write = float(sigmoid(-1.5 + 6.0 * abs(inputs[t])))
        cell[t] = forget * cell[t - 1] + write * inputs[t]
        update = float(sigmoid(-float(forget_bias) + 6.0 * abs(inputs[t])))
        gru[t] = (1.0 - update) * gru[t - 1] + update * np.tanh(inputs[t])
    rnn_grad = 1.0
    for t in range(n - 1, cue, -1):
        rnn_grad *= VANILLA_RECURRENCE * (1.0 - rnn[t] ** 2)
    gap = n - 1 - cue
    lstm_grad = forget**gap
    weights, values, read_value = memory_attention(query_index)
    return GatedDemo(
        time=np.arange(n),
        target=target,
        rnn=rnn,
        lstm_cell=cell,
        gru=gru,
        rnn_grad_log10=log10_abs(rnn_grad),
        lstm_grad_log10=log10_abs(lstm_grad),
        attention=weights,
        slot_values=values,
        read_value=read_value,
    )
