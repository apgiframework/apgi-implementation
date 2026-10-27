"""Shared ODE-based synthetic generator (S1-IMP-17).

One generator, with known parameters, drives both the integration test and the
K7c recovery simulation, so the two cannot disagree about what "truth" means.

The generating process is the continuous-time form of ``apgi_reduced`` over a
trial of unit duration, integrated numerically with fixed-step RK4:

    dx̂/dt = r_x·(x − x̂),        r_x = −ln(1 − k)
    dS/dt = r_S·(D − S),         r_S = −ln(1 − λ),   D = tanh(γ_φ Π |ε₀|) + β
    dθ/dt = −κ·(θ − θ₀)

with the drive ``D`` held at its trial-start value (zero-order hold). The rate
constants are chosen so the exact one-trial solution reproduces the scoring
model's recurrence; the generator therefore differs from the scorer only by
integration error, which the integration test bounds. Ignition and the
post-ignition reset/boost are applied at trial boundaries.

The scoring model must recover the generating composites — the
``precision_weighted_learning_composite`` ``αΠ`` and the gain ``k`` — within
tolerance; :func:`check_scoring_regression` raises if it does not, which is the
release-blocking regression named in the register.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from core.rng import RNGLike, resolve_rng
from models.reduced import (
    DELTA,
    GAMMA_PHI,
    KAPPA,
    LAM,
    RHO,
    TAU_SIGMA,
    ReducedParams,
    learning_gain,
    simulate_reduced,
)

__all__ = [
    "Design",
    "SyntheticDataset",
    "reversal_stimulus",
    "generate_dataset",
    "generating_composites",
    "check_scoring_regression",
    "RegressionError",
]


@dataclass(frozen=True)
class Design:
    """Experimental design for the reversal-learning task.

    Attributes:
        n_trials: Number of trials ``T``.
        block: Trials between reversals of the stimulus mean.
        amplitude: Magnitude of the stimulus mean (alternates ±).
        sigma_x: Stimulus noise standard deviation.
    """

    n_trials: int = 240
    block: int = 30
    amplitude: float = 0.8
    sigma_x: float = 0.3

    def __post_init__(self) -> None:
        if self.n_trials < 2 or self.block < 1 or self.sigma_x < 0 or self.amplitude <= 0:
            raise ValueError(f"invalid design: {self}")


def reversal_stimulus(design: Design, rng: RNGLike = None) -> np.ndarray:
    """Stimulus ``x_t = μ_t + σ_x·ξ_t`` with ``μ_t`` reversing every ``block`` trials."""
    t = np.arange(design.n_trials)
    mu = design.amplitude * np.where((t // design.block) % 2 == 0, 1.0, -1.0)
    return np.asarray(mu + design.sigma_x * resolve_rng(rng).standard_normal(design.n_trials))


@dataclass
class SyntheticDataset:
    """Generated data plus its ground truth."""

    stimulus: np.ndarray
    B: np.ndarray
    report: np.ndarray
    amplitude: np.ndarray
    truth: ReducedParams
    S_true: np.ndarray = field(repr=False)
    xhat_true: np.ndarray = field(repr=False)

    @property
    def n_trials(self) -> int:
        return int(self.stimulus.shape[0])


def generating_composites(params: ReducedParams) -> dict[str, float]:
    """The composites the generator is built from: ``αΠ`` and the gain ``k``."""
    return {
        "precision_weighted_learning_composite": params.alpha * params.pi,
        "learning_gain": learning_gain(params.alpha, params.pi),
    }


def _rk4(f, y: np.ndarray, h: float) -> np.ndarray:  # type: ignore[no-untyped-def]
    k1 = f(y)
    k2 = f(y + 0.5 * h * k1)
    k3 = f(y + 0.5 * h * k2)
    k4 = f(y + h * k3)
    return np.asarray(y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))


def generate_dataset(
    params: ReducedParams,
    design: Design | None = None,
    rng: RNGLike = None,
    *,
    stimulus: Sequence[float] | np.ndarray | None = None,
    n_substeps: int = 40,
) -> SyntheticDataset:
    """Generate one subject's data by ODE integration.

    Args:
        params: Generating (true) parameters.
        design: Task design; ignored if ``stimulus`` is given.
        rng: Generator or seed. A single stream drives the stimulus, the
            ignition draws and the observation noise, in that order.
        stimulus: Optional fixed stimulus.
        n_substeps: RK4 steps per unit-duration trial.

    Returns:
        The :class:`SyntheticDataset`.
    """
    gen = resolve_rng(rng)
    design = design or Design()
    x = reversal_stimulus(design, gen) if stimulus is None else np.asarray(stimulus, dtype=float)
    T = x.shape[0]
    k = learning_gain(params.alpha, params.pi)
    r_x = -math.log1p(-k)
    r_S = -math.log1p(-LAM)
    h = 1.0 / n_substeps

    u = gen.random(T)
    xhat, S, theta, B_prev = 0.0, 0.0, params.theta_base, 0

    Bs = np.empty(T, dtype=int)
    xhats = np.empty(T)
    Ss = np.empty(T)
    for t in range(T):
        xt = float(x[t])
        drive = math.tanh(GAMMA_PHI * params.pi * abs(xt - xhat)) + params.beta
        S0 = RHO * S if B_prev else S

        def f(y: np.ndarray, xt: float = xt, drive: float = drive) -> np.ndarray:
            return np.array(
                [r_x * (xt - y[0]), r_S * (drive - y[1]), -KAPPA * (y[2] - params.theta_base)]
            )

        y = np.array([xhat, S0, theta])
        for _ in range(n_substeps):
            y = _rk4(f, y, h)
        xhats[t] = xhat
        S = float(y[1])
        Ss[t] = S
        z = max(-500.0, min(500.0, (S - theta) / TAU_SIGMA))
        p = 1.0 / (1.0 + math.exp(-z))
        B = int(u[t] < p)
        Bs[t] = B
        xhat, theta, B_prev = float(y[0]), float(y[2]) + DELTA * B, B

    report = xhats + params.sigma_r * gen.standard_normal(T)
    amplitude = Ss + params.sigma_y * gen.standard_normal(T)
    return SyntheticDataset(x, Bs, report, amplitude, params, Ss, xhats)


class RegressionError(AssertionError):
    """Raised when scoring fails to reproduce generating composites (blocks release)."""


def check_scoring_regression(
    params: ReducedParams,
    design: Design | None = None,
    *,
    seed: int = 0,
    atol: float = 1e-6,
    n_substeps: int = 40,
) -> dict[str, float]:
    """Verify the scorer reproduces the generator's trajectory and composites.

    Generates a dataset by ODE integration, replays the *same* ignition train
    through the scoring model (teacher-forced) and compares ``x̂`` and ``S``.
    Because the composites enter the trajectory only through ``k``, agreement of
    ``x̂`` to ``atol`` pins ``αΠ`` to the same tolerance.

    Returns:
        ``{"max_abs_xhat", "max_abs_S", "composite_error"}``.

    Raises:
        RegressionError: If any discrepancy exceeds ``atol``.
    """
    ds = generate_dataset(params, design, seed, n_substeps=n_substeps)
    scored = simulate_reduced(params, ds.stimulus, B_forced=ds.B)
    err_x = float(np.max(np.abs(scored.xhat - ds.xhat_true)))
    err_S = float(np.max(np.abs(scored.S - ds.S_true)))
    # Recover the composite from the generator's own x̂ path (not the scorer's):
    # x̂_{t+1} − x̂_t = k·ε_t  ⇒  k, and αΠ = k/(1−k).
    eps = ds.stimulus[:-1] - ds.xhat_true[:-1]
    usable = np.abs(eps) > 1e-3
    k_hat = float(np.median((ds.xhat_true[1:] - ds.xhat_true[:-1])[usable] / eps[usable]))
    composite_hat = k_hat / (1.0 - k_hat)
    composite = generating_composites(params)["precision_weighted_learning_composite"]
    out = {
        "max_abs_xhat": err_x,
        "max_abs_S": err_S,
        "composite_error": abs(composite_hat - composite),
    }
    if max(out.values()) > atol:
        raise RegressionError(
            f"scoring does not reproduce generating composites: {out} (atol={atol})"
        )
    return out
