"""Ignition probability and hard-threshold simulations (S1-IMP-04).

Soft ignition draws ``B ~ Bernoulli(σ((S−θ)/τ_σ))``; hard ignition is the
``τ_σ → 0`` limit ``B = 1[S > θ]``. This module provides vectorised probability
and sampling helpers, session-level summaries, and a threshold sweep for both
regimes on the qualified reference model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from core.rng import RNGLike, resolve_rng, spawn_rng
from models.reduced import TAU_SIGMA, ReducedParams, simulate_reduced

__all__ = [
    "ignition_probability",
    "hard_ignition",
    "sample_ignition",
    "IgnitionSummary",
    "summarise_ignitions",
    "simulate_session",
    "threshold_sweep",
    "hard_limit_gap",
]


def ignition_probability(
    S: Sequence[float] | np.ndarray, theta: Sequence[float] | np.ndarray | float, tau: float
) -> np.ndarray:
    """Vectorised ``σ((S−θ)/τ)``, overflow-safe.

    Raises:
        ValueError: If ``tau`` is not positive.
    """
    if tau <= 0:
        raise ValueError("tau must be > 0")
    z = np.clip((np.asarray(S, dtype=float) - np.asarray(theta, dtype=float)) / tau, -500.0, 500.0)
    return np.asarray(1.0 / (1.0 + np.exp(-z)))


def hard_ignition(
    S: Sequence[float] | np.ndarray, theta: Sequence[float] | np.ndarray | float
) -> np.ndarray:
    """Hard threshold ``1[S > θ]`` as int array."""
    return (np.asarray(S, dtype=float) > np.asarray(theta, dtype=float)).astype(int)


def sample_ignition(p: Sequence[float] | np.ndarray, rng: RNGLike = None) -> np.ndarray:
    """Draw Bernoulli ignitions for probabilities ``p`` (all in [0, 1])."""
    arr = np.asarray(p, dtype=float)
    if np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")
    return (resolve_rng(rng).random(arr.shape) < arr).astype(int)


@dataclass(frozen=True)
class IgnitionSummary:
    """Session-level ignition statistics."""

    rate: float
    count: int
    first_ignition: int | None
    mean_inter_ignition_interval: float | None


def summarise_ignitions(B: Sequence[int] | np.ndarray) -> IgnitionSummary:
    """Summarise a binary ignition train."""
    b = np.asarray(B, dtype=int)
    idx = np.flatnonzero(b)
    iis = np.diff(idx)
    return IgnitionSummary(
        rate=float(b.mean()) if b.size else 0.0,
        count=int(b.sum()),
        first_ignition=int(idx[0]) if idx.size else None,
        mean_inter_ignition_interval=float(iis.mean()) if iis.size else None,
    )


def simulate_session(
    params: ReducedParams,
    stimulus: Sequence[float] | np.ndarray,
    *,
    mode: str = "soft",
    seed: int | None = 0,
) -> IgnitionSummary:
    """Simulate one session of the reference model and summarise its ignitions.

    Args:
        params: Reference-model parameters.
        stimulus: Stimulus sequence.
        mode: ``"soft"`` (Bernoulli) or ``"hard"`` (``S > θ``).
        seed: Seed for soft mode.
    """
    if mode not in ("soft", "hard"):
        raise ValueError(f"mode must be 'soft' or 'hard', got {mode!r}")
    traj = simulate_reduced(params, stimulus, rng=seed, hard=mode == "hard")
    return summarise_ignitions(traj.B)


def threshold_sweep(
    params: ReducedParams,
    stimulus: Sequence[float] | np.ndarray,
    thetas: Sequence[float],
    *,
    mode: str = "soft",
    n_sessions: int = 20,
    seed: int = 0,
) -> np.ndarray:
    """Mean ignition rate as a function of baseline threshold ``θ₀``.

    Each ``θ₀`` is evaluated on the same ``n_sessions`` spawned streams (common
    random numbers), so differences between thresholds are not sampling noise.
    Hard mode is deterministic and ignores ``n_sessions``.

    Returns:
        Array of mean rates, one per entry of ``thetas``.
    """
    streams: list[np.random.Generator | None] = (
        list(spawn_rng(seed, n_sessions)) if mode == "soft" else [None]
    )
    rates = []
    for theta in thetas:
        p = replace(params, theta_base=float(theta))
        per = []
        for g in streams:
            traj = simulate_reduced(p, stimulus, rng=g, hard=mode == "hard")
            per.append(traj.B.mean())
        rates.append(float(np.mean(per)))
        if mode == "soft":
            # Re-create identical streams for the next threshold (common random numbers).
            streams = list(spawn_rng(seed, n_sessions))
    return np.asarray(rates)


def hard_limit_gap(S: Sequence[float], theta: Sequence[float], taus: Sequence[float]) -> np.ndarray:
    """Mean ``|σ((S−θ)/τ) − 1[S>θ]|`` for each ``τ``; decreases to 0 as ``τ → 0``."""
    hard = hard_ignition(S, theta)
    return np.asarray([np.abs(ignition_probability(S, theta, t) - hard).mean() for t in taus])


#: Re-exported for callers building their own sigmoid with the model's temperature.
DEFAULT_TAU = TAU_SIGMA
