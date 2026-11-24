from __future__ import annotations

import numpy as np

from .deprecation import warn_renamed
from .rng import RNGLike, resolve_rng


def compute_ignition_probability(S: float, theta: float, tau: float = 1.0) -> float:
    """P_ignite = sigmoid((S-θ)/τ)."""
    if tau <= 0:
        raise ValueError("tau must be > 0")
    x = (S - theta) / tau
    # Clamp to prevent overflow in exp
    x = np.clip(x, -500, 500)
    return float(1.0 / (1.0 + np.exp(-x)))


def hazard_probability(p_ignite: float, dt: float, r_max: float) -> float:
    """Per-step ignition probability under the dt-invariant hazard (Math Spec Eq. 35).

    P(B=1) = 1 − exp(−r_max·Bₜ·dt), with Bₜ = ``p_ignite`` the logistic ignition
    propensity (a rate modulator, not a per-step probability) and r_max the
    maximal ignition rate in s⁻¹. Two steps of dt have the same survival
    probability as one step of 2·dt (apgi-lean: Apgi.hazard_two_steps).
    """
    if not (0.0 <= p_ignite <= 1.0):
        raise ValueError("p_ignite must be in [0,1]")
    if dt < 0 or r_max < 0:
        raise ValueError("dt and r_max must be >= 0")
    return float(-np.expm1(-r_max * p_ignite * dt))


def sample_ignition_state(
    p_ignite: float,
    rng: RNGLike = None,
    *,
    dt: float | None = None,
    r_max: float | None = None,
) -> int:
    """Sample the ignition indicator B(t) ∈ {0, 1}.

    With ``dt`` and ``r_max`` (the release v1.1 canonical form), B ~
    Bernoulli(1 − exp(−r_max·Bₜ·dt)), which is dt-invariant. Without them the
    legacy per-step Bernoulli(Bₜ) is used; it is not dt-invariant (Math Spec
    p22) and emits an :class:`APGIDeprecationWarning`.

    Args:
        p_ignite: Ignition propensity Bₜ ∈ [0, 1].
        rng: Generator, int seed, or ``None`` to use the process-global APGI
            generator (see :mod:`core.rng`). Never constructs an unseeded
            generator — that would make ignition sampling irreproducible.
        dt: Integration step in seconds.
        r_max: Maximal ignition rate in s⁻¹.

    Returns:
        Bernoulli sample B(t) ∈ {0, 1}.
    """
    if not (0.0 <= p_ignite <= 1.0):
        raise ValueError("p_ignite must be in [0,1]")
    if (dt is None) != (r_max is None):
        raise TypeError("pass both dt and r_max, or neither")
    if dt is None or r_max is None:
        warn_renamed(
            "per-step Bernoulli(B_t) sampling",
            "the hazard: sample_ignition_state(p, dt=..., r_max=...)",
        )
        p = p_ignite
    else:
        p = hazard_probability(p_ignite, dt, r_max)
    generator = resolve_rng(rng)
    return int(generator.random() < p)


def detect_ignition_event(S: float, theta: float) -> bool:
    return bool(theta < S)


def compute_margin(S: float, theta: float) -> float:
    return float(S - theta)
