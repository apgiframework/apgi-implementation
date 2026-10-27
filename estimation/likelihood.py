"""Likelihood and prior for the reference model (numpy reference implementation).

Given the observed ignition train, ``x̂_t``, ``S_t`` and ``θ_t`` are
deterministic functions of the parameters, so the likelihood is exact
(teacher forcing):

    log L = Σ_t [ log N(r_t; x̂_t, σ_r²)                      (behavior)
                + B_t log p_t + (1−B_t) log(1−p_t)          (ignition)
                + log N(y_t; S_t, σ_y²) ]                    (amplitude)

summed over the channels in use. The PyMC model in
:mod:`estimation.pymc_models` is a transcription of this function and is
checked against it numerically; this module is the reference.

Parameters are estimated on an unconstrained scale ``u`` with
``p_j = lo_j + (hi_j − lo_j)·σ(u_j)`` and a Gaussian prior on ``u``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from models.parameters import REDUCED_PARAMETER_NAMES, REDUCED_PARAMETERS
from models.reduced import (
    CHANNELS,
    DELTA,
    GAMMA_PHI,
    KAPPA,
    LAM,
    RHO,
    TAU_SIGMA,
    ReducedParams,
    learning_gain,
)

__all__ = [
    "Observed",
    "check_channels",
    "log_likelihood",
    "to_constrained",
    "to_unconstrained",
    "PRIOR_MEAN_U",
    "PRIOR_SD_U",
    "log_prior_u",
    "log_posterior_u",
    "BOUNDS",
]

#: Hard support of the estimable parameters, canonical order.
BOUNDS: np.ndarray = np.array([[p.lower, p.upper] for p in REDUCED_PARAMETERS], dtype=float)


def _plausible_mid(spec) -> float:  # type: ignore[no-untyped-def]
    if spec.log_scale:
        return math.sqrt(spec.plausible_lower * spec.plausible_upper)
    return float(0.5 * (spec.plausible_lower + spec.plausible_upper))


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


#: Prior on the unconstrained scale: centred on the plausible-range midpoint,
#: SD 1.5 so the plausible range sits comfortably inside ±1 SD of the prior
#: while the hard support stays reachable.
PRIOR_MEAN_U: np.ndarray = np.array(
    [_logit((_plausible_mid(s) - s.lower) / (s.upper - s.lower)) for s in REDUCED_PARAMETERS]
)
PRIOR_SD_U: np.ndarray = np.full(len(REDUCED_PARAMETERS), 1.5)


@dataclass
class Observed:
    """One subject's observations.

    ``B`` is always required: even when the ignition channel is not in the
    likelihood, the reset after ignition makes ``S`` depend on it.
    """

    stimulus: np.ndarray
    B: np.ndarray
    report: np.ndarray | None = None
    amplitude: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.stimulus = np.asarray(self.stimulus, dtype=float)
        self.B = np.asarray(self.B, dtype=int)
        T = self.stimulus.shape[0]
        for name in ("B", "report", "amplitude"):
            arr = getattr(self, name)
            if arr is not None:
                arr = np.asarray(arr, dtype=float if name != "B" else int)
                setattr(self, name, arr)
                if arr.shape != (T,):
                    raise ValueError(f"{name} must have shape ({T},), got {arr.shape}")


def check_channels(channels: Sequence[str]) -> tuple[str, ...]:
    """Validate and normalise a channel selection (canonical order, no duplicates)."""
    unknown = set(channels) - set(CHANNELS)
    if unknown:
        raise ValueError(f"unknown channels {sorted(unknown)}; expected subset of {CHANNELS}")
    if not channels:
        raise ValueError("at least one channel is required")
    return tuple(c for c in CHANNELS if c in channels)


_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def log_likelihood(
    params: ReducedParams, data: Observed, channels: Sequence[str] = CHANNELS
) -> float:
    """Exact log likelihood of ``data`` under ``params`` for the chosen channels.

    Raises:
        ValueError: If a selected channel's data are missing.
    """
    ch = check_channels(channels)
    use_r = "behavior" in ch
    use_b = "ignition" in ch
    use_y = "amplitude" in ch
    if use_r and data.report is None:
        raise ValueError("behavior channel selected but data.report is None")
    if use_y and data.amplitude is None:
        raise ValueError("amplitude channel selected but data.amplitude is None")

    x = data.stimulus.tolist()
    Bobs = data.B.tolist()
    rep = data.report.tolist() if use_r and data.report is not None else None
    amp = data.amplitude.tolist() if use_y and data.amplitude is not None else None

    k = learning_gain(params.alpha, params.pi)
    pi, beta, th0 = params.pi, params.beta, params.theta_base
    inv_sr, inv_sy = 1.0 / params.sigma_r, 1.0 / params.sigma_y
    norm_r = -(math.log(params.sigma_r) + _LOG_SQRT_2PI)
    norm_y = -(math.log(params.sigma_y) + _LOG_SQRT_2PI)
    exp_k = math.exp(-KAPPA)
    one_m_lam = 1.0 - LAM

    xhat, S, theta, Bprev = 0.0, 0.0, th0, 0
    ll = 0.0
    for t in range(len(x)):
        eps = x[t] - xhat
        if rep is not None:
            z = (rep[t] - xhat) * inv_sr
            ll += norm_r - 0.5 * z * z
        S = one_m_lam * (RHO if Bprev else 1.0) * S + LAM * (
            math.tanh(GAMMA_PHI * pi * abs(eps)) + beta
        )
        if amp is not None:
            z = (amp[t] - S) * inv_sy
            ll += norm_y - 0.5 * z * z
        b = Bobs[t]
        if use_b:
            a = (S - theta) / TAU_SIGMA
            a = -500.0 if a < -500.0 else (500.0 if a > 500.0 else a)
            # log σ(a) for B=1, log σ(−a) for B=0, via a stable softplus.
            s = -a if b else a
            ll -= math.log1p(math.exp(s)) if s < 30.0 else s
        theta = th0 + (theta - th0) * exp_k + DELTA * b
        xhat += k * eps
        Bprev = b
    return ll


def to_constrained(u: Sequence[float] | np.ndarray) -> np.ndarray:
    """Map unconstrained ``u`` to parameter values inside the hard support."""
    u = np.asarray(u, dtype=float)
    sig = 1.0 / (1.0 + np.exp(-np.clip(u, -500.0, 500.0)))
    return np.asarray(BOUNDS[:, 0] + (BOUNDS[:, 1] - BOUNDS[:, 0]) * sig)


def to_unconstrained(p: Sequence[float] | np.ndarray) -> np.ndarray:
    """Inverse of :func:`to_constrained`; values are clipped just inside the support."""
    p = np.asarray(p, dtype=float)
    frac = (p - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0])
    frac = np.clip(frac, 1e-9, 1.0 - 1e-9)
    return np.asarray(np.log(frac / (1.0 - frac)))


def log_prior_u(u: Sequence[float] | np.ndarray) -> float:
    """Gaussian log prior on the unconstrained scale (up to the normalising constant)."""
    z = (np.asarray(u, dtype=float) - PRIOR_MEAN_U) / PRIOR_SD_U
    return float(-0.5 * np.dot(z, z))


def log_posterior_u(
    u: Sequence[float] | np.ndarray,
    base: ReducedParams,
    data: Observed,
    channels: Sequence[str] = CHANNELS,
) -> float:
    """Unnormalised log posterior at ``u`` (noise levels taken from ``base``)."""
    params = base.with_estimable(to_constrained(u))
    return log_likelihood(params, data, channels) + log_prior_u(u)


assert len(PRIOR_MEAN_U) == len(REDUCED_PARAMETER_NAMES)
