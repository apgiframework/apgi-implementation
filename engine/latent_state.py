"""Sequential latent-state estimation (S1-IMP-01).

A bootstrap particle filter for the signal ``S_t`` of the reference model when
the accumulation carries process noise,

    S_t = (1−λ)·ρ^{B_{t−1}}·S_{t−1} + λ·drive_t + σ_s·ξ_t,   ξ_t ~ N(0, 1),

and is observed through ignition ``B_t ~ Bernoulli(σ((S_t − θ_t)/τ_σ))`` and,
optionally, an amplitude ``y_t ~ N(S_t, σ_y²)``. The prediction ``x̂_t`` and the
threshold ``θ_t`` are deterministic given the observed ignition train, so they
are shared by all particles; only ``S`` is latent.

With ``σ_s = 0`` every particle coincides and the returned log evidence equals
the teacher-forced log likelihood of the deterministic model — the check the
reference fixtures use.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

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
)

__all__ = ["FilterResult", "filter_latent_signal"]


@dataclass
class FilterResult:
    """Filtering output (all arrays length ``T``)."""

    S_mean: np.ndarray
    S_var: np.ndarray
    ess: np.ndarray
    log_evidence: float
    resampled: np.ndarray

    def as_dict(self) -> dict[str, object]:
        return {
            "S_mean": self.S_mean,
            "S_var": self.S_var,
            "ess": self.ess,
            "log_evidence": self.log_evidence,
            "resampled": self.resampled,
        }


def _logsumexp(a: np.ndarray) -> float:
    m = float(np.max(a))
    if not math.isfinite(m):
        return m
    return m + math.log(float(np.exp(a - m).sum()))


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = weights.shape[0]
    positions = (rng.random() + np.arange(n)) / n
    cum = np.cumsum(weights)
    cum[-1] = 1.0
    return np.searchsorted(cum, positions).astype(int)


def filter_latent_signal(
    params: ReducedParams,
    stimulus: Sequence[float] | np.ndarray,
    B: Sequence[int] | np.ndarray,
    y: Sequence[float] | np.ndarray | None = None,
    *,
    sigma_s: float = 0.05,
    n_particles: int = 500,
    resample_threshold: float = 0.5,
    rng: RNGLike = None,
) -> FilterResult:
    """Run the bootstrap filter.

    Args:
        params: Reference-model parameters.
        stimulus: Stimulus per trial.
        B: Observed ignition train.
        y: Observed amplitudes (optional).
        sigma_s: Process-noise standard deviation of the accumulation (≥ 0).
        n_particles: Number of particles (≥ 1; with ``sigma_s == 0`` one suffices).
        resample_threshold: Resample when ``ESS < threshold · n_particles``.
        rng: Generator or seed.

    Returns:
        :class:`FilterResult` with the filtering mean/variance of ``S_t``, the
        effective sample size before resampling, the log marginal likelihood of
        the observations, and a flag per step marking where resampling occurred.

    Raises:
        ValueError: On inconsistent lengths or invalid arguments.
    """
    x = np.asarray(stimulus, dtype=float)
    b = np.asarray(B, dtype=int)
    T = x.shape[0]
    if b.shape != (T,):
        raise ValueError("B must have the same length as stimulus")
    yy = None if y is None else np.asarray(y, dtype=float)
    if yy is not None and yy.shape != (T,):
        raise ValueError("y must have the same length as stimulus")
    if sigma_s < 0:
        raise ValueError("sigma_s must be >= 0")
    if n_particles < 1:
        raise ValueError("n_particles must be >= 1")
    gen = resolve_rng(rng)

    n = n_particles
    S = np.zeros(n)
    log_w = np.full(n, -math.log(n))
    xhat = 0.0
    theta = params.theta_base
    B_prev = 0
    k = learning_gain(params.alpha, params.pi)
    exp_k = math.exp(-KAPPA)
    S_mean = np.empty(T)
    S_var = np.empty(T)
    ess_hist = np.empty(T)
    resampled = np.zeros(T, dtype=bool)
    log_evidence = 0.0

    for t in range(T):
        eps = x[t] - xhat
        drive = math.tanh(GAMMA_PHI * params.pi * abs(eps)) + params.beta
        carry = RHO if B_prev else 1.0
        S = (1.0 - LAM) * carry * S + LAM * drive
        if sigma_s > 0:
            S = S + sigma_s * gen.standard_normal(n)
        z = np.clip((S - theta) / TAU_SIGMA, -500.0, 500.0)
        # log Bernoulli: B·log σ(z) + (1−B)·log σ(−z), written stably via logaddexp.
        log_obs = -np.logaddexp(0.0, -z) if b[t] else -np.logaddexp(0.0, z)
        if yy is not None:
            log_obs = log_obs - 0.5 * ((yy[t] - S) / params.sigma_y) ** 2
            log_obs = log_obs - math.log(params.sigma_y * math.sqrt(2.0 * math.pi))
        log_w = log_w + log_obs
        log_norm = _logsumexp(log_w)
        log_evidence += log_norm
        log_w = log_w - log_norm
        w = np.exp(log_w)
        S_mean[t] = float(np.dot(w, S))
        S_var[t] = float(np.dot(w, (S - S_mean[t]) ** 2))
        ess = 1.0 / float(np.dot(w, w))
        ess_hist[t] = ess
        if ess < resample_threshold * n and sigma_s > 0:
            S = S[_systematic_resample(w, gen)]
            log_w = np.full(n, -math.log(n))
            resampled[t] = True
        theta = params.theta_base + (theta - params.theta_base) * exp_k + DELTA * int(b[t])
        xhat = xhat + k * eps
        B_prev = int(b[t])

    return FilterResult(S_mean, S_var, ess_hist, log_evidence, resampled)
