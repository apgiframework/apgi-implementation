"""Posterior estimation for the reference model without PyMC.

Three estimators share one interface and return a :class:`PosteriorSummary`:

* ``map``     — maximum a posteriori (multi-start L-BFGS-B) on the unconstrained
  scale;
* ``laplace`` — Gaussian approximation at the MAP from a finite-difference
  Hessian (fast; used for the CI smoke test and screening);
* ``mcmc``    — differential-evolution MCMC with split-R̂ and bulk ESS (the
  convergence evidence K7c needs, separately from identifiability).

:mod:`estimation.pymc_models` provides the NUTS alternative when PyMC is
installed. Convergence of *any* sampler says nothing about identifiability;
that is what :mod:`estimation.structural_identifiability` and
:mod:`estimation.profile_likelihood` are for.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize  # type: ignore[import-untyped]

from core.rng import RNGLike, resolve_rng
from models.parameters import REDUCED_PARAMETER_NAMES
from models.reduced import CHANNELS, ReducedParams

from .likelihood import (
    PRIOR_MEAN_U,
    PRIOR_SD_U,
    Observed,
    check_channels,
    log_posterior_u,
    to_constrained,
)

__all__ = [
    "PosteriorSummary",
    "fit_map",
    "numerical_hessian",
    "fit_laplace",
    "fit_mcmc",
    "split_rhat",
    "effective_sample_size",
    "fit",
]


@dataclass
class PosteriorSummary:
    """Posterior over the estimable parameters (canonical order).

    ``samples`` has shape ``(n, 4)`` in constrained space. ``rhat`` and ``ess``
    are populated by sampling estimators only.
    """

    estimator: str
    map_estimate: np.ndarray
    samples: np.ndarray
    log_posterior: float
    hessian_pd: bool = True
    rhat: np.ndarray | None = None
    ess: np.ndarray | None = None
    accept_rate: float | None = None
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def mean(self) -> np.ndarray:
        return np.asarray(self.samples.mean(axis=0))

    def interval(self, level: float = 0.95) -> np.ndarray:
        """Central credible interval, shape ``(4, 2)``."""
        a = 0.5 * (1.0 - level)
        return np.asarray(np.quantile(self.samples, [a, 1.0 - a], axis=0).T)

    def correlation(self) -> np.ndarray:
        """Posterior correlation matrix (identity if a parameter has no spread)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.corrcoef(self.samples, rowvar=False)
        return np.nan_to_num(c, nan=0.0) + 0.0

    def as_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "estimator": self.estimator,
            "map": dict(zip(REDUCED_PARAMETER_NAMES, map(float, self.map_estimate), strict=True)),
            "mean": dict(zip(REDUCED_PARAMETER_NAMES, map(float, self.mean), strict=True)),
            "hessian_pd": self.hessian_pd,
        }
        if self.rhat is not None:
            out["rhat"] = dict(zip(REDUCED_PARAMETER_NAMES, map(float, self.rhat), strict=True))
        if self.ess is not None:
            out["ess"] = dict(zip(REDUCED_PARAMETER_NAMES, map(float, self.ess), strict=True))
        return out


def _neg_logpost(
    u: np.ndarray, base: ReducedParams, data: Observed, channels: tuple[str, ...]
) -> float:
    val = log_posterior_u(u, base, data, channels)
    return -val if np.isfinite(val) else 1e12


def fit_map(
    base: ReducedParams,
    data: Observed,
    channels: Sequence[str] = CHANNELS,
    *,
    n_starts: int = 4,
    rng: RNGLike = None,
) -> tuple[np.ndarray, float]:
    """Multi-start MAP on the unconstrained scale.

    Returns:
        ``(u_map, log_posterior)``. The first start is the prior mean, the rest
        are prior draws, so the result is deterministic given ``rng``.
    """
    ch = check_channels(channels)
    gen = resolve_rng(rng)
    starts = [PRIOR_MEAN_U.copy()] + [
        PRIOR_MEAN_U + PRIOR_SD_U * gen.standard_normal(PRIOR_MEAN_U.shape)
        for _ in range(max(n_starts - 1, 0))
    ]
    best_u, best_val = PRIOR_MEAN_U.copy(), np.inf
    for u0 in starts:
        res = optimize.minimize(
            _neg_logpost,
            u0,
            args=(base, data, ch),
            method="L-BFGS-B",
            bounds=[(-12.0, 12.0)] * len(u0),
        )
        if res.fun < best_val:
            best_u, best_val = np.asarray(res.x, dtype=float), float(res.fun)
    return best_u, -best_val


def numerical_hessian(
    f: Callable[[np.ndarray], float], x: np.ndarray, step: float = 1e-3
) -> np.ndarray:
    """Central-difference Hessian of scalar ``f`` at ``x`` (symmetrised)."""
    n = x.shape[0]
    H = np.empty((n, n))
    f0 = f(x)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = step
        H[i, i] = (f(x + ei) - 2.0 * f0 + f(x - ei)) / step**2
        for j in range(i + 1, n):
            ej = np.zeros(n)
            ej[j] = step
            H[i, j] = H[j, i] = (
                f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)
            ) / (4.0 * step**2)
    return H


def fit_laplace(
    base: ReducedParams,
    data: Observed,
    channels: Sequence[str] = CHANNELS,
    *,
    n_samples: int = 2000,
    n_starts: int = 4,
    rng: RNGLike = None,
) -> PosteriorSummary:
    """Laplace approximation at the MAP.

    The negative Hessian of the log posterior (unconstrained scale) is
    eigen-clipped to be positive definite before inversion; ``hessian_pd`` is
    ``False`` when clipping was needed, which marks a flat or ridged posterior
    where the Gaussian approximation should not be trusted.
    """
    ch = check_channels(channels)
    gen = resolve_rng(rng)
    u_map, lp = fit_map(base, data, ch, n_starts=n_starts, rng=gen)
    H = numerical_hessian(lambda u: _neg_logpost(u, base, data, ch), u_map)
    w, V = np.linalg.eigh(0.5 * (H + H.T))
    pd = bool(np.all(w > 1e-8))
    w = np.clip(w, 1e-8, None)
    cov = (V / w) @ V.T
    cov = 0.5 * (cov + cov.T)
    draws_u = gen.multivariate_normal(u_map, cov, size=n_samples, method="eigh")
    samples = np.stack([to_constrained(u) for u in draws_u])
    return PosteriorSummary(
        "laplace",
        to_constrained(u_map),
        samples,
        lp,
        hessian_pd=pd,
        extra={"min_eig": float(w.min())},
    )


def split_rhat(chains: np.ndarray) -> np.ndarray:
    """Split-R̂ per parameter for ``chains`` of shape ``(n_chains, n_draws, n_params)``."""
    c, n, _ = chains.shape
    half = n // 2
    if half < 2:
        raise ValueError("need at least 4 draws per chain")
    split = np.concatenate([chains[:, :half], chains[:, half : 2 * half]], axis=0)
    m = split.shape[0]
    chain_mean = split.mean(axis=1)
    chain_var = split.var(axis=1, ddof=1)
    W = chain_var.mean(axis=0)
    B = half * chain_mean.var(axis=0, ddof=1)
    var_hat = (half - 1) / half * W + B / half
    with np.errstate(divide="ignore", invalid="ignore"):
        rhat = np.sqrt(var_hat / W)
    del m
    return np.where(W > 0, rhat, 1.0)


def effective_sample_size(chains: np.ndarray) -> np.ndarray:
    """Bulk ESS per parameter (Geyer initial-positive-sequence truncation)."""
    c, n, p = chains.shape
    ess = np.empty(p)
    for j in range(p):
        x = chains[:, :, j] - chains[:, :, j].mean(axis=1, keepdims=True)
        var = x.var(axis=1, ddof=0)
        if np.all(var == 0):
            ess[j] = c * n
            continue
        f = np.fft.rfft(x, n=2 * n, axis=1)
        acov = np.fft.irfft(f * np.conj(f), axis=1)[:, :n] / n
        rho = acov.mean(axis=0) / acov[:, 0].mean()
        tau = 1.0
        for lag in range(1, n - 1, 2):
            pair = rho[lag] + rho[lag + 1]
            if pair <= 0:
                break
            tau += 2.0 * pair
        ess[j] = c * n / max(tau, 1.0)
    return ess


def fit_mcmc(
    base: ReducedParams,
    data: Observed,
    channels: Sequence[str] = CHANNELS,
    *,
    n_chains: int = 12,
    n_warmup: int = 600,
    n_draws: int = 1000,
    rng: RNGLike = None,
) -> PosteriorSummary:
    """Differential-evolution MCMC (ter Braak 2006) on the unconstrained scale.

    Each chain proposes ``u_i + γ(u_a − u_b) + ε`` from two other chains, which
    adapts the proposal to the posterior's scale and orientation without tuning.
    That matters here: the ``α``–``Π``–``β`` ridge defeats a fixed-covariance
    random walk (effective sample sizes of tens in thousands of draws), while this
    scheme reaches hundreds. ``γ = 2.38/√(2d)``, switching to ``γ = 1`` on 10% of
    proposals to hop between modes. Chains start over-dispersed around the MAP.

    Returns split-R̂ and bulk ESS per parameter on the constrained scale;
    these diagnose *sampling*, not identifiability.

    Raises:
        ValueError: If fewer than 4 chains are requested (the proposal needs two
            distinct donor chains besides the current one).
    """
    if n_chains < 4:
        raise ValueError("n_chains must be >= 4 for differential-evolution proposals")
    ch = check_channels(channels)
    gen = resolve_rng(rng)
    d = PRIOR_MEAN_U.shape[0]
    u_map, lp_map = fit_map(base, data, ch, n_starts=3, rng=gen)
    H = numerical_hessian(lambda u: _neg_logpost(u, base, data, ch), u_map)
    w, V = np.linalg.eigh(0.5 * (H + H.T))
    pd = bool(np.all(w > 1e-8))
    w = np.clip(w, 1e-4, None)
    L = np.linalg.cholesky((V / w) @ V.T + 1e-12 * np.eye(d))

    X = u_map + 1.5 * (L @ gen.standard_normal((d, n_chains))).T
    lp = np.array([-_neg_logpost(x, base, data, ch) for x in X])
    gamma0 = 2.38 / np.sqrt(2.0 * d)
    draws = np.empty((n_chains, n_draws, d))
    accepted = 0
    total = 0
    others = [np.array([j for j in range(n_chains) if j != i]) for i in range(n_chains)]
    for it in range(n_warmup + n_draws):
        for i in range(n_chains):
            a, b = gen.choice(others[i], 2, replace=False)
            gamma = 1.0 if gen.random() < 0.1 else gamma0
            prop = X[i] + gamma * (X[a] - X[b]) + 1e-4 * gen.standard_normal(d)
            lp_prop = -_neg_logpost(prop, base, data, ch)
            total += 1
            if np.log(gen.random()) < lp_prop - lp[i]:
                X[i], lp[i] = prop, lp_prop
                accepted += 1
        if it >= n_warmup:
            draws[:, it - n_warmup] = X
    cons = np.stack([[to_constrained(u) for u in chain] for chain in draws])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rhat = split_rhat(cons)
        ess = effective_sample_size(cons)
    return PosteriorSummary(
        "mcmc",
        to_constrained(u_map),
        cons.reshape(-1, d),
        lp_map,
        hessian_pd=pd,
        rhat=rhat,
        ess=ess,
        accept_rate=accepted / max(total, 1),
    )


def fit(
    base: ReducedParams,
    data: Observed,
    channels: Sequence[str] = CHANNELS,
    *,
    estimator: str = "laplace",
    rng: RNGLike = None,
    **kwargs: int,
) -> PosteriorSummary:
    """Dispatch to an estimator by name (``map``, ``laplace``, ``mcmc``)."""
    if estimator == "map":
        u, lp = fit_map(base, data, channels, rng=rng, **kwargs)
        p = to_constrained(u)
        return PosteriorSummary("map", p, p[None, :], lp)
    if estimator == "laplace":
        return fit_laplace(base, data, channels, rng=rng, **kwargs)
    if estimator == "mcmc":
        return fit_mcmc(base, data, channels, rng=rng, **kwargs)
    raise ValueError(f"unknown estimator {estimator!r}; expected map, laplace or mcmc")
