"""Profile-likelihood identifiability (practical, data-conditional).

For each parameter the likelihood is maximised over the others on a grid of
fixed values. A parameter is *practically identifiable* when the profile rises
above the χ²₁ 95% threshold (deviance 3.84) on **both** sides inside the
parameter's hard support; a profile that stays under the threshold on a side
leaves that side of the confidence interval unbounded, i.e. a flat ridge.

Structural analysis (:mod:`estimation.structural_identifiability`) asks whether
parameters *could* be determined from noise-free data; the profile asks whether
this particular data set determines them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import optimize  # type: ignore[import-untyped]

from models.parameters import REDUCED_PARAMETER_NAMES
from models.reduced import CHANNELS, ReducedParams

from .likelihood import BOUNDS, Observed, check_channels, log_likelihood

__all__ = ["CHI2_95_1DF", "ProfileResult", "fit_mle", "profile_parameter", "profile_all"]

#: χ²₁ 95% quantile: deviance threshold for a 95% profile confidence interval.
CHI2_95_1DF: float = 3.841458820694124

_NAMES = REDUCED_PARAMETER_NAMES


def _neg_ll(
    free_idx: list[int],
    fixed: dict[int, float],
    base: ReducedParams,
    data: Observed,
    channels: tuple[str, ...],
    x: np.ndarray,
) -> float:
    theta = np.empty(len(_NAMES))
    for i, v in fixed.items():
        theta[i] = v
    theta[free_idx] = x
    try:
        val = log_likelihood(base.with_estimable(theta), data, channels)
    except ValueError:
        return 1e12
    return -val if np.isfinite(val) else 1e12


def _maximise(
    base: ReducedParams,
    data: Observed,
    channels: tuple[str, ...],
    fixed: dict[int, float],
    starts: Sequence[np.ndarray],
) -> tuple[np.ndarray, float]:
    free = [i for i in range(len(_NAMES)) if i not in fixed]
    bounds = [(BOUNDS[i, 0], BOUNDS[i, 1]) for i in free]
    best_x, best = None, np.inf
    for s in starts:
        x0 = np.clip(
            np.asarray(s, dtype=float)[free], [b[0] for b in bounds], [b[1] for b in bounds]
        )
        res = optimize.minimize(
            lambda x: _neg_ll(free, fixed, base, data, channels, x),
            x0,
            method="L-BFGS-B",
            bounds=bounds,
        )
        if res.fun < best:
            best_x, best = np.asarray(res.x, dtype=float), float(res.fun)
    assert best_x is not None
    theta = np.empty(len(_NAMES))
    for i, v in fixed.items():
        theta[i] = v
    theta[free] = best_x
    return theta, -best


def fit_mle(
    base: ReducedParams,
    data: Observed,
    channels: Sequence[str] = CHANNELS,
    *,
    starts: Sequence[np.ndarray] | None = None,
) -> tuple[np.ndarray, float]:
    """Maximum-likelihood estimate (bounded L-BFGS-B, several starts)."""
    ch = check_channels(channels)
    if starts is None:
        mid = 0.5 * (BOUNDS[:, 0] + BOUNDS[:, 1])
        starts = [base.estimable(), mid, BOUNDS[:, 0] + 0.25 * (BOUNDS[:, 1] - BOUNDS[:, 0])]
    return _maximise(base, data, ch, {}, starts)


@dataclass
class ProfileResult:
    """Profile for one parameter."""

    name: str
    grid: np.ndarray
    deviance: np.ndarray
    mle: float
    ci_lower: float | None
    ci_upper: float | None
    status: str

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "mle": self.mle,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "status": self.status,
            "grid": [float(g) for g in self.grid],
            "deviance": [float(d) for d in self.deviance],
        }


def _crossing(grid: np.ndarray, dev: np.ndarray, mle_idx: int, direction: int) -> float | None:
    """Linearly interpolated point where the deviance crosses the threshold, or None."""
    idx = mle_idx
    while 0 <= idx + direction < len(grid):
        nxt = idx + direction
        if dev[nxt] >= CHI2_95_1DF:
            d0, d1 = dev[idx], dev[nxt]
            frac = (CHI2_95_1DF - d0) / (d1 - d0) if d1 > d0 else 1.0
            return float(grid[idx] + frac * (grid[nxt] - grid[idx]))
        idx = nxt
    return None


def profile_parameter(
    base: ReducedParams,
    data: Observed,
    channels: Sequence[str],
    name: str,
    *,
    n_grid: int = 21,
    mle: tuple[np.ndarray, float] | None = None,
) -> ProfileResult:
    """Profile likelihood of one parameter over its full hard support.

    Args:
        base: Noise levels and starting values.
        data: Observations.
        channels: Channels in the likelihood.
        name: Parameter to profile.
        n_grid: Grid points across the support (the MLE is inserted).
        mle: Precomputed ``(theta_hat, loglik)`` to avoid refitting.
    """
    ch = check_channels(channels)
    j = _NAMES.index(name)
    theta_hat, ll_hat = mle if mle is not None else fit_mle(base, data, ch)
    grid = np.unique(np.append(np.linspace(BOUNDS[j, 0], BOUNDS[j, 1], n_grid), theta_hat[j]))
    mle_idx = int(np.argmin(np.abs(grid - theta_hat[j])))
    dev = np.empty(len(grid))
    dev[mle_idx] = 0.0
    # Walk outward from the MLE, warm-starting each fit from its neighbour.
    for direction in (+1, -1):
        prev = theta_hat
        idx = mle_idx + direction
        while 0 <= idx < len(grid):
            theta, ll = _maximise(base, data, ch, {j: float(grid[idx])}, [prev, theta_hat])
            dev[idx] = max(2.0 * (ll_hat - ll), 0.0)
            prev = theta
            idx += direction
    lo = _crossing(grid, dev, mle_idx, -1)
    hi = _crossing(grid, dev, mle_idx, +1)
    if lo is not None and hi is not None:
        status = "identifiable"
    elif lo is None and hi is None:
        status = "non-identifiable"
    else:
        status = "one-sided"
    return ProfileResult(name, grid, dev, float(theta_hat[j]), lo, hi, status)


def profile_all(
    base: ReducedParams,
    data: Observed,
    channels: Sequence[str] = CHANNELS,
    *,
    n_grid: int = 21,
) -> dict[str, ProfileResult]:
    """Profile every estimable parameter (shares a single MLE fit)."""
    ch = check_channels(channels)
    mle = fit_mle(base, data, ch)
    return {n: profile_parameter(base, data, ch, n, n_grid=n_grid, mle=mle) for n in _NAMES}
