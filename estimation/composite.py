"""``precision_weighted_learning_composite`` — the only learning quantity reported (S1-IMP-18).

Reversal behaviour constrains only the product ``αₜ·Πₜ``: the learning gain is
``k = αΠ/(1+αΠ)`` and nothing else in the behavioural output depends on the
factors (K7a, :func:`estimation.structural_identifiability.product_symmetry_gap`).
Reporting ``Π`` alone from such a task would therefore be reporting a prior, not
a measurement.

This module fits ``c = αΠ`` from behaviour, reports it with an interval, and
refuses to expose the factors until a design that separates them is declared
(:func:`design_separates_factors`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import optimize  # type: ignore[import-untyped]

from models.reduced import ReducedParams

__all__ = [
    "COMPOSITE_NAME",
    "FactorsNotSeparableError",
    "CompositeEstimate",
    "fit_composite",
    "composite_from_samples",
    "design_separates_factors",
    "expose_factors",
]

COMPOSITE_NAME = "precision_weighted_learning_composite"


class FactorsNotSeparableError(RuntimeError):
    """Raised on any attempt to report ``α`` or ``Π`` alone from a non-separating design."""


@dataclass(frozen=True)
class CompositeEstimate:
    """Estimate of ``c = αΠ``.

    The dataclass deliberately has no ``alpha`` or ``pi`` field: outputs expose
    the product only.
    """

    value: float
    ci95: tuple[float, float]
    method: str
    n_trials: int
    name: str = COMPOSITE_NAME

    def as_dict(self) -> dict[str, object]:
        return {
            self.name: self.value,
            "ci95": list(self.ci95),
            "method": self.method,
            "n_trials": self.n_trials,
            "factors_exposed": False,
        }


def _xhat_path(k: float, stimulus: np.ndarray) -> np.ndarray:
    out = np.empty_like(stimulus)
    xhat = 0.0
    for t, x in enumerate(stimulus):
        out[t] = xhat
        xhat += k * (x - xhat)
    return out


def fit_composite(
    stimulus: Sequence[float] | np.ndarray,
    report: Sequence[float] | np.ndarray,
    sigma_r: float = ReducedParams().sigma_r,
) -> CompositeEstimate:
    """Maximum-likelihood ``c = αΠ`` from the behavioural report alone.

    The report is ``r_t ~ N(x̂_t(k), σ_r²)``; the gain ``k`` is fitted by bounded
    1-D optimisation and the interval comes from the observed information at the
    optimum, mapped to ``c = k/(1−k)`` by the delta method.

    Raises:
        ValueError: On mismatched lengths or ``sigma_r <= 0``.
    """
    x = np.asarray(stimulus, dtype=float)
    r = np.asarray(report, dtype=float)
    if x.shape != r.shape or x.ndim != 1:
        raise ValueError("stimulus and report must be 1-D arrays of equal length")
    if sigma_r <= 0:
        raise ValueError("sigma_r must be > 0")

    def sse(k: float) -> float:
        return float(np.sum((r - _xhat_path(k, x)) ** 2))

    res = optimize.minimize_scalar(
        sse, bounds=(1e-6, 1.0 - 1e-6), method="bounded", options={"xatol": 1e-10}
    )
    k_hat = float(res.x)
    h = 1e-5
    curv = (sse(min(k_hat + h, 1 - 1e-7)) - 2 * sse(k_hat) + sse(max(k_hat - h, 1e-7))) / h**2
    se_k = sigma_r / math.sqrt(curv) if curv > 0 else float("inf")
    c_hat = k_hat / (1.0 - k_hat)
    dc_dk = 1.0 / (1.0 - k_hat) ** 2
    se_c = dc_dk * se_k
    return CompositeEstimate(
        value=c_hat,
        ci95=(max(c_hat - 1.959964 * se_c, 0.0), c_hat + 1.959964 * se_c),
        method="behavior-ml",
        n_trials=int(x.shape[0]),
    )


def composite_from_samples(
    alpha: np.ndarray, pi: np.ndarray, n_trials: int = 0
) -> CompositeEstimate:
    """Composite from joint posterior draws — only the product leaves this function."""
    a, p = np.asarray(alpha, dtype=float), np.asarray(pi, dtype=float)
    if a.shape != p.shape:
        raise ValueError("alpha and pi draws must have the same shape")
    c = a * p
    lo, hi = np.quantile(c, [0.025, 0.975])
    return CompositeEstimate(float(c.mean()), (float(lo), float(hi)), "posterior-product", n_trials)


def design_separates_factors(channels: Sequence[str], *, n_points: int = 40, seed: int = 0) -> bool:
    """True iff K7a finds ``α`` and ``Π`` separable under ``channels``.

    Runs the suspect-pair test on a plausible-range sweep, so the answer comes
    from the same criterion the gate table uses rather than from a hard-coded
    rule about channel names.
    """
    from .structural_identifiability import (
        AnalysisSetup,
        sweep_identifiability,
        test_pair,
    )

    _, analyses = sweep_identifiability(channels, AnalysisSetup(), n_points=n_points, seed=seed)
    return test_pair(analyses, "alpha", "pi")["classification"] == "identifiable"


def expose_factors(
    posterior_alpha: np.ndarray,
    posterior_pi: np.ndarray,
    channels: Sequence[str],
    *,
    separable: bool | None = None,
) -> dict[str, np.ndarray]:
    """Return the individual factors only if the design separates them.

    Args:
        posterior_alpha: Draws of ``α``.
        posterior_pi: Draws of ``Π``.
        channels: Observation channels of the design.
        separable: Precomputed verdict of :func:`design_separates_factors`.

    Raises:
        FactorsNotSeparableError: If the factors are not separable under ``channels``.
    """
    ok = design_separates_factors(channels) if separable is None else separable
    if not ok:
        raise FactorsNotSeparableError(
            f"design with channels {tuple(channels)} does not separate α from Π; "
            f"report {COMPOSITE_NAME} instead"
        )
    return {"alpha": np.asarray(posterior_alpha), "pi": np.asarray(posterior_pi)}
