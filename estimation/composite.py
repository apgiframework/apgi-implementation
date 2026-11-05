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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import optimize  # type: ignore[import-untyped]

from core.finite import require_finite_array, require_positive_finite
from models.reduced import ReducedParams

from .names import COMPOSITE_NAME

__all__ = [
    "COMPOSITE_NAME",
    "FactorsNotSeparableError",
    "CompositeEstimate",
    "fit_composite",
    "composite_from_samples",
    "design_separates_factors",
    "diagnostic_factors",
    "expose_factors",
]

#: Search interval for the learning gain k.
_K_BOUNDS: tuple[float, float] = (1e-6, 1.0 - 1e-6)
#: Two-sided 95% standard-normal quantile, scipy.stats.norm.ppf(0.975).
_Z95 = 1.959963984540054


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

    Observed information (AUD-DEF-04). The Gaussian negative log-likelihood is
    ``NLL(k) = SSE(k)/(2σ_r²) + const``, so its curvature is ``SSE''(k)/(2σ_r²)``
    and ``SE(k) = sqrt(2)·σ_r/sqrt(SSE''(k))``. The previous code omitted the
    factor 2, understating every standard error by 1/√2 (≈ 0.7071) and
    producing intervals with ~83% rather than 95% nominal coverage.

    If the optimum sits on the search boundary the quadratic approximation is
    invalid; the method is then labelled ``behavior-ml-boundary`` and the
    interval is left unbounded on that side rather than reported as precise.

    Raises:
        ValueError: On mismatched lengths, non-finite data or ``sigma_r <= 0``.
    """
    x = require_finite_array("stimulus", stimulus, allow_empty=False)
    r = require_finite_array("report", report, allow_empty=False)
    if x.shape != r.shape or x.ndim != 1:
        raise ValueError("stimulus and report must be 1-D arrays of equal length")
    sigma_r = require_positive_finite("sigma_r", sigma_r)

    def sse(k: float) -> float:
        return float(np.sum((r - _xhat_path(k, x)) ** 2))

    res = optimize.minimize_scalar(
        sse, bounds=_K_BOUNDS, method="bounded", options={"xatol": 1e-10}
    )
    k_hat = float(res.x)
    lo_b, hi_b = _K_BOUNDS
    h = 1e-5
    on_boundary = k_hat - lo_b < 10 * h or hi_b - k_hat < 10 * h
    c_hat = k_hat / (1.0 - k_hat)
    if on_boundary:
        lower = 0.0 if k_hat - lo_b < 10 * h else c_hat
        upper = math.inf if hi_b - k_hat < 10 * h else c_hat
        return CompositeEstimate(
            value=c_hat,
            ci95=(lower, upper),
            method="behavior-ml-boundary",
            n_trials=int(x.shape[0]),
        )
    curv = (sse(k_hat + h) - 2 * sse(k_hat) + sse(k_hat - h)) / h**2
    se_k = math.sqrt(2.0) * sigma_r / math.sqrt(curv) if curv > 0 else float("inf")
    dc_dk = 1.0 / (1.0 - k_hat) ** 2
    se_c = dc_dk * se_k
    return CompositeEstimate(
        value=c_hat,
        ci95=(max(c_hat - _Z95 * se_c, 0.0), c_hat + _Z95 * se_c),
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


def diagnostic_factors(
    posterior_alpha: np.ndarray,
    posterior_pi: np.ndarray,
    channels: Sequence[str],
    *,
    separable: bool | None = None,
) -> dict[str, object]:
    """Internal, diagnostic-only access to the factors (never for reporting).

    Used by the identifiability machinery itself. The result is labelled
    ``diagnostic=True`` and ``reportable=False`` so it cannot be mistaken for a
    production output (AUD-DEF-11).

    Raises:
        FactorsNotSeparableError: If the design does not separate α from Π.
    """
    ok = design_separates_factors(channels) if separable is None else separable
    if not ok:
        raise FactorsNotSeparableError(
            f"design with channels {tuple(channels)} does not separate α from Π; "
            f"report {COMPOSITE_NAME} instead"
        )
    return {
        "alpha": np.asarray(posterior_alpha),
        "pi": np.asarray(posterior_pi),
        "diagnostic": True,
        "reportable": False,
    }


def expose_factors(
    posterior_alpha: np.ndarray,
    posterior_pi: np.ndarray,
    channels: Sequence[str],
    *,
    gate_table: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Production reporting of the individual factors.

    There is no caller-supplied bypass (AUD-DEF-11). All of the following must
    hold, or :class:`FactorsNotSeparableError` is raised:

    * the design separates α from Π by the K7a criterion (re-computed here);
    * the gate table was built for exactly these ``channels``;
    * the table is approved and both ``alpha`` and ``pi`` are ``exposable`` —
      which requires K7a, K7b *and* K7c CONFIRMED (see
      :func:`estimation.gate_table.build_gate_table`). Desk-only verdicts with
      K7b pending are provisional and never authorise exposure.
    """
    from .gate_table import allow_list

    if list(gate_table.get("design", {}).get("channels", [])) != list(channels):
        raise FactorsNotSeparableError(
            f"gate table design {gate_table.get('design', {}).get('channels')} does not "
            f"match channels {list(channels)}"
        )
    allowed = set(allow_list(gate_table))
    missing = {"alpha", "pi"} - allowed
    if missing:
        raise FactorsNotSeparableError(
            f"gate table does not authorise exposure of {sorted(missing)} "
            f"(requires approved table with K7a, K7b and K7c CONFIRMED); "
            f"report {COMPOSITE_NAME} instead"
        )
    if not design_separates_factors(channels):
        raise FactorsNotSeparableError(
            f"design with channels {tuple(channels)} does not separate α from Π"
        )
    return {"alpha": np.asarray(posterior_alpha), "pi": np.asarray(posterior_pi)}
