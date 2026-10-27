"""Sensitivity analysis of the reference model (S1-IMP-05).

* :func:`local_elasticities` — normalised local sensitivities
  ``(∂m/∂θ_j)(θ_j/m)`` of scalar summary metrics, by central differences.
* :func:`oat_sweep` — one-at-a-time sweeps across the plausible range.
* :func:`sobol_indices` — first-order and total variance-based indices
  (Saltelli sampling; Saltelli 2010 first-order and Jansen total estimators).

Metrics are computed with the hard threshold so they are deterministic
functions of the parameters (no sampling noise inflates the indices).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from core.rng import RNGLike, resolve_rng
from models.parameters import REDUCED_PARAMETER_NAMES, REDUCED_PARAMETERS
from models.reduced import ReducedParams, simulate_reduced

from .structural_identifiability import sample_plausible_points

__all__ = ["METRICS", "metric_vector", "local_elasticities", "oat_sweep", "sobol_indices"]

_NAMES = REDUCED_PARAMETER_NAMES


def _ignition_rate(tr) -> float:  # type: ignore[no-untyped-def]
    return float(tr.B.mean())


def _mean_p(tr) -> float:  # type: ignore[no-untyped-def]
    return float(tr.p.mean())


def _mean_signal(tr) -> float:  # type: ignore[no-untyped-def]
    return float(tr.S.mean())


def _final_abs_error(tr) -> float:  # type: ignore[no-untyped-def]
    return float(np.abs(tr.eps[-20:]).mean())


#: Summary metrics of a hard-threshold run.
METRICS: dict[str, Callable] = {  # type: ignore[type-arg]
    "ignition_rate": _ignition_rate,
    "mean_ignition_probability": _mean_p,
    "mean_signal": _mean_signal,
    "late_abs_error": _final_abs_error,
}


def metric_vector(
    theta: Sequence[float] | np.ndarray, stimulus: np.ndarray, base: ReducedParams | None = None
) -> dict[str, float]:
    """Evaluate every metric at ``theta`` (hard threshold)."""
    params = (base or ReducedParams()).with_estimable(theta)
    tr = simulate_reduced(params, stimulus, hard=True)
    return {name: fn(tr) for name, fn in METRICS.items()}


def local_elasticities(
    theta: Sequence[float] | np.ndarray,
    stimulus: np.ndarray,
    base: ReducedParams | None = None,
    rel_step: float = 1e-4,
) -> dict[str, dict[str, float]]:
    """Elasticity of each metric to each parameter at ``theta``.

    Parameters equal to zero have no relative scale, so their step and
    normalisation fall back to the plausible-range width (``β`` can be 0).
    """
    t0 = np.asarray(theta, dtype=float)
    m0 = metric_vector(t0, stimulus, base)
    out: dict[str, dict[str, float]] = {m: {} for m in METRICS}
    for j, name in enumerate(_NAMES):
        scale = (
            abs(t0[j])
            if abs(t0[j]) > 1e-12
            else REDUCED_PARAMETERS[j].plausible_upper - REDUCED_PARAMETERS[j].plausible_lower
        )
        h = rel_step * scale
        tp, tm = t0.copy(), t0.copy()
        tp[j] += h
        tm[j] -= h
        mp, mm = metric_vector(tp, stimulus, base), metric_vector(tm, stimulus, base)
        for m in METRICS:
            deriv = (mp[m] - mm[m]) / (2.0 * h)
            out[m][name] = (
                float(deriv * scale / m0[m]) if abs(m0[m]) > 1e-12 else float(deriv * scale)
            )
    return out


def oat_sweep(
    stimulus: np.ndarray,
    base: ReducedParams | None = None,
    n_grid: int = 11,
) -> dict[str, dict[str, np.ndarray]]:
    """Sweep each parameter across its plausible range, others at nominal.

    Returns:
        ``{parameter: {"grid": ..., metric: values...}}``.
    """
    base = base or ReducedParams()
    nominal = base.estimable()
    res: dict[str, dict[str, np.ndarray]] = {}
    for j, spec in enumerate(REDUCED_PARAMETERS):
        grid = (
            np.geomspace(spec.plausible_lower, spec.plausible_upper, n_grid)
            if spec.log_scale
            else np.linspace(spec.plausible_lower, spec.plausible_upper, n_grid)
        )
        entry: dict[str, np.ndarray] = {"grid": grid}
        vals = []
        for g in grid:
            t = nominal.copy()
            t[j] = g
            vals.append(metric_vector(t, stimulus, base))
        for m in METRICS:
            entry[m] = np.array([v[m] for v in vals])
        res[spec.name] = entry
    return res


def sobol_indices(
    stimulus: np.ndarray,
    base: ReducedParams | None = None,
    n: int = 256,
    rng: RNGLike = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """First-order (``S1``) and total (``ST``) Sobol indices over the plausible box.

    Uses two independent Latin-hypercube matrices ``A``, ``B`` and the hybrid
    matrices ``AB_j`` (column ``j`` taken from ``B``): ``S1_j`` by Saltelli
    (2010), ``ST_j`` by Jansen. Metrics with zero variance report zeros.

    Returns:
        ``{metric: {"S1": {param: v}, "ST": {param: v}}}``.
    """
    gen = resolve_rng(rng)
    A = sample_plausible_points(n, gen)
    Bm = sample_plausible_points(n, gen)
    base = base or ReducedParams()

    def evaluate(M: np.ndarray) -> dict[str, np.ndarray]:
        rows = [metric_vector(r, stimulus, base) for r in M]
        return {m: np.array([r[m] for r in rows]) for m in METRICS}

    fA, fB = evaluate(A), evaluate(Bm)
    out: dict[str, dict[str, dict[str, float]]] = {m: {"S1": {}, "ST": {}} for m in METRICS}
    var = {m: float(np.var(np.concatenate([fA[m], fB[m]]), ddof=1)) for m in METRICS}
    for j, name in enumerate(_NAMES):
        ABj = A.copy()
        ABj[:, j] = Bm[:, j]
        fAB = evaluate(ABj)
        for m in METRICS:
            if var[m] <= 1e-18:
                out[m]["S1"][name] = 0.0
                out[m]["ST"][name] = 0.0
                continue
            s1 = np.mean(fB[m] * (fAB[m] - fA[m])) / var[m]
            st = 0.5 * np.mean((fA[m] - fAB[m]) ** 2) / var[m]
            out[m]["S1"][name] = float(s1)
            out[m]["ST"][name] = float(st)
    return out
