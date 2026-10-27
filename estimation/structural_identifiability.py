"""K7a — structural identifiability of the reference model.

Three complementary analyses, all on the noise-free output map of the
teacher-forced model (conditioning on the observed ignition train, exactly as
inference does):

1. **Symmetry / differential-algebra result.** The behavioural output depends on
   ``(α, Π)`` only through ``k = αΠ/(1+αΠ)``; every pair on a level set of the
   product yields identical behaviour. :func:`product_symmetry_gap` verifies
   this to machine precision, and shows which observation channels break it.
2. **Sensitivity rank and conditioning across the plausible range.** Full rank
   at a single point shows *local* identifiability only, so the Jacobian of the
   noise-scaled output map is formed at many points of the plausible box.
   Outputs are scaled so that ``JᵀJ`` is the Fisher information and columns are
   scaled by the plausible-range width, so singular values read as "signal, in
   noise SDs, from moving a parameter across its whole plausible range".
3. **Equivalent solutions and subset trade-offs.** Multi-start least squares on
   noise-free targets finds distinct parameter vectors with indistinguishable
   output (symmetries); null vectors with three or more active parameters flag
   trade-offs that pairwise checks miss. Suspect pairs are tested explicitly.

Output-map blocks (``σ`` are the known noise SDs):

* behavior   — ``x̂_t / σ_r``
* amplitude  — ``S_t / σ_y``
* ignition   — ``2·arcsin√p_t`` (variance-stabilising, so Bernoulli information
  is ``(∂f)²``)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize  # type: ignore[import-untyped]

from core.rng import RNGLike, resolve_rng
from models.parameters import REDUCED_PARAMETER_NAMES, REDUCED_PARAMETERS
from models.reduced import CHANNELS, ReducedParams, simulate_reduced

from .likelihood import check_channels
from .synthetic_generator import Design, reversal_stimulus

__all__ = [
    "RANK_TOL",
    "COND_LIMIT",
    "COLLINEAR_LIMIT",
    "COVERAGE_REQUIRED",
    "SUSPECT_PAIRS",
    "WIDTH",
    "AnalysisSetup",
    "output_map",
    "sensitivity_matrix",
    "sample_plausible_points",
    "analyze_point",
    "PointAnalysis",
    "sweep_identifiability",
    "SweepResult",
    "product_symmetry_gap",
    "partial_collinearity",
    "test_pair",
    "multi_start_equivalents",
    "classify_parameters",
    "composite_identifiability",
    "run_k7a",
]

#: Relative singular-value tolerance below which a direction counts as exactly
#: unobservable (finite-difference noise is ~1e-8 of the leading value).
RANK_TOL: float = 1e-6
#: Condition number (of the range-normalised Jacobian) above which a point is
#: considered ill-conditioned.
COND_LIMIT: float = 1e3
#: |cos| of partial collinearity at or above which a pair is degenerate
#: (variance inflation 1/(1−c²) ≈ 50).
COLLINEAR_LIMIT: float = 0.99
#: Fraction of sweep points at which a parameter must be identifiable for CONFIRMED.
COVERAGE_REQUIRED: float = 0.95
#: Pairs the register names as known degeneracy candidates: α–Π and θₜ–β.
SUSPECT_PAIRS: tuple[tuple[str, str], ...] = (("alpha", "pi"), ("theta_base", "beta"))

NAMES = REDUCED_PARAMETER_NAMES
#: Plausible-range widths used to normalise Jacobian columns.
WIDTH: np.ndarray = np.array([p.plausible_upper - p.plausible_lower for p in REDUCED_PARAMETERS])


@dataclass
class AnalysisSetup:
    """Design and nuisance settings shared by all analyses."""

    design: Design = field(default_factory=Design)
    base: ReducedParams = field(default_factory=ReducedParams)
    seed: int = 0

    def stimulus(self) -> np.ndarray:
        return reversal_stimulus(self.design, np.random.default_rng(self.seed))


def output_map(
    theta: np.ndarray,
    stimulus: np.ndarray,
    B_ref: np.ndarray,
    channels: Sequence[str],
    base: ReducedParams,
) -> np.ndarray:
    """Noise-scaled, teacher-forced output vector ``f(θ)``."""
    p = base.with_estimable(theta)
    tr = simulate_reduced(p, stimulus, B_forced=B_ref)
    blocks = []
    if "behavior" in channels:
        blocks.append(tr.xhat / p.sigma_r)
    if "ignition" in channels:
        blocks.append(2.0 * np.arcsin(np.sqrt(np.clip(tr.p, 1e-12, 1.0 - 1e-12))))
    if "amplitude" in channels:
        blocks.append(tr.S / p.sigma_y)
    return np.concatenate(blocks)


def sensitivity_matrix(
    theta: np.ndarray,
    stimulus: np.ndarray,
    B_ref: np.ndarray,
    channels: Sequence[str],
    base: ReducedParams,
    rel_step: float = 1e-6,
) -> np.ndarray:
    """Range-normalised Jacobian ``∂f/∂θ_j · width_j`` by central differences."""
    cols = []
    for j in range(len(NAMES)):
        h = rel_step * WIDTH[j]
        tp, tm = theta.copy(), theta.copy()
        tp[j] += h
        tm[j] -= h
        cols.append(
            (
                output_map(tp, stimulus, B_ref, channels, base)
                - output_map(tm, stimulus, B_ref, channels, base)
            )
            / (2.0 * h)
            * WIDTH[j]
        )
    return np.stack(cols, axis=1)


def sample_plausible_points(n: int, rng: RNGLike = None) -> np.ndarray:
    """Latin-hypercube sample of the plausible box (log-uniform where specified)."""
    gen = resolve_rng(rng)
    d = len(NAMES)
    u = (np.argsort(gen.random((n, d)), axis=0) + gen.random((n, d))) / n
    pts = np.empty((n, d))
    for j, spec in enumerate(REDUCED_PARAMETERS):
        if spec.log_scale:
            lo, hi = np.log(spec.plausible_lower), np.log(spec.plausible_upper)
            pts[:, j] = np.exp(lo + (hi - lo) * u[:, j])
        else:
            pts[:, j] = spec.plausible_lower + WIDTH[j] * u[:, j]
    return pts


@dataclass
class PointAnalysis:
    """Rank/conditioning of the sensitivity matrix at one point."""

    theta: np.ndarray
    singular_values: np.ndarray
    rank: int
    condition: float
    null_vectors: np.ndarray
    J: np.ndarray = field(repr=False)

    def nonidentifiable_mask(self, support_tol: float = 0.1) -> np.ndarray:
        """Parameters with weight above ``support_tol`` in some exact null vector."""
        if self.null_vectors.shape[1] == 0:
            return np.zeros(len(NAMES), dtype=bool)
        return np.asarray(np.any(np.abs(self.null_vectors) > support_tol, axis=1))


def analyze_point(
    theta: np.ndarray,
    stimulus: np.ndarray,
    channels: Sequence[str],
    base: ReducedParams,
    *,
    ref_seed: int = 0,
) -> PointAnalysis:
    """Rank, condition number and null space at ``theta``.

    The reference ignition train is drawn from the model *at* ``theta`` with a
    fixed seed, mimicking the data that would be collected there.
    """
    p = base.with_estimable(theta)
    B_ref = simulate_reduced(p, stimulus, rng=ref_seed).B
    J = sensitivity_matrix(theta, stimulus, B_ref, channels, base)
    U, s, Vt = np.linalg.svd(J, full_matrices=True)
    smax = float(s[0]) if s.size else 0.0
    rank = int(np.sum(s > RANK_TOL * smax)) if smax > 0 else 0
    null = Vt[rank:].T
    cond = float(smax / s[rank - 1]) if rank >= 1 else float("inf")
    if rank < len(NAMES):
        cond = float("inf")
    return PointAnalysis(theta.copy(), s, rank, cond, null, J)


@dataclass
class SweepResult:
    """Aggregate of :func:`analyze_point` over the plausible range."""

    channels: tuple[str, ...]
    n_points: int
    full_rank_fraction: float
    ident_fraction: dict[str, float]
    median_condition: float
    p95_condition: float
    ill_conditioned_fraction: float
    min_singular_median: float
    rank_counts: dict[int, int]

    def as_dict(self) -> dict[str, object]:
        return {
            "channels": list(self.channels),
            "n_points": self.n_points,
            "full_rank_fraction": self.full_rank_fraction,
            "identifiable_fraction": self.ident_fraction,
            "median_condition": self.median_condition,
            "p95_condition": self.p95_condition,
            "ill_conditioned_fraction": self.ill_conditioned_fraction,
            "median_min_singular_value": self.min_singular_median,
            "rank_counts": {str(k): v for k, v in sorted(self.rank_counts.items())},
        }


def sweep_identifiability(
    channels: Sequence[str],
    setup: AnalysisSetup | None = None,
    *,
    n_points: int = 60,
    seed: int = 0,
) -> tuple[SweepResult, list[PointAnalysis]]:
    """Sensitivity rank and conditioning at ``n_points`` points of the plausible box."""
    ch = check_channels(channels)
    setup = setup or AnalysisSetup()
    x = setup.stimulus()
    pts = sample_plausible_points(n_points, seed)
    analyses = [
        analyze_point(t, x, ch, setup.base, ref_seed=setup.seed + i) for i, t in enumerate(pts)
    ]
    ident = np.stack([~a.nonidentifiable_mask() for a in analyses])
    conds = np.array([a.condition for a in analyses])
    finite = conds[np.isfinite(conds)]
    ranks = [a.rank for a in analyses]
    return (
        SweepResult(
            channels=ch,
            n_points=n_points,
            full_rank_fraction=float(np.mean([r == len(NAMES) for r in ranks])),
            ident_fraction={n: float(ident[:, j].mean()) for j, n in enumerate(NAMES)},
            median_condition=float(np.median(finite)) if finite.size else float("inf"),
            p95_condition=float(np.quantile(finite, 0.95)) if finite.size else float("inf"),
            ill_conditioned_fraction=float(np.mean(~(conds <= COND_LIMIT))),
            min_singular_median=float(np.median([a.singular_values[-1] for a in analyses])),
            rank_counts={r: ranks.count(r) for r in set(ranks)},
        ),
        analyses,
    )


def product_symmetry_gap(
    setup: AnalysisSetup | None = None,
    channels: Sequence[str] = ("behavior",),
    *,
    scale: float = 2.0,
) -> float:
    """Max output difference between ``(α, Π)`` and ``(α·s, Π/s)`` (same product).

    For the behaviour channel this is zero to machine precision — the exact
    ``αΠ`` symmetry. Channels that see Π separately (ignition, amplitude) break
    it and return a clearly nonzero gap.
    """
    ch = check_channels(channels)
    setup = setup or AnalysisSetup()
    x = setup.stimulus()
    base = setup.base
    B_ref = simulate_reduced(base, x, rng=setup.seed).B
    alt = ReducedParams(
        alpha=base.alpha * scale,
        pi=base.pi / scale,
        theta_base=base.theta_base,
        beta=base.beta,
        sigma_r=base.sigma_r,
        sigma_y=base.sigma_y,
    )
    f0 = output_map(base.estimable(), x, B_ref, ch, base)
    f1 = output_map(alt.estimable(), x, B_ref, ch, base)
    return float(np.max(np.abs(f0 - f1)))


def partial_collinearity(J: np.ndarray, i: int, j: int) -> float:
    """|cos| between columns ``i`` and ``j`` after projecting out all other columns.

    Returns 1.0 if either residual column vanishes (the parameter is explained
    entirely by the others).
    """
    others = [k for k in range(J.shape[1]) if k not in (i, j)]
    Ji, Jj = J[:, i].copy(), J[:, j].copy()
    if others:
        Q, _ = np.linalg.qr(J[:, others])
        Ji -= Q @ (Q.T @ Ji)
        Jj -= Q @ (Q.T @ Jj)
    ni, nj = float(np.linalg.norm(Ji)), float(np.linalg.norm(Jj))
    scale = max(float(np.linalg.norm(J[:, i])), float(np.linalg.norm(J[:, j])), 1e-300)
    if ni < RANK_TOL * scale or nj < RANK_TOL * scale:
        return 1.0
    return float(min(1.0, abs(Ji @ Jj) / (ni * nj)))


def test_pair(analyses: Sequence[PointAnalysis], a: str, b: str) -> dict[str, object]:
    """Classify a pair across the sweep as ``identifiable`` or ``degenerate``."""
    i, j = NAMES.index(a), NAMES.index(b)
    c = np.array([partial_collinearity(an.J, i, j) for an in analyses])
    frac_deg = float(np.mean(c >= COLLINEAR_LIMIT))
    return {
        "pair": [a, b],
        "median_partial_collinearity": float(np.median(c)),
        "p95_partial_collinearity": float(np.quantile(c, 0.95)),
        "degenerate_fraction": frac_deg,
        "classification": "degenerate" if frac_deg > 1.0 - COVERAGE_REQUIRED else "identifiable",
    }


def multi_start_equivalents(
    channels: Sequence[str],
    setup: AnalysisSetup | None = None,
    *,
    theta_true: np.ndarray | None = None,
    n_starts: int = 12,
    seed: int = 0,
    cost_tol: float = 1e-6,
    distance_tol: float = 0.05,
) -> dict[str, object]:
    """Search for parameter vectors that reproduce the true noise-free output.

    Least squares is run from random starts on noise-free targets. A solution
    whose cost is below ``cost_tol`` (summed squared noise-SD units, per output)
    but whose range-normalised distance from the truth exceeds ``distance_tol``
    is an *equivalent solution*. Null vectors at the truth with three or more
    active parameters are reported as subset trade-offs.
    """
    ch = check_channels(channels)
    setup = setup or AnalysisSetup()
    x = setup.stimulus()
    base = setup.base
    truth = base.estimable() if theta_true is None else np.asarray(theta_true, dtype=float)
    B_ref = simulate_reduced(base.with_estimable(truth), x, rng=setup.seed).B
    target = output_map(truth, x, B_ref, ch, base)
    lo = np.array([p.plausible_lower for p in REDUCED_PARAMETERS])
    hi = np.array([p.plausible_upper for p in REDUCED_PARAMETERS])
    gen = resolve_rng(seed)

    def resid(t: np.ndarray) -> np.ndarray:
        return np.asarray(output_map(t, x, B_ref, ch, base) - target)

    equivalents: list[list[float]] = []
    for _ in range(n_starts):
        t0 = lo + (hi - lo) * gen.random(len(NAMES))
        sol = optimize.least_squares(resid, t0, bounds=(lo, hi), xtol=1e-12, ftol=1e-12, gtol=1e-12)
        cost = float(np.mean(sol.fun**2))
        dist = float(np.max(np.abs(sol.x - truth) / WIDTH))
        if cost < cost_tol and dist > distance_tol:
            equivalents.append([float(v) for v in sol.x])
    an = analyze_point(truth, x, ch, base, ref_seed=setup.seed)
    trade_offs = []
    for v in an.null_vectors.T:
        active = [NAMES[j] for j in range(len(NAMES)) if abs(v[j]) > 0.1]
        trade_offs.append({"active": active, "vector": [float(c) for c in v]})
    U, sv, Vt = np.linalg.svd(an.J)
    weakest = Vt[-1]
    return {
        "channels": list(ch),
        "weakest_direction": {
            "singular_value": float(sv[-1]),
            "vector": {n: float(c) for n, c in zip(NAMES, weakest, strict=True)},
            "n_active_gt_0.3": int(np.sum(np.abs(weakest) > 0.3)),
        },
        "n_starts": n_starts,
        "n_equivalent_solutions": len(equivalents),
        "equivalent_solutions": equivalents[:5],
        "null_vectors": trade_offs,
        "subset_trade_off": any(len(t["active"]) >= 3 for t in trade_offs),  # type: ignore[arg-type]
        "pair_trade_off": any(len(t["active"]) == 2 for t in trade_offs),  # type: ignore[arg-type]
    }


def classify_parameters(
    sweep: SweepResult,
    pairs: dict[str, dict[str, object]],
    equivalents: dict[str, object],
) -> dict[str, str]:
    """K7a classification per parameter: CONFIRMED, FALSIFIED or INDETERMINATE.

    * FALSIFIED — the parameter lies in an exact null direction at more than half
      of the plausible range, or has an equivalent solution at the truth.
    * CONFIRMED — identifiable at ≥ ``COVERAGE_REQUIRED`` of points, median
      condition ≤ ``COND_LIMIT``, no degenerate suspect pair, no equivalent
      solution or trade-off.
    * INDETERMINATE — otherwise (e.g. identifiable but ill-conditioned).
    """
    degenerate_members = {
        n
        for info in pairs.values()
        if info["classification"] == "degenerate"
        for n in info["pair"]  # type: ignore[attr-defined]
    }
    has_equiv = int(equivalents["n_equivalent_solutions"]) > 0  # type: ignore[call-overload]
    out: dict[str, str] = {}
    for name in NAMES:
        frac = sweep.ident_fraction[name]
        if frac < 0.5 or name in degenerate_members or has_equiv:
            out[name] = "FALSIFIED"
        elif (
            frac >= COVERAGE_REQUIRED
            and sweep.median_condition <= COND_LIMIT
            and not equivalents["subset_trade_off"]
        ):
            out[name] = "CONFIRMED"
        else:
            out[name] = "INDETERMINATE"
    return out


def composite_identifiability(analyses: Sequence[PointAnalysis]) -> dict[str, object]:
    """Identifiability of the composite ``c = αΠ`` (range-normalised gradient).

    A function ``g(θ)`` is identifiable at a point iff its gradient has no
    component in the exact null space of the sensitivity matrix. The gradient
    is expressed in the same range-normalised units as the Jacobian columns.
    """
    ia, ip = NAMES.index("alpha"), NAMES.index("pi")
    leaks = []
    for an in analyses:
        g = np.zeros(len(NAMES))
        g[ia] = an.theta[ip] * WIDTH[ia]
        g[ip] = an.theta[ia] * WIDTH[ip]
        if an.null_vectors.shape[1] == 0:
            leaks.append(0.0)
        else:
            leaks.append(float(np.linalg.norm(an.null_vectors.T @ g) / np.linalg.norm(g)))
    arr = np.asarray(leaks)
    frac = float(np.mean(arr < 1e-6))
    return {
        "composite": "precision_weighted_learning_composite",
        "identifiable_fraction": frac,
        "max_null_leak": float(arr.max()),
        "status": "CONFIRMED"
        if frac >= COVERAGE_REQUIRED
        else "FALSIFIED"
        if frac < 0.5
        else "INDETERMINATE",
    }


def run_k7a(
    channels: Sequence[str] = CHANNELS,
    setup: AnalysisSetup | None = None,
    *,
    n_points: int = 60,
    n_starts: int = 12,
    seed: int = 0,
) -> dict[str, object]:
    """Run the full K7a battery for one observation-channel set.

    Returns a JSON-serialisable results table: symmetry gaps, sweep summary,
    suspect-pair classification, equivalent-solution search and the resulting
    per-parameter classification.
    """
    ch = check_channels(channels)
    setup = setup or AnalysisSetup()
    sweep, analyses = sweep_identifiability(ch, setup, n_points=n_points, seed=seed)
    pairs = {f"{a}-{b}": test_pair(analyses, a, b) for a, b in SUSPECT_PAIRS}
    equivalents = multi_start_equivalents(ch, setup, n_starts=n_starts, seed=seed)
    return {
        "channels": list(ch),
        "product_symmetry_gap": {
            "behavior_only": product_symmetry_gap(setup, ("behavior",)),
            "this_channel_set": product_symmetry_gap(setup, ch),
        },
        "sweep": sweep.as_dict(),
        "suspect_pairs": pairs,
        "equivalent_solutions": equivalents,
        "classification": classify_parameters(sweep, pairs, equivalents),
        "composite": composite_identifiability(analyses),
    }
