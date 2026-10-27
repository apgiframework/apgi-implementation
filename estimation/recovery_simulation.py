"""K7c — parameter-recovery simulation at realistic noise (S1-IMP-05, S1-K7-11).

Procedure: draw ``N`` true parameter vectors from the plausible box, generate a
dataset for each with the shared ODE generator
(:mod:`estimation.synthetic_generator`), fit it, and compare estimates with
truth. Per parameter the following are reported, overall and per **tercile of
the true value** (low / middle / high):

* recovery ``r`` — Pearson correlation of posterior mean with truth;
* bias — mean of (estimate − truth);
* nRMSE — RMSE divided by the plausible-range width;
* 95% coverage — fraction of central 95% credible intervals containing truth;
* mean absolute pairwise posterior correlation, per parameter pair;
* for sampling estimators, the worst split-R̂ and smallest ESS and the fraction of
  subjects meeting R̂ ≤ 1.05 and ESS ≥ 200 (convergence is reported *separately*
  from identifiability; a parameter counts as converged when ≥ 95% of subjects do).

Acceptance (register S1-K7-11): recovery ``r ≥ 0.70``; pairwise posterior
``|r| < 0.60``; 95% coverage in ``[0.90, 0.98]``.

Composite recovery for ``αΠ`` is reported alongside, since the individual
factors are not separately constrained by behaviour alone.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from core.rng import RNGLike, resolve_rng, spawn_rng
from models.parameters import REDUCED_PARAMETER_NAMES, REDUCED_PARAMETERS
from models.reduced import CHANNELS, ReducedParams

from .fitting import fit
from .likelihood import Observed, check_channels
from .structural_identifiability import sample_plausible_points
from .synthetic_generator import Design, generate_dataset

__all__ = [
    "R_MIN",
    "PAIR_R_MAX",
    "COVERAGE_RANGE",
    "RHAT_MAX",
    "ESS_MIN",
    "CONVERGED_FRACTION",
    "RecoveryResult",
    "run_recovery",
    "evaluate_criteria",
]

R_MIN: float = 0.70
PAIR_R_MAX: float = 0.60
COVERAGE_RANGE: tuple[float, float] = (0.90, 0.98)
#: Convergence thresholds for sampling estimators (reported separately).
RHAT_MAX: float = 1.05
ESS_MIN: float = 200.0
#: Fraction of subjects that must meet both thresholds for a parameter to count as converged.
CONVERGED_FRACTION: float = 0.95

_NAMES = REDUCED_PARAMETER_NAMES
_WIDTH = np.array([p.plausible_upper - p.plausible_lower for p in REDUCED_PARAMETERS])


@dataclass
class RecoveryResult:
    """Everything K7c reports."""

    channels: tuple[str, ...]
    estimator: str
    n_subjects: int
    truth: np.ndarray
    estimate: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    pair_abs_corr: np.ndarray
    composite_truth: np.ndarray
    composite_estimate: np.ndarray
    composite_lower: np.ndarray
    composite_upper: np.ndarray
    rhat_max: np.ndarray | None = None
    ess_min: np.ndarray | None = None
    rhat_all: np.ndarray | None = None
    ess_all: np.ndarray | None = None
    hessian_pd_fraction: float = 1.0
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    tercile_metrics: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    composite_metrics: dict[str, float] = field(default_factory=dict)

    def converged_fraction(self) -> np.ndarray:
        """Per parameter, the fraction of subjects with R̂ ≤ ``RHAT_MAX`` and ESS ≥ ``ESS_MIN``."""
        if self.rhat_all is None or self.ess_all is None:
            raise ValueError("no sampling diagnostics: estimator did not sample")
        ok = (self.rhat_all <= RHAT_MAX) & (self.ess_all >= ESS_MIN)
        return np.asarray(ok.mean(axis=0))

    def pair_table(self) -> dict[str, float]:
        out = {}
        for i in range(len(_NAMES)):
            for j in range(i + 1, len(_NAMES)):
                out[f"{_NAMES[i]}-{_NAMES[j]}"] = float(self.pair_abs_corr[i, j])
        return out

    def as_dict(self) -> dict[str, object]:
        return {
            "channels": list(self.channels),
            "estimator": self.estimator,
            "n_subjects": self.n_subjects,
            "metrics": self.metrics,
            "tercile_metrics": self.tercile_metrics,
            "pairwise_abs_posterior_correlation": self.pair_table(),
            "composite": self.composite_metrics,
            "convergence": None
            if self.rhat_max is None or self.ess_min is None
            else {
                "max_rhat": dict(zip(_NAMES, map(float, self.rhat_max), strict=True)),
                "min_ess": dict(zip(_NAMES, map(float, self.ess_min), strict=True)),
                "converged_fraction": dict(
                    zip(_NAMES, map(float, self.converged_fraction()), strict=True)
                ),
            },
            "hessian_pd_fraction": self.hessian_pd_fraction,
        }


def _metrics(
    truth: np.ndarray, est: np.ndarray, lo: np.ndarray, hi: np.ndarray, width: float
) -> dict[str, float]:
    n = truth.shape[0]
    if n == 0:
        return {
            "n": 0.0,
            "r": float("nan"),
            "bias": float("nan"),
            "nrmse": float("nan"),
            "coverage95": float("nan"),
        }
    if n >= 3 and np.std(truth) > 0 and np.std(est) > 0:
        r = float(np.corrcoef(truth, est)[0, 1])
    else:
        r = float("nan")
    return {
        "n": float(n),
        "r": r,
        "bias": float(np.mean(est - truth)),
        "nrmse": float(np.sqrt(np.mean((est - truth) ** 2)) / width),
        "coverage95": float(np.mean((truth >= lo) & (truth <= hi))),
    }


def run_recovery(
    n_subjects: int = 60,
    channels: Sequence[str] = CHANNELS,
    *,
    estimator: str = "laplace",
    design: Design | None = None,
    base: ReducedParams | None = None,
    rng: RNGLike = None,
    n_samples: int = 1000,
    n_chains: int = 12,
    n_warmup: int = 600,
    n_draws: int = 1000,
) -> RecoveryResult:
    """Run the recovery simulation.

    Args:
        n_subjects: Number of synthetic subjects.
        channels: Observation channels used for fitting (data are generated
            with all channels; the fit sees only these).
        estimator: ``mcmc`` | ``laplace`` | ``map`` (see :mod:`estimation.fitting`).
            Use ``mcmc`` for anything that feeds the gate table: the Laplace
            approximation is local and under-covers along the ``α``–``Π``–``β``
            ridge, so it is a screening tool and a CI smoke test only.
        design: Task design.
        base: Noise levels (``sigma_r``, ``sigma_y``) — "realistic noise".
        rng: Master seed; each subject gets an independent spawned stream.
        n_samples: Posterior draws per subject (``laplace`` only).
        n_chains: Chains per subject (``mcmc`` only; at least 4).
        n_warmup: Warm-up iterations per chain (``mcmc`` only).
        n_draws: Retained iterations per chain (``mcmc`` only).
    """
    ch = check_channels(channels)
    base = base or ReducedParams()
    design = design or Design()
    gen = resolve_rng(rng)
    truth_pts = sample_plausible_points(n_subjects, gen)
    streams = spawn_rng(gen, n_subjects)
    d = len(_NAMES)
    est = np.empty((n_subjects, d))
    lo = np.empty((n_subjects, d))
    hi = np.empty((n_subjects, d))
    pair = np.zeros((d, d))
    comp_t = np.empty(n_subjects)
    comp_e = np.empty(n_subjects)
    comp_lo = np.empty(n_subjects)
    comp_hi = np.empty(n_subjects)
    rhat_all = np.empty((n_subjects, d)) if estimator == "mcmc" else None
    ess_all = np.empty((n_subjects, d)) if estimator == "mcmc" else None
    pd_count = 0
    ia, ip = _NAMES.index("alpha"), _NAMES.index("pi")

    for s in range(n_subjects):
        truth = base.with_estimable(truth_pts[s])
        ds = generate_dataset(truth, design, streams[s])
        data = Observed(ds.stimulus, ds.B, ds.report, ds.amplitude)
        extra: dict[str, int] = {}
        if estimator == "laplace":
            extra = {"n_samples": n_samples}
        elif estimator == "mcmc":
            extra = {"n_chains": n_chains, "n_warmup": n_warmup, "n_draws": n_draws}
        post = fit(base, data, ch, estimator=estimator, rng=streams[s], **extra)
        est[s] = post.mean
        iv = post.interval(0.95)
        lo[s], hi[s] = iv[:, 0], iv[:, 1]
        pair += np.abs(post.correlation())
        c_samples = post.samples[:, ia] * post.samples[:, ip]
        comp_t[s] = truth_pts[s, ia] * truth_pts[s, ip]
        comp_e[s] = c_samples.mean()
        comp_lo[s], comp_hi[s] = np.quantile(c_samples, [0.025, 0.975])
        pd_count += int(post.hessian_pd)
        if (
            rhat_all is not None
            and ess_all is not None
            and post.rhat is not None
            and post.ess is not None
        ):
            rhat_all[s], ess_all[s] = post.rhat, post.ess

    result = RecoveryResult(
        channels=ch,
        estimator=estimator,
        n_subjects=n_subjects,
        truth=truth_pts,
        estimate=est,
        lower=lo,
        upper=hi,
        pair_abs_corr=pair / n_subjects,
        composite_truth=comp_t,
        composite_estimate=comp_e,
        composite_lower=comp_lo,
        composite_upper=comp_hi,
        rhat_max=None if rhat_all is None else rhat_all.max(axis=0),
        ess_min=None if ess_all is None else ess_all.min(axis=0),
        rhat_all=rhat_all,
        ess_all=ess_all,
        hessian_pd_fraction=pd_count / n_subjects,
    )
    for j, name in enumerate(_NAMES):
        result.metrics[name] = _metrics(truth_pts[:, j], est[:, j], lo[:, j], hi[:, j], _WIDTH[j])
        edges = np.quantile(truth_pts[:, j], [1 / 3, 2 / 3])
        bins = np.digitize(truth_pts[:, j], edges)
        result.tercile_metrics[name] = {}
        for b, label in enumerate(("low", "mid", "high")):
            m = bins == b
            result.tercile_metrics[name][label] = _metrics(
                truth_pts[m, j], est[m, j], lo[m, j], hi[m, j], _WIDTH[j]
            )
    comp_width = float(np.ptp(comp_t)) or 1.0
    result.composite_metrics = _metrics(comp_t, comp_e, comp_lo, comp_hi, comp_width)
    return result


def evaluate_criteria(result: RecoveryResult) -> dict[str, dict[str, object]]:
    """Apply the K7c acceptance criteria per parameter.

    Returns:
        ``{parameter: {"r_ok", "coverage_ok", "pairs_ok", "converged", "passes", ...}}``.
        ``converged`` is ``None`` for non-sampling estimators. ``passes`` is the
        conjunction of the recovery, coverage and pairwise criteria (convergence
        is reported separately and, when available, also required).
    """
    out: dict[str, dict[str, object]] = {}
    for j, name in enumerate(_NAMES):
        m = result.metrics[name]
        worst_pair = max(
            (float(result.pair_abs_corr[j, k]) for k in range(len(_NAMES)) if k != j), default=0.0
        )
        r_ok = bool(np.isfinite(m["r"]) and m["r"] >= R_MIN)
        cov_ok = COVERAGE_RANGE[0] <= m["coverage95"] <= COVERAGE_RANGE[1]
        pairs_ok = worst_pair < PAIR_R_MAX
        converged: bool | None = None
        if result.rhat_all is not None and result.ess_all is not None:
            converged = bool(result.converged_fraction()[j] >= CONVERGED_FRACTION)
        out[name] = {
            "r": m["r"],
            "bias": m["bias"],
            "nrmse": m["nrmse"],
            "coverage95": m["coverage95"],
            "worst_pairwise_abs_corr": worst_pair,
            "r_ok": r_ok,
            "coverage_ok": cov_ok,
            "pairs_ok": pairs_ok,
            "converged": converged,
            "passes": bool(r_ok and cov_ok and pairs_ok and converged is not False),
        }
    return out
