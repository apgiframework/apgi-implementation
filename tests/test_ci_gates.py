"""CI gates for apgi-implementation (XC-CI-02).

One marker per required gate, so each is a separately named, separately failing CI job:

* ``parity``         — numerical parity between independent implementations;
* ``stability``      — finite, bounded behaviour at extremes and under perturbation;
* ``performance``    — throughput floors (run without coverage tracing);
* ``repeatability``  — seeded runs reproduce across processes and hash seeds;
* ``recovery_smoke`` — a small K7c recovery run completes and recovers what it should.

Performance floors are deliberately ~10x below measured throughput: they catch an
accidental quadratic or a lost vectorisation, not a noisy runner.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest

from config import CONFIG
from engine import PrecisionWeightedErrorChannel, filter_latent_signal
from estimation import (
    Observed,
    check_scoring_regression,
    fit_laplace,
    generate_dataset,
    log_likelihood,
    run_k7a,
    run_recovery,
)
from model_registry import create_model
from models.parameters import REDUCED_PARAMETERS
from models.reduced import ReducedParams, simulate_reduced
from pipeline import APGIPipeline
from qualification import reference as qref

ROOT = Path(__file__).resolve().parent.parent
FULL = ("behavior", "ignition", "amplitude")


# --------------------------------------------------------------------------- parity
@pytest.mark.parity
class TestNumericalParity:
    @pytest.mark.parametrize("case", list(qref.CASES))
    def test_reference_matches_independent_implementation(self, case):
        p = ReducedParams(**qref.CASES[case])
        x = qref.fixture_stimulus()
        ref = qref.independent_trajectory(p, x)
        mine = simulate_reduced(p, x, hard=True)
        for key in ("xhat", "S", "theta", "p"):
            np.testing.assert_allclose(getattr(mine, key), ref[key], rtol=0, atol=1e-12)

    def test_ode_generator_matches_scorer(self):
        for params in (ReducedParams(), ReducedParams(alpha=0.8, pi=4.0, beta=-0.2)):
            assert max(check_scoring_regression(params, atol=1e-6).values()) < 1e-6

    def test_reference_variant_matches_the_pipeline_it_wraps(self):
        x = np.sin(0.1 * np.arange(80))
        wrapped = create_model("apgi_reference", seed=4).run(x, 0.3 * x)
        pipe = APGIPipeline({**CONFIG, "seed": 4})
        direct = [pipe.step(float(a), float(0.3 * a)) for a in x]
        assert np.array_equal(wrapped["S"], [r["S"] for r in direct])
        assert np.array_equal(wrapped["B"], [r["B"] for r in direct])

    def test_error_channel_matches_pipeline(self):
        cfg = {**CONFIG, "seed": 1, "use_internal_predictions": False}
        pipe = APGIPipeline(cfg)
        ch = PrecisionWeightedErrorChannel(cfg["alpha_e"], cfg["pi_min"], cfg["pi_max"], cfg["eps"])
        rng = np.random.default_rng(0)
        for x, h in zip(rng.normal(size=60), rng.normal(size=60), strict=True):
            r = pipe.step(float(x), 0.0, float(h), 0.0)
            assert ch.update(float(x), float(h)).pi == pytest.approx(r["pi_e"], abs=1e-12)

    def test_likelihood_matches_particle_filter_evidence(self):
        p = ReducedParams()
        ds = generate_dataset(p, rng=5)
        obs = Observed(ds.stimulus, ds.B, ds.report, ds.amplitude)
        f = filter_latent_signal(p, ds.stimulus, ds.B, ds.amplitude, sigma_s=0.0, n_particles=1)
        assert f.log_evidence == pytest.approx(
            log_likelihood(p, obs, ("ignition", "amplitude")), abs=1e-9
        )

    def test_fixture_is_current(self):
        assert qref.verify_record() == []


# ------------------------------------------------------------------------- stability
@pytest.mark.stability
class TestStability:
    @pytest.mark.parametrize("scale", [1e-12, 1.0, 1e3, 1e8])
    def test_extreme_stimulus_magnitudes_stay_finite_and_bounded(self, scale):
        gen = np.random.default_rng(0)
        p = ReducedParams()
        tr = simulate_reduced(p, scale * gen.normal(size=3000), rng=1)
        for arr in (tr.xhat, tr.S, tr.theta, tr.p):
            assert np.all(np.isfinite(arr))
        assert (
            tr.S.min() >= min(0.0, p.beta) - 1e-12 and tr.S.max() <= max(0.0, 1.0 + p.beta) + 1e-12
        )
        # θ is bounded by baseline + δ/(1−e^{−κ}) however often the model ignites.
        assert tr.theta.max() <= p.theta_base + 0.5 / (1 - np.exp(-0.15)) + 1e-9

    def test_every_corner_of_the_support_is_stable(self):
        lo = [s.lower for s in REDUCED_PARAMETERS]
        hi = [s.upper for s in REDUCED_PARAMETERS]
        x = np.tile([0.8] * 20 + [-0.8] * 20, 10)
        for mask in range(16):
            theta = [hi[j] if mask >> j & 1 else lo[j] for j in range(4)]
            p = ReducedParams().with_estimable(theta)
            tr = simulate_reduced(p, x, rng=mask)
            assert np.all(np.isfinite(tr.S)) and np.all(np.isfinite(tr.xhat)), theta

    def test_long_run_has_no_drift_or_blow_up(self):
        out = create_model("apgi_reduced", seed=2).run(np.sin(0.01 * np.arange(50_000)))
        assert (
            np.all(np.isfinite(out["S"]))
            and abs(out["S"][-10_000:].mean() - out["S"][:10_000].mean()) < 0.5
        )

    def test_output_is_continuous_in_parameters(self):
        x = np.tile([0.8] * 20 + [-0.8] * 20, 6)
        forced = simulate_reduced(ReducedParams(), x, rng=3).B
        base = simulate_reduced(ReducedParams(), x, B_forced=forced)
        for name in ("alpha", "pi", "theta_base", "beta"):
            bumped = ReducedParams(**{name: getattr(ReducedParams(), name) * (1 + 1e-9) + 1e-12})
            tr = simulate_reduced(bumped, x, B_forced=forced)
            assert (
                np.max(np.abs(tr.S - base.S)) < 1e-6 and np.max(np.abs(tr.xhat - base.xhat)) < 1e-6
            )

    def test_perturbed_initial_state_decays(self):
        from models.reduced import ReducedState, reduced_step

        p = ReducedParams(theta_base=0.85)
        a, b = ReducedState(theta=0.85), ReducedState(xhat=0.5, S=0.4, theta=0.85)
        for _ in range(300):
            a, _ = reduced_step(p, a, 0.2, hard=True)
            b, _ = reduced_step(p, b, 0.2, hard=True)
        assert abs(a.xhat - b.xhat) < 1e-9 and abs(a.S - b.S) < 1e-9

    @pytest.mark.parametrize("kind", ["no_ignition", "all_ignition", "constant"])
    def test_estimation_survives_degenerate_data(self, kind):
        n = 120
        x = np.zeros(n) if kind == "constant" else np.tile([0.8] * 30 + [-0.8] * 30, 2)
        B = np.ones(n, dtype=int) if kind == "all_ignition" else np.zeros(n, dtype=int)
        data = Observed(x, B, np.zeros(n), np.zeros(n))
        post = fit_laplace(ReducedParams(), data, rng=0, n_samples=100)
        assert np.all(np.isfinite(post.samples))
        assert np.all(post.samples >= [s.lower for s in REDUCED_PARAMETERS])
        assert np.all(post.samples <= [s.upper for s in REDUCED_PARAMETERS])


# ---------------------------------------------------------------------- performance
def _best_of(fn, repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


@pytest.mark.performance
class TestPerformance:
    def test_reference_model_throughput(self):
        x = np.sin(0.01 * np.arange(20_000))
        seconds = _best_of(lambda: simulate_reduced(ReducedParams(), x, rng=1))
        assert 20_000 / seconds > 50_000, f"{20_000 / seconds:.0f} steps/s"

    def test_likelihood_evaluation_time(self):
        ds = generate_dataset(ReducedParams(), rng=1)
        obs = Observed(ds.stimulus, ds.B, ds.report, ds.amplitude)
        seconds = _best_of(lambda: [log_likelihood(ReducedParams(), obs) for _ in range(50)]) / 50
        assert seconds < 2e-3, f"{seconds * 1e3:.2f} ms per evaluation"

    def test_laplace_fit_time(self):
        ds = generate_dataset(ReducedParams(), rng=1)
        obs = Observed(ds.stimulus, ds.B, ds.report, ds.amplitude)
        assert _best_of(lambda: fit_laplace(ReducedParams(), obs, rng=0, n_samples=200), 2) < 10.0

    def test_ode_generator_time(self):
        assert _best_of(lambda: generate_dataset(ReducedParams(), rng=1)) < 2.0

    def test_pipeline_variant_throughput(self):
        x = np.sin(0.05 * np.arange(500))
        seconds = _best_of(lambda: create_model("apgi_reference", seed=1).run(x), 2)
        assert 500 / seconds > 1_500, f"{500 / seconds:.0f} steps/s"


# ------------------------------------------------------------------- repeatability
_CHILD = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, {root!r})
    import numpy as np
    from engine import record_run
    from estimation import Observed, fit_laplace, generate_dataset
    from models.reduced import ReducedParams
    x = np.sin(0.07 * np.arange(150))
    rec, _ = record_run("apgi_reduced", {{"alpha": 0.4}}, 11, x)
    ds = generate_dataset(ReducedParams(), rng=4)
    post = fit_laplace(ReducedParams(), Observed(ds.stimulus, ds.B, ds.report, ds.amplitude), rng=2, n_samples=100)
    print(rec.output_digest, post.samples.sum().hex())
    """
)


@pytest.mark.repeatability
@pytest.mark.parametrize("hashseed", ["0", "12345"])
def test_seeded_results_identical_across_processes_and_hash_seeds(hashseed):
    def run(seed: str) -> str:
        env = {**os.environ, "PYTHONHASHSEED": seed}
        out = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-c", _CHILD.format(root=str(ROOT))],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(ROOT),
            check=True,
        )
        return out.stdout.strip().splitlines()[-1]

    assert run(hashseed) == run("999")


# ----------------------------------------------------------------------- recovery
@pytest.mark.recovery_smoke
class TestRecoverySmoke:
    def test_full_design_recovers_composite_and_threshold(self):
        t = time.perf_counter()
        r = run_recovery(14, FULL, rng=2)
        assert time.perf_counter() - t < 60
        assert r.composite_metrics["r"] > 0.95
        assert r.metrics["theta_base"]["r"] > 0.6 and r.metrics["beta"]["r"] > 0.6

    def test_behaviour_only_recovers_the_composite_but_not_threshold(self):
        r = run_recovery(14, ("behavior",), rng=2)
        assert r.composite_metrics["r"] > 0.95
        assert abs(r.metrics["theta_base"]["r"]) < 0.7

    def test_k7a_smoke_agrees_with_k7c_smoke(self):
        k7a = run_k7a(FULL, n_points=12, n_starts=2)
        assert k7a["classification"]["theta_base"] == "CONFIRMED"
        assert k7a["composite"]["status"] == "CONFIRMED"
