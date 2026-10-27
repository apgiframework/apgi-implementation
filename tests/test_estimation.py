"""Estimation package: generator, likelihood, fitting, K7a, profile, K7c, composite, gate table."""

from __future__ import annotations

import json

import numpy as np
import pytest

import estimation.gate_table as gt
from estimation import (
    COMPOSITE_NAME,
    AnalysisSetup,
    Design,
    FactorsNotSeparableError,
    Observed,
    build_gate_table,
    check_scoring_regression,
    design_separates_factors,
    evaluate_criteria,
    expose_factors,
    fit,
    fit_composite,
    fit_laplace,
    fit_map,
    fit_mcmc,
    fitting,
    generate_dataset,
    likelihood,
    log_likelihood,
    profile_all,
    profile_likelihood,
    profile_parameter,
    reversal_stimulus,
    run_k7a,
    run_recovery,
    write_gate_table,
)
from estimation import composite as comp
from estimation import recovery_simulation as rec
from estimation import sensitivity as sens
from estimation import structural_identifiability as sid
from estimation.pymc_models import pymc_available, require_pymc
from estimation.synthetic_generator import RegressionError, generating_composites
from models.parameters import REDUCED_PARAMETER_NAMES
from models.reduced import ReducedParams, simulate_reduced

BASE = ReducedParams()
FULL = ("behavior", "ignition", "amplitude")


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset(BASE, rng=3)


@pytest.fixture(scope="module")
def observed(dataset):
    return Observed(dataset.stimulus, dataset.B, dataset.report, dataset.amplitude)


class TestSyntheticGenerator:
    def test_ode_generator_agrees_with_scorer(self):
        out = check_scoring_regression(BASE, atol=1e-6)
        assert max(out.values()) < 1e-6

    @pytest.mark.parametrize(
        "params",
        [{"alpha": 0.9, "pi": 5.0}, {"alpha": 0.05, "pi": 0.3, "beta": -0.3}, {"theta_base": 0.25}],
    )
    def test_agreement_across_the_plausible_range(self, params):
        check_scoring_regression(ReducedParams(**params), atol=1e-6, seed=4)

    def test_regression_check_fails_when_scorer_drifts(self, monkeypatch):
        import estimation.synthetic_generator as g

        real = g.simulate_reduced

        def drifted(params, stimulus, **kw):
            tr = real(params, stimulus, **kw)
            tr.xhat = tr.xhat * 1.001
            return tr

        monkeypatch.setattr(g, "simulate_reduced", drifted)
        with pytest.raises(RegressionError, match="does not reproduce"):
            check_scoring_regression(BASE)

    def test_integration_error_shrinks_with_substeps(self):
        coarse = check_scoring_regression(BASE, n_substeps=2, atol=1.0)["max_abs_S"]
        fine = check_scoring_regression(BASE, n_substeps=40, atol=1.0)["max_abs_S"]
        assert fine < coarse

    def test_composites(self):
        c = generating_composites(ReducedParams(alpha=0.2, pi=2.0))
        assert c["precision_weighted_learning_composite"] == pytest.approx(0.4)
        assert c["learning_gain"] == pytest.approx(0.4 / 1.4)

    def test_design_and_stimulus(self):
        d = Design(n_trials=60, block=10, amplitude=1.0, sigma_x=0.0)
        x = reversal_stimulus(d, 0)
        assert x[:10].tolist() == [1.0] * 10 and x[10:20].tolist() == [-1.0] * 10
        for bad in ({"n_trials": 1}, {"block": 0}, {"sigma_x": -1}, {"amplitude": 0}):
            with pytest.raises(ValueError):
                Design(**bad)

    def test_fixed_stimulus_and_dataset_shapes(self, dataset):
        assert dataset.n_trials == 240
        assert dataset.B.shape == dataset.report.shape == dataset.amplitude.shape == (240,)
        ds = generate_dataset(BASE, rng=1, stimulus=np.zeros(20))
        assert ds.n_trials == 20


class TestLikelihood:
    def test_channels_are_additive(self, observed):
        parts = [log_likelihood(BASE, observed, [c]) for c in FULL]
        assert log_likelihood(BASE, observed, FULL) == pytest.approx(sum(parts), abs=1e-9)

    def test_truth_beats_perturbed_parameters(self, observed):
        ll = log_likelihood(BASE, observed, FULL)
        for name, delta in (("alpha", 0.3), ("pi", 2.0), ("theta_base", 0.25), ("beta", 0.2)):
            from dataclasses import replace

            assert ll > log_likelihood(
                replace(BASE, **{name: getattr(BASE, name) + delta}), observed, FULL
            )

    def test_matches_independent_formula(self, dataset, observed):
        tr = simulate_reduced(BASE, dataset.stimulus, B_forced=dataset.B)
        from scipy import stats

        manual = (
            stats.norm.logpdf(dataset.report, tr.xhat, BASE.sigma_r).sum()
            + stats.norm.logpdf(dataset.amplitude, tr.S, BASE.sigma_y).sum()
            + (dataset.B * np.log(tr.p) + (1 - dataset.B) * np.log1p(-tr.p)).sum()
        )
        assert log_likelihood(BASE, observed, FULL) == pytest.approx(manual, abs=1e-8)

    def test_missing_channel_data_and_bad_channels(self, dataset):
        only_b = Observed(dataset.stimulus, dataset.B)
        with pytest.raises(ValueError, match="report"):
            log_likelihood(BASE, only_b, ["behavior"])
        with pytest.raises(ValueError, match="amplitude"):
            log_likelihood(BASE, only_b, ["amplitude"])
        assert np.isfinite(log_likelihood(BASE, only_b, ["ignition"]))
        with pytest.raises(ValueError, match="unknown channels"):
            likelihood.check_channels(["nope"])
        with pytest.raises(ValueError, match="at least one"):
            likelihood.check_channels([])
        assert likelihood.check_channels(["amplitude", "behavior"]) == ("behavior", "amplitude")

    def test_observed_validates_lengths(self):
        with pytest.raises(ValueError, match="report"):
            Observed(np.zeros(5), np.zeros(5, dtype=int), report=np.zeros(4))

    def test_transform_roundtrip_and_prior(self):
        p = BASE.estimable()
        np.testing.assert_allclose(
            likelihood.to_constrained(likelihood.to_unconstrained(p)), p, atol=1e-6
        )
        lo, hi = likelihood.BOUNDS[:, 0], likelihood.BOUNDS[:, 1]
        c = likelihood.to_constrained(np.array([-50.0, 50.0, 0.0, 0.0]))
        assert np.all(c >= lo) and np.all(c <= hi)
        assert likelihood.log_prior_u(likelihood.PRIOR_MEAN_U) == 0.0
        assert likelihood.log_prior_u(likelihood.PRIOR_MEAN_U + 1.5) < 0


class TestFitting:
    def test_map_recovers_with_all_channels(self, observed):
        u, lp = fit_map(BASE, observed, FULL, rng=1)
        est = likelihood.to_constrained(u)
        assert np.all(np.abs(est - BASE.estimable()) <= [0.15, 0.6, 0.12, 0.12])
        assert np.isfinite(lp)

    def test_laplace_is_seeded_and_brackets_its_own_map(self, observed):
        post = fit_laplace(BASE, observed, FULL, rng=1, n_samples=800)
        iv = post.interval()
        assert np.all((post.map_estimate >= iv[:, 0]) & (post.map_estimate <= iv[:, 1]))
        assert post.hessian_pd and post.samples.shape == (800, 4)
        again = fit_laplace(BASE, observed, FULL, rng=1, n_samples=800)
        assert np.array_equal(post.samples, again.samples)

    def test_behaviour_only_leaves_alpha_pi_ridge_wide(self, observed):
        post = fit_laplace(BASE, observed, ["behavior"], rng=1, n_samples=800)
        w = post.interval()[:, 1] - post.interval()[:, 0]
        full = fit_laplace(BASE, observed, FULL, rng=1, n_samples=800)
        wf = full.interval()[:, 1] - full.interval()[:, 0]
        assert np.all(w[:2] > 5 * wf[:2])  # α and Π are unconstrained by behaviour alone
        assert abs(post.correlation()[0, 1]) > 0.5

    def test_mcmc_converges_and_agrees_with_laplace(self, observed):
        m = fit_mcmc(BASE, observed, FULL, n_chains=8, n_warmup=400, n_draws=600, rng=2)
        assert np.all(m.rhat < 1.1) and np.all(m.ess > 100)
        la = fit_laplace(BASE, observed, FULL, rng=2, n_samples=2000)
        assert np.all(np.abs(m.mean - la.mean) <= [0.1, 0.4, 0.1, 0.08])
        assert 0.05 < m.accept_rate < 0.8
        assert set(m.as_dict()) >= {"rhat", "ess", "map", "mean"}

    def test_mcmc_needs_enough_chains(self, observed):
        with pytest.raises(ValueError, match="n_chains"):
            fit_mcmc(BASE, observed, FULL, n_chains=3)

    def test_dispatch(self, observed):
        assert fit(BASE, observed, FULL, estimator="map", rng=1).samples.shape == (1, 4)
        assert (
            fit(BASE, observed, FULL, estimator="laplace", rng=1, n_samples=50).estimator
            == "laplace"
        )
        with pytest.raises(ValueError, match="unknown estimator"):
            fit(BASE, observed, FULL, estimator="gibbs")

    def test_diagnostics_on_known_distributions(self):
        rng = np.random.default_rng(0)
        iid = rng.normal(size=(4, 2000, 1))
        assert fitting.split_rhat(iid)[0] == pytest.approx(1.0, abs=0.01)
        assert fitting.effective_sample_size(iid)[0] > 6000
        shifted = iid.copy()
        shifted[0] += 3.0
        assert fitting.split_rhat(shifted)[0] > 1.5
        ar = np.zeros((4, 4000, 1))
        for c in range(4):
            e = rng.normal(size=4000)
            for t in range(1, 4000):
                ar[c, t, 0] = 0.95 * ar[c, t - 1, 0] + e[t]
        assert fitting.effective_sample_size(ar)[0] < 1500
        const = np.ones((2, 10, 1))
        assert fitting.effective_sample_size(const)[0] == 20
        with pytest.raises(ValueError):
            fitting.split_rhat(np.zeros((2, 3, 1)))

    def test_numerical_hessian_of_quadratic(self):
        A = np.array([[2.0, 0.5], [0.5, 1.0]])
        H = fitting.numerical_hessian(
            lambda x: float(0.5 * x @ A @ x), np.array([0.3, -0.2]), step=1e-3
        )
        np.testing.assert_allclose(H, A, atol=1e-5)


class TestStructuralIdentifiability:
    def test_product_symmetry_is_exact_for_behaviour_and_broken_by_other_channels(self):
        s = AnalysisSetup()
        assert sid.product_symmetry_gap(s, ("behavior",)) < 1e-12
        assert sid.product_symmetry_gap(s, ("ignition",)) > 1e-3
        assert sid.product_symmetry_gap(s, ("amplitude",)) > 1e-3

    def test_rank_of_behaviour_is_one_and_full_design_is_four(self):
        s = AnalysisSetup()
        x = s.stimulus()
        a = sid.analyze_point(BASE.estimable(), x, ("behavior",), BASE)
        assert a.rank == 1 and a.null_vectors.shape[1] == 3
        full = sid.analyze_point(BASE.estimable(), x, FULL, BASE)
        assert full.rank == 4 and full.condition < sid.COND_LIMIT

    def test_null_space_names_the_alpha_pi_trade_off(self):
        s = AnalysisSetup()
        a = sid.analyze_point(BASE.estimable(), s.stimulus(), ("behavior",), BASE)
        mask = a.nonidentifiable_mask()
        assert mask.all()  # every estimable parameter is in some null direction
        v = a.null_vectors
        # the (alpha, pi) direction keeps alpha*pi constant to first order
        ia, ip = 0, 1
        combo = (
            v[ia] * BASE.pi + v[ip] * BASE.alpha
        )  # d(alpha*pi) along each null vector, in normalised units
        assert np.sum(np.abs(combo) < 1e-6) >= 1

    def test_amplitude_only_cannot_see_threshold(self):
        s = AnalysisSetup()
        a = sid.analyze_point(BASE.estimable(), s.stimulus(), ("amplitude",), BASE)
        assert a.rank == 3
        assert a.nonidentifiable_mask()[REDUCED_PARAMETER_NAMES.index("theta_base")]

    def test_partial_collinearity(self):
        J = np.array([[1.0, 2.0, 0.0], [2.0, 4.0, 1.0], [3.0, 6.0, 0.0], [0.0, 0.0, 1.0]])
        assert sid.partial_collinearity(J, 0, 1) == 1.0  # exactly proportional columns
        rng = np.random.default_rng(0)
        assert sid.partial_collinearity(rng.normal(size=(50, 4)), 0, 1) < 0.5

    def test_sweep_and_pair_tests_by_design(self):
        res_b, an_b = sid.sweep_identifiability(("behavior",), n_points=12, seed=1)
        assert res_b.full_rank_fraction == 0.0 and res_b.rank_counts == {1: 12}
        assert sid.test_pair(an_b, "alpha", "pi")["classification"] == "degenerate"
        res_f, an_f = sid.sweep_identifiability(FULL, n_points=12, seed=1)
        assert res_f.full_rank_fraction == 1.0 and res_f.median_condition < sid.COND_LIMIT
        assert sid.test_pair(an_f, "theta_base", "beta")["classification"] == "identifiable"
        assert set(res_f.as_dict()) >= {"identifiable_fraction", "rank_counts", "p95_condition"}

    def test_multi_start_finds_equivalent_solutions_only_when_degenerate(self):
        beh = sid.multi_start_equivalents(("behavior",), n_starts=6, seed=1)
        assert beh["n_equivalent_solutions"] > 0 and beh["pair_trade_off"]
        full = sid.multi_start_equivalents(FULL, n_starts=6, seed=1)
        assert full["n_equivalent_solutions"] == 0 and not full["subset_trade_off"]
        assert full["weakest_direction"]["singular_value"] > 1.0

    def test_run_k7a_classification_matches_the_science(self):
        beh = run_k7a(("behavior",), n_points=12, n_starts=4)
        assert set(beh["classification"].values()) == {"FALSIFIED"}
        assert beh["composite"]["status"] == "CONFIRMED"  # αΠ is identifiable from behaviour
        full = run_k7a(FULL, n_points=40, n_starts=4)
        assert full["classification"]["theta_base"] == "CONFIRMED"
        assert full["classification"]["beta"] == "CONFIRMED"
        assert (
            full["classification"]["alpha"] == "FALSIFIED"
            and full["classification"]["pi"] == "FALSIFIED"
        )
        json.dumps(full)

    def test_latin_hypercube_stays_in_plausible_box(self):
        pts = sid.sample_plausible_points(50, 0)
        from models.parameters import REDUCED_PARAMETERS

        for j, spec in enumerate(REDUCED_PARAMETERS):
            assert (
                pts[:, j].min() >= spec.plausible_lower and pts[:, j].max() <= spec.plausible_upper
            )
        assert np.array_equal(pts, sid.sample_plausible_points(50, 0))

    def test_suspect_pairs_are_the_registered_ones(self):
        assert sid.SUSPECT_PAIRS == (("alpha", "pi"), ("theta_base", "beta"))


class TestProfileLikelihood:
    def test_full_design_identifies_every_parameter(self, observed):
        prof = profile_all(BASE, observed, FULL, n_grid=13)
        for name, r in prof.items():
            assert r.status == "identifiable", name
            truth = getattr(BASE, name)
            assert r.ci_lower <= truth <= r.ci_upper or abs(r.mle - truth) < 0.3
            assert r.deviance.min() == 0.0
        json.dumps(prof["alpha"].as_dict())

    def test_behaviour_only_leaves_open_ci_sides(self, observed):
        prof = profile_all(BASE, observed, ["behavior"], n_grid=13)
        assert prof["theta_base"].status == "non-identifiable"
        assert prof["beta"].status == "non-identifiable"
        assert prof["alpha"].status in ("one-sided", "non-identifiable")

    def test_single_parameter_with_precomputed_mle(self, observed):
        mle = profile_likelihood.fit_mle(BASE, observed, FULL)
        r = profile_parameter(BASE, observed, FULL, "theta_base", n_grid=9, mle=mle)
        assert r.mle == pytest.approx(mle[0][2])
        assert pytest.approx(3.8415, abs=1e-3) == profile_likelihood.CHI2_95_1DF


class TestRecovery:
    def test_smoke_full_design(self):
        r = run_recovery(16, FULL, rng=7)
        assert r.truth.shape == (16, 4) and r.pair_abs_corr.shape == (4, 4)
        assert r.metrics["theta_base"]["r"] > 0.6
        assert r.composite_metrics["r"] > 0.95
        out = r.as_dict()
        json.dumps(out)
        assert set(out["tercile_metrics"]["alpha"]) == {"low", "mid", "high"}
        assert sum(out["tercile_metrics"]["alpha"][t]["n"] for t in ("low", "mid", "high")) == 16

    def test_behaviour_only_fails_the_criteria_for_theta_and_beta(self):
        r = run_recovery(16, ["behavior"], rng=7)
        ev = evaluate_criteria(r)
        assert not ev["theta_base"]["passes"] and not ev["beta"]["passes"]
        assert r.composite_metrics["r"] > 0.95  # but the composite is recovered

    def test_pairwise_criterion_flags_alpha_pi(self):
        ev = evaluate_criteria(run_recovery(16, FULL, rng=7))
        assert ev["alpha"]["pairs_ok"] is False and ev["pi"]["pairs_ok"] is False
        assert ev["theta_base"]["pairs_ok"] is True

    def test_thresholds_are_the_registered_ones(self):
        assert rec.R_MIN == 0.70 and rec.PAIR_R_MAX == 0.60 and rec.COVERAGE_RANGE == (0.90, 0.98)

    def test_mcmc_reports_convergence(self):
        r = run_recovery(2, FULL, estimator="mcmc", rng=1, n_chains=6, n_warmup=100, n_draws=150)
        assert r.rhat_all is not None and r.converged_fraction().shape == (4,)
        assert "convergence" in r.as_dict() and r.as_dict()["convergence"] is not None
        lap = run_recovery(2, FULL, rng=1)
        with pytest.raises(ValueError, match="no sampling diagnostics"):
            lap.converged_fraction()

    def test_metrics_helper_edge_cases(self):
        m = rec._metrics(
            np.array([1.0, 1.0]),
            np.array([1.0, 2.0]),
            np.array([0.0, 0.0]),
            np.array([2.0, 2.0]),
            1.0,
        )
        assert np.isnan(m["r"]) and m["coverage95"] == 1.0

    def test_seeded(self):
        a = run_recovery(4, FULL, rng=3)
        b = run_recovery(4, FULL, rng=3)
        assert np.array_equal(a.estimate, b.estimate)


class TestComposite:
    def test_fit_recovers_product_and_interval_covers(self):
        covered = 0
        for seed in range(12):
            p = ReducedParams(alpha=0.25, pi=2.0)
            ds = generate_dataset(p, rng=seed)
            est = fit_composite(ds.stimulus, ds.report)
            covered += est.ci95[0] <= 0.5 <= est.ci95[1]
            assert est.value == pytest.approx(0.5, abs=0.15)
        assert covered >= 9

    def test_output_exposes_only_the_product(self):
        ds = generate_dataset(BASE, rng=1)
        est = fit_composite(ds.stimulus, ds.report)
        assert not hasattr(est, "alpha") and not hasattr(est, "pi")
        d = est.as_dict()
        assert COMPOSITE_NAME in d and d["factors_exposed"] is False
        assert "alpha" not in d and "pi" not in d

    def test_input_validation(self):
        with pytest.raises(ValueError):
            fit_composite(np.zeros(5), np.zeros(4))
        with pytest.raises(ValueError):
            fit_composite(np.zeros(5), np.zeros(5), sigma_r=0)

    def test_posterior_product(self):
        e = comp.composite_from_samples(np.full(100, 0.2), np.full(100, 3.0), n_trials=10)
        assert e.value == pytest.approx(0.6) and e.method == "posterior-product"
        with pytest.raises(ValueError):
            comp.composite_from_samples(np.zeros(3), np.zeros(4))

    def test_factors_refused_unless_design_separates_them(self):
        a, p = np.ones(5), np.ones(5)
        with pytest.raises(FactorsNotSeparableError, match=COMPOSITE_NAME):
            expose_factors(a, p, ("behavior",), separable=False)
        assert set(expose_factors(a, p, FULL, separable=True)) == {"alpha", "pi"}

    def test_design_verdict_comes_from_k7a(self):
        assert design_separates_factors(("behavior",), n_points=10) is False
        with pytest.raises(FactorsNotSeparableError):
            expose_factors(np.ones(2), np.ones(2), ("behavior",))


def _fake_k7a(**over):
    base = {
        "channels": list(FULL),
        "classification": {
            "alpha": "FALSIFIED",
            "pi": "FALSIFIED",
            "theta_base": "CONFIRMED",
            "beta": "CONFIRMED",
        },
        "composite": {"status": "CONFIRMED"},
    }
    base.update(over)
    return base


class TestGateTable:
    @pytest.fixture(scope="module")
    def result(self):
        return run_recovery(24, FULL, rng=11)

    def test_table_structure_and_allow_list(self, result):
        t = build_gate_table(_fake_k7a(), result, meta={"seed": 11})
        rows = {r["parameter"]: r for r in t["parameters"]}
        assert set(rows) == set(REDUCED_PARAMETER_NAMES) | {COMPOSITE_NAME}
        for r in rows.values():
            assert r["k7b"] == "PENDING_DATA"
            assert {r["k7a"], r["k7c"]} <= set(gt.STATUSES)
            assert r["exposable"] == (r["k7a"] == "CONFIRMED" and r["k7c"] == "CONFIRMED")
        assert t["allow_list"] == gt.allow_list(t) == [n for n, r in rows.items() if r["exposable"]]
        assert rows["alpha"]["exposable"] is False  # K7a falsified ⇒ never exposable
        assert "K7b" in t["k7b_note"]

    def test_k7a_falsified_blocks_exposure_even_with_good_recovery(self, result):
        t = build_gate_table(
            _fake_k7a(classification=dict.fromkeys(REDUCED_PARAMETER_NAMES, "FALSIFIED")), result
        )
        assert all(
            not r["exposable"] for r in t["parameters"] if r["parameter"] in REDUCED_PARAMETER_NAMES
        )

    def test_yaml_is_valid_and_equivalent_to_json(self, result, tmp_path):
        yaml = pytest.importorskip("yaml")
        t = build_gate_table(_fake_k7a(), result)
        path = write_gate_table(t, tmp_path / "gates" / "gate-table.yaml")
        assert yaml.safe_load(path.read_text(encoding="utf-8")) == json.loads(json.dumps(t))
        assert json.loads(path.with_suffix(".json").read_text()) == json.loads(json.dumps(t))

    def test_yaml_emitter_edge_cases(self):
        out = gt.dump_yaml(
            {
                "a": [],
                "b": {},
                "c": [{"x": 1, "y": [1, 2]}, {"x": None}],
                "d": 'q"uote',
                "e": True,
                "f": 1.5,
                "9k": 1,
            }
        )
        assert "a: []" in out and "b: {}" in out and 'd: "q\\"uote"' in out and '"9k": 1' in out
        yaml = pytest.importorskip("yaml")
        assert yaml.safe_load(out) == {
            "a": [],
            "b": {},
            "c": [{"x": 1, "y": [1, 2]}, {"x": None}],
            "d": 'q"uote',
            "e": True,
            "f": 1.5,
            "9k": 1,
        }

    def test_classification_rules(self, result):
        cls = gt.classify_k7c(result)
        assert cls[COMPOSITE_NAME]["status"] in gt.STATUSES
        # non-convergence → INDETERMINATE regardless of everything else
        import copy

        r2 = copy.copy(result)
        r2.rhat_all = np.full((result.n_subjects, 4), 2.0)
        r2.ess_all = np.full((result.n_subjects, 4), 5.0)
        assert all(
            v["status"] == "INDETERMINATE"
            for k, v in gt.classify_k7c(r2).items()
            if k != COMPOSITE_NAME
        )

    def test_coverage_marginality_uses_binomial_error(self):
        assert gt._coverage_marginal(0.85, 12) is True  # 12 subjects cannot resolve 0.85 vs 0.90
        assert gt._coverage_marginal(0.60, 500) is False  # clearly miscalibrated
        assert gt._coverage_marginal(0.95, 60) is False  # inside the window: not "marginal"
        lo, hi = gt._wilson(0.5, 100)
        assert lo < 0.5 < hi and gt._wilson(0.5, 0) == (0.0, 1.0)


class TestSensitivity:
    X = reversal_stimulus(Design(), 0)

    def test_elasticities_have_expected_signs(self):
        el = sens.local_elasticities(BASE.estimable(), self.X)
        assert el["mean_signal"]["pi"] > 0 and el["mean_signal"]["alpha"] < 0
        assert (
            el["mean_signal"]["theta_base"] == 0.0
        )  # threshold cannot change S (no ignition feedback at nominal)

    def test_oat_sweep(self):
        s = sens.oat_sweep(self.X, n_grid=5)
        assert set(s) == set(REDUCED_PARAMETER_NAMES)
        assert np.all(np.diff(s["pi"]["grid"]) > 0) and s["pi"]["grid"][0] == pytest.approx(0.3)
        assert s["beta"]["mean_signal"][-1] > s["beta"]["mean_signal"][0]

    def test_sobol_indices_are_sane(self):
        r = sens.sobol_indices(self.X, n=96, rng=2)
        assert set(r) == set(sens.METRICS)
        late = r["late_abs_error"]["ST"]
        # The error late in the session depends on learning (α, Π) and not on the
        # threshold or the bias, which only act downstream of the prediction.
        assert late["theta_base"] == pytest.approx(0.0, abs=1e-9) and late["beta"] == pytest.approx(
            0.0, abs=1e-9
        )
        assert late["alpha"] > 0.5
        ign = r["mean_ignition_probability"]["ST"]
        assert ign["theta_base"] > 0.3 and ign["beta"] > 0.3

    def test_metric_vector_is_deterministic(self):
        a = sens.metric_vector(BASE.estimable(), self.X)
        assert a == sens.metric_vector(BASE.estimable(), self.X)


class TestPymcBoundary:
    def test_import_is_lazy_and_error_is_actionable(self):
        import sys

        assert "pymc" not in sys.modules or pymc_available()
        if not pymc_available():
            with pytest.raises(ImportError, match="apgi\\[estimation\\]"):
                require_pymc()
