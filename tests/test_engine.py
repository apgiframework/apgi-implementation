"""Computation engine primitives (S1-IMP-01..08)."""

from __future__ import annotations

import numpy as np
import pytest

from config import CONFIG
from engine import (
    Condition,
    PrecisionWeightedErrorChannel,
    SomaticThresholdAdapter,
    array_digest,
    baseline,
    clamp_ne_gain,
    contrast,
    dopamine_blockade,
    filter_latent_signal,
    hard_ignition,
    hard_limit_gap,
    ignition_probability,
    interoceptive_precision,
    precision_lesion,
    precision_weighted_error,
    precision_weighted_errors,
    run_conditions,
    sample_ignition,
    simulate_session,
    somatic_gain,
    stimulus_gain,
    summarise_ignitions,
    threshold_sweep,
)
from estimation import Observed, generate_dataset, log_likelihood
from models.reduced import ReducedParams, simulate_reduced
from pipeline import APGIPipeline


class TestPredictionError:
    def test_parity_with_pipeline_exteroceptive_channel(self):
        """Steps 1–5 reproduce the pipeline's z-score, μ, σ² and Π exactly."""
        cfg = {**CONFIG, "seed": 1, "use_internal_predictions": False}
        pipe = APGIPipeline(cfg)
        ch = PrecisionWeightedErrorChannel(
            alpha=cfg["alpha_e"], pi_min=cfg["pi_min"], pi_max=cfg["pi_max"], eps=cfg["eps"]
        )
        rng = np.random.default_rng(0)
        xs, hats = rng.normal(0, 1, 80), rng.normal(0, 0.5, 80)
        for x, h in zip(xs, hats, strict=True):
            r = pipe.step(float(x), 0.0, float(h), 0.0)
            mine = ch.update(float(x), float(h))
            assert mine.eps == pytest.approx(r["z_e"], abs=1e-15)
            assert mine.z_norm == pytest.approx(r["z_e_norm"], abs=1e-12)
            assert mine.mu == pytest.approx(r["mu_e"], abs=1e-12)
            assert mine.pi == pytest.approx(r["pi_e"], abs=1e-12)

    def test_precision_clamped_and_weighting(self):
        ch = PrecisionWeightedErrorChannel(alpha=1.0, sigma2_0=1e-9, pi_max=10.0)
        assert ch.pi == 10.0
        r = ch.update(0.0, 0.0)
        assert r.pi == 10.0 and r.weighted == pytest.approx(10.0 * r.phi)

    def test_vectorised_driver(self):
        x = np.linspace(-1, 1, 30)
        out = precision_weighted_errors(x, np.zeros(30), alpha=0.1)
        assert set(out) == {"eps", "z_norm", "phi", "mu", "sigma2", "pi", "weighted"}
        np.testing.assert_allclose(out["eps"], x)
        with pytest.raises(ValueError):
            precision_weighted_errors(x, np.zeros(29))

    @pytest.mark.parametrize(
        "kw", [{"alpha": 0.0}, {"alpha": 1.5}, {"pi_min": 0.0}, {"pi_min": 2.0, "pi_max": 1.0}]
    )
    def test_invalid_channel_args(self, kw):
        with pytest.raises(ValueError):
            PrecisionWeightedErrorChannel(**kw)

    def test_negative_precision_rejected(self):
        with pytest.raises(ValueError):
            precision_weighted_error(1.0, -1.0)
        assert precision_weighted_error(2.0, 3.0) == 6.0


class TestSomatic:
    def test_gain_and_clamped_precision(self):
        assert somatic_gain(0.0) == 1.0
        assert somatic_gain(1.0, 0.5) == pytest.approx(np.exp(0.5))
        assert interoceptive_precision(5.0, 2.0, 2.5) == 10.0  # clamp after exp
        assert interoceptive_precision(1.0, -2.0, 1.2) >= 0.1

    def test_ne_clamp_rescales_gain(self):
        assert clamp_ne_gain(10.0, 0.1) == pytest.approx(5.0)  # γ·g = 0.5
        assert clamp_ne_gain(1.0, 0.1) == 1.0
        assert clamp_ne_gain(3.0, 0.0) == 3.0

    def test_threshold_decays_and_boosts(self):
        ad = SomaticThresholdAdapter(theta0=0.5, kappa=0.2, delta=0.4)
        out = ad.step(0.5, B=1)
        assert out.theta == pytest.approx(0.5 + 0.4)
        out = ad.step(0.5, B=0)
        assert out.theta == pytest.approx(0.5 + 0.4 * np.exp(-0.2))
        for _ in range(200):
            out = ad.step(0.5, B=0)
        assert out.theta == pytest.approx(0.5, abs=1e-9)

    def test_marker_relaxes_toward_target_and_stays_bounded(self):
        ad = SomaticThresholdAdapter(tau_M=10.0)
        for _ in range(400):
            out = ad.step(1.0, B=0, dt=1.0)
        assert -2.0 <= out.M <= 2.0 and out.M > 0.5
        assert out.gain == pytest.approx(np.exp(0.3 * out.M))

    def test_ne_pathway_is_clamped_and_serotonin_offsets_effective_only(self):
        ad = SomaticThresholdAdapter(gamma_ne=0.1, g_ne=100.0, beta_5ht=0.2)
        assert ad.gamma_ne * ad.g_ne == pytest.approx(0.5)
        out = ad.step(0.5, B=0)
        assert out.theta_eff == pytest.approx(out.theta + 0.2)

    @pytest.mark.parametrize("kw", [{"kappa": -1}, {"delta": -1}, {"tau_M": 0}, {"M0": 3}])
    def test_invalid_args(self, kw):
        with pytest.raises(ValueError):
            SomaticThresholdAdapter(**kw)

    def test_bad_B(self):
        with pytest.raises(ValueError, match="0 or 1"):
            SomaticThresholdAdapter().step(0.5, B=2)

    def test_non_finite_threshold_detected(self):
        ad = SomaticThresholdAdapter(theta0=1e308, gamma_ne=0.4, g_ne=1.0)
        with pytest.raises(FloatingPointError):
            for _ in range(5):
                ad.step(0.5, B=0)


class TestIgnitionSim:
    def test_probability_properties(self):
        S = np.linspace(-3, 3, 13)
        p = ignition_probability(S, 0.0, 0.5)
        assert np.all(np.diff(p) > 0) and p[6] == pytest.approx(0.5)
        assert ignition_probability(np.array([1e6]), 0.0, 1.0)[0] == 1.0  # no overflow
        with pytest.raises(ValueError):
            ignition_probability(S, 0.0, 0.0)

    def test_hard_limit_gap_shrinks_with_temperature(self):
        rng = np.random.default_rng(1)
        S, th = rng.normal(0.5, 0.3, 500), np.full(500, 0.5)
        gaps = hard_limit_gap(S, th, [1.0, 0.1, 0.01, 0.001])
        assert np.all(np.diff(gaps) < 0) and gaps[-1] < 0.01
        np.testing.assert_array_equal(hard_ignition([0.4, 0.6], 0.5), [0, 1])

    def test_sampling_matches_probability_and_validates(self):
        b = sample_ignition(np.full(20000, 0.3), rng=1)
        assert b.mean() == pytest.approx(0.3, abs=0.01)
        assert np.array_equal(sample_ignition([0.5] * 50, 3), sample_ignition([0.5] * 50, 3))
        with pytest.raises(ValueError):
            sample_ignition([1.2])

    def test_summaries(self):
        s = summarise_ignitions([0, 1, 0, 0, 1, 1])
        assert (s.count, s.first_ignition) == (3, 1)
        assert s.mean_inter_ignition_interval == pytest.approx((3 + 1) / 2)
        none = summarise_ignitions([0, 0])
        assert none.first_ignition is None and none.mean_inter_ignition_interval is None

    def test_sessions_and_modes(self):
        x = np.tile([0.8] * 30 + [-0.8] * 30, 4)
        p = ReducedParams(theta_base=0.3)
        hard = simulate_session(p, x, mode="hard")
        assert hard.count > 0 and simulate_session(p, x, mode="hard") == hard
        soft = simulate_session(p, x, mode="soft", seed=3)
        assert soft == simulate_session(p, x, mode="soft", seed=3)
        with pytest.raises(ValueError, match="mode"):
            simulate_session(p, x, mode="medium")

    def test_threshold_sweep_decreases_with_threshold(self):
        x = np.tile([0.8] * 30 + [-0.8] * 30, 4)
        rates = threshold_sweep(ReducedParams(), x, [0.25, 0.5, 0.85], n_sessions=6, seed=1)
        assert rates[0] > rates[1] > rates[2]
        hard = threshold_sweep(ReducedParams(), x, [0.25, 0.85], mode="hard")
        assert hard[0] >= hard[1]
        assert np.array_equal(
            rates, threshold_sweep(ReducedParams(), x, [0.25, 0.5, 0.85], n_sessions=6, seed=1)
        )


class TestLatentState:
    def _data(self, seed=3):
        p = ReducedParams()
        ds = generate_dataset(p, rng=seed)
        return p, ds

    def test_zero_process_noise_recovers_teacher_forced_likelihood(self):
        p, ds = self._data()
        d = Observed(ds.stimulus, ds.B, ds.report, ds.amplitude)
        f = filter_latent_signal(p, ds.stimulus, ds.B, ds.amplitude, sigma_s=0.0, n_particles=1)
        assert f.log_evidence == pytest.approx(
            log_likelihood(p, d, ["ignition", "amplitude"]), abs=1e-9
        )
        np.testing.assert_allclose(
            f.S_mean, simulate_reduced(p, ds.stimulus, B_forced=ds.B).S, atol=1e-12
        )
        assert not f.resampled.any() and np.all(f.S_var == 0)

    def test_tracks_the_latent_signal_with_process_noise(self):
        p, ds = self._data()
        f = filter_latent_signal(
            p, ds.stimulus, ds.B, ds.amplitude, sigma_s=0.05, n_particles=400, rng=1
        )
        assert np.abs(f.S_mean - ds.S_true).mean() < 0.06
        assert f.resampled.any() and f.ess.min() > 0
        assert np.all(f.S_var >= 0)

    def test_seeded_and_works_without_amplitude(self):
        p, ds = self._data()
        a = filter_latent_signal(p, ds.stimulus, ds.B, sigma_s=0.05, n_particles=100, rng=2)
        b = filter_latent_signal(p, ds.stimulus, ds.B, sigma_s=0.05, n_particles=100, rng=2)
        assert np.array_equal(a.S_mean, b.S_mean) and np.isfinite(a.log_evidence)
        assert set(a.as_dict()) == {"S_mean", "S_var", "ess", "log_evidence", "resampled"}

    def test_true_params_beat_wrong_params_on_evidence(self):
        p, ds = self._data()
        wrong = ReducedParams(alpha=0.9, pi=5.0, theta_base=0.3, beta=-0.2)
        good = filter_latent_signal(
            p, ds.stimulus, ds.B, ds.amplitude, sigma_s=0.02, n_particles=300, rng=1
        )
        bad = filter_latent_signal(
            wrong, ds.stimulus, ds.B, ds.amplitude, sigma_s=0.02, n_particles=300, rng=1
        )
        assert good.log_evidence > bad.log_evidence

    def test_argument_validation(self):
        p, ds = self._data()
        with pytest.raises(ValueError, match="B must"):
            filter_latent_signal(p, ds.stimulus, ds.B[:-1])
        with pytest.raises(ValueError, match="y must"):
            filter_latent_signal(p, ds.stimulus, ds.B, ds.amplitude[:-1])
        with pytest.raises(ValueError, match="sigma_s"):
            filter_latent_signal(p, ds.stimulus, ds.B, sigma_s=-1)
        with pytest.raises(ValueError, match="n_particles"):
            filter_latent_signal(p, ds.stimulus, ds.B, n_particles=0)


class TestCounterfactual:
    X = np.tile([0.8] * 30 + [-0.8] * 30, 3)

    def test_common_random_numbers_make_identical_conditions_identical(self):
        res = run_conditions(
            "apgi_reduced", {}, self.X, [baseline(), Condition("clone")], n_seeds=4, seed=1
        )
        assert np.array_equal(res["baseline"].B, res["clone"].B)
        assert contrast(res, "baseline", "clone")["mean_diff"] == 0.0

    def test_interventions_move_outcomes_in_the_expected_direction(self):
        res = run_conditions(
            "apgi_reduced",
            {"theta_base": 0.4},
            self.X,
            [baseline(), dopamine_blockade(), precision_lesion(), stimulus_gain(2.0)],
            n_seeds=8,
            seed=2,
        )
        assert res["dopamine_blockade"].S.mean() < res["baseline"].S.mean()  # β=0 removes drive
        assert res["stimulus_gain_2"].summary()["mean_S"] > res["baseline"].summary()["mean_S"]
        c = contrast(res, "baseline", "dopamine_blockade")
        assert c["n"] == 8 and np.isfinite(c["se"])

    def test_apply_does_not_mutate(self):
        base = {"beta": 0.1}
        x = np.ones(3)
        p, xs = Condition("c", {"beta": 0.0}, gain=2.0, offset=1.0).apply(base, x)
        assert base == {"beta": 0.1} and np.array_equal(x, np.ones(3))
        assert p["beta"] == 0.0 and np.array_equal(xs, np.full(3, 3.0))

    def test_validation(self):
        with pytest.raises(ValueError, match="unique"):
            run_conditions("apgi_reduced", {}, self.X, [baseline(), baseline()])
        with pytest.raises(ValueError, match="n_seeds"):
            run_conditions("apgi_reduced", {}, self.X, [baseline()], n_seeds=0)

    def test_single_seed_contrast_has_nan_se(self):
        res = run_conditions(
            "apgi_reduced", {}, self.X, [baseline(), dopamine_blockade()], n_seeds=1
        )
        assert np.isnan(contrast(res, "baseline", "dopamine_blockade")["se"])
        assert res["baseline"].summary()["ignition_rate_sd"] == 0.0

    def test_works_for_pipeline_variants(self):
        res = run_conditions("apgi_reference", {}, self.X[:60], [baseline()], n_seeds=2)
        assert res["baseline"].S.shape == (2, 60)


class TestDigest:
    def test_order_independent_rounding_and_negative_zero(self):
        a = array_digest({"x": [1.0, 2.0], "y": [0.0]})
        assert a == array_digest({"y": [0.0], "x": [1.0, 2.0]})
        assert array_digest({"y": [-0.0]}) == array_digest({"y": [0.0]})
        assert array_digest({"x": [1.0 + 1e-12]}) == array_digest({"x": [1.0]})
        assert array_digest({"x": [1.0 + 1e-6]}) != array_digest({"x": [1.0]})

    def test_non_finite_values_hash_distinctly(self):
        d = {array_digest({"x": [v]}) for v in (float("nan"), float("inf"), float("-inf"), 0.0)}
        assert len(d) == 4
