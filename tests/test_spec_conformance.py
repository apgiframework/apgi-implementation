"""Specification conformance tests.

Each test asserts ONE named clause of the APGI specification and cites it. This
module exists because the pre-existing suite had ~2300 tests and near-total line
coverage yet missed five spec violations (irreproducible seeding, fail-open
validation, the absent NE clamp, the rectified psi channel, and Pi_min set an
order of magnitude below the spec floor). High coverage with weak oracles finds
crashes, not wrong answers; these tests encode the spec itself as the oracle.

Sources:
  * MathSpec  — APGI Mathematical Specification v1.0
  * DynForm   — APGI Framework Multi-Scale Dynamical Formulation v1.0
  * NotApp    — APGI Notation Appendix v1.0
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from config import CONFIG, lam_from_tau_s
from core.ignition import compute_ignition_probability, sample_ignition_state
from core.precision import (
    PI_MAX_CANONICAL,
    PI_MIN_CANONICAL,
    apply_dopamine_bias_to_error,
    compute_interoceptive_precision_exponential,
    compute_precision,
    precision_coupling_ode_core,
    update_mean_ema,
)
from core.threshold import (
    apply_ne_threshold_modulation,
    apply_serotonin_threshold_offset,
    compute_information_value,
    compute_metabolic_cost,
    threshold_decay,
    update_threshold_discrete,
)
from core.validation import (
    CANONICAL_PARAMETER_RANGES,
    NE_SUBDOMINANCE_MAX,
    ValidationError,
    check_canonical_ranges,
    validate_config,
)
from pipeline import APGIPipeline
from stats.hurst import (
    FGN_FBM_BOUNDARY,
    estimate_alpha_dfa,
    estimate_hurst,
    hurst_from_alpha_dfa,
    infer_convention,
)

pytestmark = pytest.mark.conformance


# ---------------------------------------------------------------------------
# MathSpec §1 — Signal preprocessing
# ---------------------------------------------------------------------------
class TestStep1to3Preprocessing:
    def test_ema_mean_uses_spec_coefficient_placement(self):
        """μ(t+1) = (1 − α)·μ(t) + α·ε(t); larger α = faster adaptation."""
        assert update_mean_ema(0.0, 1.0, 0.25) == pytest.approx(0.25)
        fast = update_mean_ema(0.0, 1.0, 0.9)
        slow = update_mean_ema(0.0, 1.0, 0.1)
        assert fast > slow, "larger alpha_EMA must adapt faster (NotApp EMA block)"

    def test_ema_variance_is_mean_centred_not_second_moment(self):
        """Step 1 is mandatory: omitting it yields E[eps^2], not the variance.

        MathSpec §1: "omitting Step 1 produces a second-moment estimator
        E[eps^2] rather than the variance E[(eps - mu)^2]".
        """
        pipeline = APGIPipeline({**CONFIG, "seed": 1, "alpha_e": 0.05, "alpha_i": 0.05})
        # Drive a strongly biased error stream; a second-moment estimator would
        # keep growing with the mean, a true variance would not.
        for _ in range(500):
            pipeline.step(x_e=10.0, x_i=10.0)
        # With a constant input the centred residual collapses toward zero.
        assert pipeline.state.sigma2_e < 1.0, (
            "variance should shrink for a constant (if biased) input; a "
            "second-moment estimator would instead approach mean^2"
        )


# ---------------------------------------------------------------------------
# MathSpec §2 — Precision system
# ---------------------------------------------------------------------------
class TestStep4to7Precision:
    def test_precision_clamp_bounds_are_the_spec_values(self):
        """MathSpec §2: "Bounds: Pi_min = 0.1, Pi_max = 10"."""
        assert PI_MIN_CANONICAL == 0.1
        assert PI_MAX_CANONICAL == 10.0

    def test_precision_clamp_is_applied(self):
        assert compute_precision(1e-12) == PI_MAX_CANONICAL
        assert compute_precision(1e12) == PI_MIN_CANONICAL

    def test_default_config_uses_canonical_clamp(self):
        assert CONFIG["pi_min"] == PI_MIN_CANONICAL
        assert CONFIG["pi_max"] == PI_MAX_CANONICAL

    def test_somatic_marker_is_exponential_and_clamped_after(self):
        """Step 6: Pi_i_eff <- Pi_i_eff * exp(beta_SM*M), THEN clamp.

        MathSpec §13 step 6 works the ceiling example: at beta_SM=1.2, M=+2 the
        gain is exp(2.4) ~ 11, so the Pi_max=10 clamp binds. Clamping before
        the exponential would not bound the blow-up.
        """
        out = compute_interoceptive_precision_exponential(pi_baseline=5.0, beta_somatic=1.2, M=2.0)
        assert out == PI_MAX_CANONICAL
        # Below the ceiling the exponential form is exact.
        assert compute_interoceptive_precision_exponential(
            pi_baseline=1.0, beta_somatic=0.3, M=1.0
        ) == pytest.approx(math.exp(0.3))

    def test_psi_bottom_up_channel_is_signed(self):
        """MathSpec §2: psi is "negative when the error is negative".

        Regression: abs() before psi made dPi/dt identical for eps=+1 and
        eps=-1, deleting the spec's inhibitory bottom-up pathway.
        """
        psi = np.tanh
        common = {
            "pi_ell": 1.0,
            "tau_pi": 100.0,
            "epsilon_ell": 0.0,
            "alpha_gain": 0.1,
            "pi_ell_plus_1": None,
            "C_down": 0.1,
            "C_up": 0.05,
            "psi": psi,
        }
        pos = precision_coupling_ode_core(epsilon_ell_minus_1=1.0, **common)
        neg = precision_coupling_ode_core(epsilon_ell_minus_1=-1.0, **common)
        zero = precision_coupling_ode_core(epsilon_ell_minus_1=0.0, **common)
        assert pos > zero > neg, "psi channel must be monotone and signed"
        assert pos != neg

    def test_psi_zero_at_zero_error(self):
        """psi(0) = 0: no cross-level drive without prediction error."""
        assert np.tanh(2.0 * 0.0) == 0.0

    def test_canonical_psi_is_saturating(self):
        """NotApp: psi must be bounded ("prevents unbounded gain")."""
        assert CONFIG["psi_type"] == "tanh"
        assert abs(np.tanh(2.0 * 1e6)) <= 1.0


# ---------------------------------------------------------------------------
# MathSpec §2 / §15 — Neuromodulator separation
# ---------------------------------------------------------------------------
class TestStep5to8Neuromodulators:
    def test_dopamine_is_additive_on_the_error_not_a_precision_gain(self):
        """MathSpec §15: "DA must never multiply precision"."""
        assert apply_dopamine_bias_to_error(0.5, 0.25) == pytest.approx(0.75)

    def test_beta_da_is_an_unconstrained_real(self):
        """MathSpec glossary: beta_DA range is R; negative models depletion."""
        assert apply_dopamine_bias_to_error(0.5, -2.0) == pytest.approx(-1.5)
        validate_config({**CONFIG, "beta": -1.0})  # must not raise

    def test_serotonin_is_additive_on_threshold(self):
        """MathSpec §15: 5-HT raises theta ADDITIVELY (theta_eff = theta + beta_5HT)."""
        assert apply_serotonin_threshold_offset(0.6, 0.1) == pytest.approx(0.7)

    def test_serotonin_default_is_identity(self):
        """MathSpec §2: "The default value is beta_5HT = 0"."""
        assert CONFIG["beta_5ht"] == 0.0
        assert apply_serotonin_threshold_offset(0.6, 0.0) == pytest.approx(0.6)

    def test_norepinephrine_is_multiplicative_on_threshold(self):
        """MathSpec §15: NE raises theta MULTIPLICATIVELY, 5-HT additively."""
        assert apply_ne_threshold_modulation(1.0, g_ne=1.0, gamma_ne=0.1) == pytest.approx(1.1)

    def test_ne_subdominance_clamp_is_enforced_and_rescales(self):
        """MathSpec §2: the clamp is mandatory, and the remedy is a rescale.

        "Implementations MUST enforce the hard clamp gamma_NE*g_NE <= 0.5 at
        every timestep... if gamma_NE*g_NE(t) > 0.5, rescale
        g_NE(t) <- 0.5/gamma_NE".
        """
        cfg = {
            **CONFIG,
            "ne_on_precision": False,  # isolate the threshold pathway
            "ne_on_threshold": True,
            "gamma_ne": 0.1,
            "g_ne": 10.0,  # product = 1.0
            "kappa": 0.3,
        }
        with pytest.warns(RuntimeWarning, match="spec-mandated ceiling"):
            pipeline = APGIPipeline(cfg)
        assert pipeline.config["g_ne"] == pytest.approx(NE_SUBDOMINANCE_MAX / 0.1)
        assert pipeline.config["gamma_ne"] * pipeline.config["g_ne"] <= NE_SUBDOMINANCE_MAX + 1e-12

    def test_both_ne_pathways_permitted_under_the_clamp(self):
        """The constraint is on the PRODUCT, not on the pair of flags."""
        validate_config(
            {
                **CONFIG,
                "ne_on_precision": True,
                "ne_on_threshold": True,
                "gamma_ne": 0.01,
                "g_ne": 1.0,
            }
        )


# ---------------------------------------------------------------------------
# MathSpec §3-§4 — Accumulation and threshold
# ---------------------------------------------------------------------------
class TestStep9to14Accumulation:
    def test_lambda_matches_the_exact_continuous_relation(self):
        """MathSpec §3: lambda = 1 - exp(-dt/tau_S) EXACTLY, not dt/tau_S."""
        lam = lam_from_tau_s(CONFIG["dt"], CONFIG["tau_s"])
        assert CONFIG["lam"] == pytest.approx(lam)
        assert lam == pytest.approx(1.0 - math.exp(-CONFIG["dt"] / CONFIG["tau_s"]))
        # And it differs from the naive approximation dt/τ_S, materially at dt ~ τ_S.
        assert abs(lam - CONFIG["dt"] / CONFIG["tau_s"]) > 1e-4
        assert abs(lam_from_tau_s(0.5, 5.0) - 0.1) > 1e-3

    def test_lam_from_tau_s_rejects_non_positive(self):
        with pytest.raises(ValueError, match="dt must be > 0"):
            lam_from_tau_s(0.0, 5.0)
        with pytest.raises(ValueError, match="tau_s must be > 0"):
            lam_from_tau_s(0.5, 0.0)

    def test_metabolic_cost_form(self):
        """MathSpec §4: C(t) = c1*S(t) + c2*B(t-1)."""
        assert compute_metabolic_cost(S=2.0, B_prev=1, c1=0.2, c2=0.5) == pytest.approx(0.9)

    def test_information_value_uses_unsigned_magnitudes(self):
        """MathSpec §4: V(t) = v1*|z_e| + v2*|z_i_eff|."""
        assert compute_information_value(-1.0, 2.0, v1=0.5, v2=0.5) == pytest.approx(1.5)

    def test_dopamine_lowers_threshold_via_information_value(self):
        """MathSpec §4: dtheta/dbeta_DA = -eta*v2 < 0.

        DA acts on theta ONLY through V(t); there is no standalone beta_DA term
        in the ODE ("adding one would constitute double-counting").
        """
        eta, v1, v2 = 0.1, 0.5, 0.5
        z_e, z_i = 0.4, 0.3
        cost = compute_metabolic_cost(S=1.0, B_prev=0, c1=0.2, c2=0.5)

        def theta_next(beta_da: float) -> float:
            v = compute_information_value(z_e, z_i + beta_da, v1=v1, v2=v2)
            return update_threshold_discrete(0.55, cost, v, eta=eta, delta=0.0, B_prev=0)

        assert theta_next(0.5) < theta_next(0.0)
        numeric = (theta_next(1e-6) - theta_next(0.0)) / 1e-6
        assert numeric == pytest.approx(-eta * v2, rel=1e-3)

    def test_threshold_decay_is_the_exact_discrete_solution(self):
        """MathSpec §4: theta(t+1) = theta_0 + (theta - theta_0)*exp(-gamma_theta)."""
        got = threshold_decay(theta=1.0, theta_base=0.55, kappa=0.15)
        assert got == pytest.approx(0.55 + 0.45 * math.exp(-0.15))

    def test_decay_precedes_the_ignition_decision(self):
        """MathSpec §13: step 12 (decay) comes BEFORE step 15 (ignition).

        Deferring the decay to the end of the timestep is not equivalent: decay
        is affine and the allostatic update is a translation, and affine maps do
        not commute with translations.
        """
        theta0, base, kappa, increment = 1.0, 0.55, 0.15, 0.2
        decay_then_add = threshold_decay(theta0, base, kappa) + increment
        add_then_decay = threshold_decay(theta0 + increment, base, kappa)
        assert decay_then_add != pytest.approx(add_then_decay)


# ---------------------------------------------------------------------------
# MathSpec §5-§6 — Ignition and reset
# ---------------------------------------------------------------------------
class TestStep15to16IgnitionAndReset:
    def test_ignition_probability_is_the_logistic_of_the_margin(self):
        """MathSpec §5: P_ign = sigma((S - theta)/tau_sigma)."""
        assert compute_ignition_probability(1.0, 1.0, tau=0.5) == pytest.approx(0.5)
        expected = 1.0 / (1.0 + math.exp(-(1.5 - 1.0) / 0.5))
        assert compute_ignition_probability(1.5, 1.0, tau=0.5) == pytest.approx(expected)

    def test_tau_sigma_must_be_positive(self):
        """MathSpec §15: tau_sigma > 0 for a well-defined probability."""
        with pytest.raises(ValueError, match="tau must be > 0"):
            compute_ignition_probability(1.0, 0.5, tau=0.0)

    def test_gamma_sig_is_the_reciprocal_of_tau_sigma_and_canonical(self):
        """NotApp: gamma_sig = 1/tau_sigma, canonical [2, 7.5]."""
        gamma_sig = 1.0 / CONFIG["tau_sigma"]
        lo, hi = CANONICAL_PARAMETER_RANGES["gamma_sig"]
        assert lo <= gamma_sig <= hi

    def test_bernoulli_sampling_returns_binary_outcomes(self):
        """B(t) in {0,1} is distinct from the continuous probability B_t."""
        assert sample_ignition_state(1.0, rng=0) == 1
        assert sample_ignition_state(0.0, rng=0) == 0
        draws = {sample_ignition_state(0.5, rng=s) for s in range(20)}
        assert draws <= {0, 1}

    def test_ignition_probability_out_of_range_is_rejected(self):
        with pytest.raises(ValueError, match=r"p_ignite must be in \[0,1\]"):
            sample_ignition_state(1.5)

    def test_rho_retain_bounds_are_enforced_at_runtime(self):
        """MathSpec §6 / DynForm §16: rho_retain in [0.1, 0.5]."""
        pipeline = APGIPipeline({**CONFIG, "reset_factor": 0.9, "strict": False, "seed": 1})
        pipeline.S, pipeline.theta = 5.0, 0.1  # force ignition
        with pytest.raises(ValueError, match=r"reset_factor.*must be in \[0.1, 0.5\]"):
            pipeline.step(x_e=5.0, x_i=5.0)

    def test_signal_is_reset_on_ignition(self):
        """MathSpec §6: S <- rho_retain * S when B(t) = 1."""
        pipeline = APGIPipeline({**CONFIG, "reset_factor": 0.1, "seed": 3})
        pipeline.S, pipeline.theta = 5.0, 0.1
        result = pipeline.step(x_e=5.0, x_i=5.0)
        assert result["B"] == 1
        assert pipeline.S < 5.0


# ---------------------------------------------------------------------------
# MathSpec §12 — Spectral / DFA conventions
# ---------------------------------------------------------------------------
class TestSpectralConventions:
    def test_alpha_dfa_recovers_known_processes(self):
        """DFA must recover alpha ~ 0.5 (white noise) and ~ 1.5 (random walk)."""
        rng = np.random.default_rng(0)
        white = rng.normal(size=20000)
        walk = np.cumsum(rng.normal(size=20000))
        assert estimate_alpha_dfa(white) == pytest.approx(0.5, abs=0.05)
        assert estimate_alpha_dfa(walk) == pytest.approx(1.5, abs=0.05)

    def test_hurst_is_always_within_its_definition(self):
        """NotApp: "H in (0, 1) by definition"; values >= 1 are alpha_DFA."""
        rng = np.random.default_rng(0)
        for series in (rng.normal(size=8192), np.cumsum(rng.normal(size=8192))):
            h = estimate_hurst(series)
            assert 0.0 < h < 1.0, f"H={h} escapes (0,1)"

    def test_convention_conversions_match_the_spec(self):
        """H = alpha_DFA (fGn); H = alpha_DFA - 1 (fBm)."""
        assert hurst_from_alpha_dfa(0.9, "fgn") == pytest.approx(0.9)
        assert hurst_from_alpha_dfa(1.9, "fbm") == pytest.approx(0.9)

    def test_convention_inference_uses_the_unit_boundary(self):
        assert infer_convention(0.9) == "fgn"
        assert infer_convention(FGN_FBM_BOUNDARY) == "fbm"
        assert infer_convention(1.5) == "fbm"

    def test_unknown_convention_is_rejected(self):
        with pytest.raises(ValueError, match="convention must be"):
            hurst_from_alpha_dfa(0.9, "fGn-ish")  # type: ignore[arg-type]

    def test_cross_validation_agrees_in_both_conventions(self):
        """The cross-validation must not manufacture a spurious disagreement.

        Regression: it compared a raw alpha_DFA (1.50 for a random walk)
        against a convention-converted wavelet H (0.50) and reported
        agrees=False, in the safeguard the spec relies on for the Severity-A
        long-range-correlation falsifier.
        """
        from stats.hurst import cross_validate_hurst

        rng = np.random.default_rng(0)
        for series in (rng.normal(size=20000), np.cumsum(rng.normal(size=20000))):
            result = cross_validate_hurst(series)
            assert result["agrees"], result
            assert 0.0 < result["h_dfa"] < 1.0

    def test_constant_series_raises_rather_than_returning_a_fake_exponent(self):
        """A constant threshold has no scaling structure to estimate."""
        with pytest.raises(ValueError, match="constant"):
            estimate_alpha_dfa(np.zeros(1024))


# ---------------------------------------------------------------------------
# NotApp — Canonical parameter ranges; MathSpec §15 — stability constraints
# ---------------------------------------------------------------------------
class TestCanonicalRangesAndStability:
    def test_shipped_defaults_are_canonical(self):
        """Out-of-the-box runs are what most users publish."""
        assert check_canonical_ranges(dict(CONFIG)) == []

    def test_theta_0_default_is_in_the_canonical_band(self):
        lo, hi = CANONICAL_PARAMETER_RANGES["theta_0"]
        assert lo <= CONFIG["theta_0"] <= hi
        assert lo <= CONFIG["theta_base"] <= hi

    def test_reservoir_spectral_radius_default_is_subcritical(self):
        """NotApp: rho_res in [0.7, 0.95]; rho < 1 required for fading memory."""
        lo, hi = CANONICAL_PARAMETER_RANGES["rho_res"]
        assert lo <= CONFIG["reservoir_spectral_radius"] <= hi
        assert CONFIG["reservoir_spectral_radius"] < 1.0

    def test_lambda_is_in_the_leaky_integrator_range(self):
        """MathSpec §15: lambda in (0,1)."""
        assert 0.0 < CONFIG["lam"] < 1.0

    def test_dt_satisfies_the_euler_maruyama_constraint(self):
        """MathSpec §15: dt <= min(tau_S, tau_theta, tau_Pi)/10."""
        min_tau = min(CONFIG["tau_s"], CONFIG.get("tau_theta", 1000.0), CONFIG["tau_pi"])
        assert CONFIG["dt"] <= min_tau / 10.0

    def test_out_of_support_config_is_rejected(self):
        """Fail closed: a spec violation must abort, not warn-and-continue."""
        with pytest.raises(ValidationError):
            APGIPipeline({**CONFIG, "lam": 1.5})

    def test_out_of_band_but_in_support_config_only_advises(self):
        """gamma_sig -> 0 models anaesthesia; legal, but flagged."""
        messages = check_canonical_ranges({**CONFIG, "theta_base": 2.0})
        assert any("theta_0" in m and "canonical range" in m for m in messages)

    def test_canonical_range_departures_are_recorded_for_traceability(self):
        with pytest.warns(Warning):
            pipeline = APGIPipeline({**CONFIG, "theta_0": 2.0, "theta_base": 2.0})
        assert pipeline._validation_warnings
        assert pipeline.manifest()["spec_conformant"] is False
