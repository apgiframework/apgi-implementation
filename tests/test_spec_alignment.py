"""Alignment with APGI specification 0.3.0 (apgi-lean PD-* divergences).

PD-FLOOR, PD-KAPPA-SYMBOL, PD-KAPPA-RESIDUAL, PD-DT, PD-TAU-THETA, PD-TAU-S,
PD-HAZARD and PD-CLIP are checked here from the production side; apgi-lean's test_production.py checks
the same defaults against its sha256-pinned registry.
"""

from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

from config import CONFIG, decay_exponent, lam_from_tau_s
from core.allostatic import allostatic_threshold_ode, update_threshold_euler
from core.deprecation import APGIDeprecationWarning, config_value
from core.ignition import hazard_probability, sample_ignition_state
from core.thermodynamics import compute_landauer_cost
from core.threshold import compute_metabolic_cost_realistic, threshold_decay


def test_floor_default_is_zero():
    """PD-FLOOR: Math Spec §4.5 default θ_floor = 0."""
    assert inspect.signature(update_threshold_euler).parameters["theta_min"].default == 0.0
    assert update_threshold_euler(0.05, -10.0, dt=0.01) == 0.0


def test_reference_step_and_timescales():
    """PD-DT, PD-TAU-THETA, PD-TAU-S: dt₀ = 10 ms, τ_θ = 20 s, τ_S = 300 ms."""
    assert CONFIG["dt"] == 0.01
    assert CONFIG["tau_theta"] == 20.0
    assert 0.2 <= CONFIG["tau_s"] <= 0.5
    assert CONFIG["dt"] <= min(CONFIG["tau_s"], CONFIG["tau_theta"], CONFIG["tau_pi"]) / 10


def test_decay_exponent_is_dt_over_tau_theta():
    assert decay_exponent(CONFIG) == pytest.approx(CONFIG["dt"] / CONFIG["tau_theta"])
    assert decay_exponent({"dt": 0.5, "tau_theta": 20.0}) == pytest.approx(0.025)


def test_per_step_quantities_rescaled():
    """λ through lam_from_tau_s; η_θ per step = rate × dt (0.2 s⁻¹)."""
    assert CONFIG["lam"] == pytest.approx(lam_from_tau_s(CONFIG["dt"], CONFIG["tau_s"]), rel=1e-12)
    assert CONFIG["eta"] == pytest.approx(0.2 * CONFIG["dt"])


def test_kappa_names_retired_from_config():
    """PD-KAPPA-SYMBOL: bare κ is ATP/bit; κ_meta is deprecated (Notation Appendix §11)."""
    for key in ("kappa", "kappa_meta", "kappa_units"):
        assert key not in CONFIG
    assert "kappa" not in inspect.signature(threshold_decay).parameters


def test_legacy_kappa_config_warns_and_is_honoured():
    with pytest.warns(APGIDeprecationWarning):
        assert decay_exponent({"kappa": 0.15, "dt": 0.5}) == 0.15


def test_legacy_keyword_aliases_warn():
    with pytest.warns(APGIDeprecationWarning):
        assert threshold_decay(2.0, 1.0, kappa=1.0) == pytest.approx(threshold_decay(2.0, 1.0, 1.0))
    with pytest.warns(APGIDeprecationWarning):
        legacy = compute_landauer_cost(1.0, 0.01, kappa_meta=2.0, kappa_units="dimensionless")
    assert legacy == pytest.approx(compute_landauer_cost(1.0, 0.01, r_l=2.0))
    with pytest.warns(APGIDeprecationWarning):
        compute_metabolic_cost_realistic(1.0, 0, kappa_meta=1.0)
    with pytest.warns(APGIDeprecationWarning):
        allostatic_threshold_ode(0.5, 0.35, 0.05, 0, 0.0, kappa_meta=0.1, noise_std=0.0, rng=0)
    with pytest.raises(TypeError):
        threshold_decay(2.0, 1.0, decay_exponent=1.0, kappa=1.0)


def test_config_value_prefers_explicit_legacy_key():
    with pytest.warns(APGIDeprecationWarning):
        assert config_value({"r_l": 1.0, "kappa_meta": 5.0}, "r_l", "kappa_meta", 0.0) == 5.0
    assert config_value({"r_l": 1.0}, "r_l", "kappa_meta", 0.0) == 1.0


@pytest.mark.parametrize("B", [0.0, 0.05, 0.5, 1.0])
@pytest.mark.parametrize("dt", [0.001, 0.01, 0.1])
def test_hazard_is_dt_invariant(B, dt):
    """PD-HAZARD: (1 − p_dt)² = 1 − p_2dt (apgi-lean Apgi.hazard_two_steps)."""
    r = CONFIG["r_max"]
    assert (1 - hazard_probability(B, dt, r)) ** 2 == pytest.approx(
        1 - hazard_probability(B, 2 * dt, r), rel=1e-12, abs=1e-15
    )
    assert hazard_probability(B, dt, r) == pytest.approx(
        1 - math.exp(-r * B * dt), rel=1e-12, abs=1e-15
    )


def test_hazard_window_probability_independent_of_dt():
    """Math Spec p22: per-step Bernoulli(Bₜ) makes a 300 ms window depend on dt; the hazard does not."""
    B, window, r = 0.05, 0.3, CONFIG["r_max"]
    win = [1 - (1 - hazard_probability(B, dt, r)) ** round(window / dt) for dt in (0.01, 0.001)]
    assert win[0] == pytest.approx(win[1], rel=1e-9)


def test_sampler_uses_hazard_and_warns_on_legacy():
    rng = np.random.default_rng(0)
    draws = [sample_ignition_state(0.5, rng=rng, dt=0.01, r_max=100.0) for _ in range(4000)]
    assert np.mean(draws) == pytest.approx(hazard_probability(0.5, 0.01, 100.0), abs=0.03)
    with pytest.warns(APGIDeprecationWarning):
        sample_ignition_state(0.5, rng=0)
    with pytest.raises(TypeError):
        sample_ignition_state(0.5, rng=0, dt=0.01)


def test_theta_clip_floor_is_spec_floor():
    """PD-CLIP: the pipeline's lower θ clamp is θ_floor = 0 (Math Spec §4.5)."""
    assert CONFIG["theta_clip_min"] == 0.0
    assert CONFIG["theta_clip_max"] > CONFIG["theta_clip_min"]


def test_bare_kappa_retired_from_non_atp_quantities():
    """PD-KAPPA-RESIDUAL: no bare κ names a learning rate, decay exponent or R_L."""
    import analysis.stability as stability
    import core.dynamics as dynamics
    import core.preprocessing as preprocessing
    import energy.bold_calibration as bold
    import models.reduced as reduced
    from core.config_schema import APGIConfig
    from engine.somatic import SomaticThresholdAdapter

    for fn in (dynamics.update_prediction, preprocessing.update_prediction):
        assert "eta_pred" in inspect.signature(fn).parameters
    assert "decay_exponent" in inspect.signature(stability.compute_jacobian_discrete).parameters
    assert "decay_exponent" in inspect.signature(SomaticThresholdAdapter).parameters
    assert "r_l" in inspect.signature(bold.estimate_bits_from_bold_energy).parameters
    assert "KAPPA" not in reduced.__all__ and reduced.DECAY_EXPONENT == 0.15
    for key in ("kappa_e", "kappa_i"):
        assert key not in CONFIG and key not in APGIConfig.model_fields
    assert CONFIG["eta_pred_e"] < 2.0 / CONFIG["pi_max"]


def test_residual_kappa_aliases_warn_and_agree():
    import analysis.stability as stability
    import energy.bold_calibration as bold
    import models.reduced as reduced
    from core.config_schema import APGIConfig
    from core.dynamics import update_prediction
    from energy.calibration_utils import calibrate_for_realistic_kappa
    from engine.somatic import SomaticThresholdAdapter

    with pytest.warns(APGIDeprecationWarning):
        assert update_prediction(0.0, 1.0, 2.0, kappa=0.1) == update_prediction(0.0, 1.0, 2.0, 0.1)
    with pytest.warns(APGIDeprecationWarning):
        J = stability.compute_jacobian_discrete(0.2, kappa=0.05, c1=0.2, eta=0.1)
    assert J[1, 1] == pytest.approx(math.exp(-0.05))
    with pytest.warns(APGIDeprecationWarning):
        assert SomaticThresholdAdapter(kappa=0.2).decay_exponent == 0.2
    with pytest.warns(APGIDeprecationWarning):
        assert SomaticThresholdAdapter(decay_exponent=0.2).kappa == 0.2
    with pytest.warns(APGIDeprecationWarning):
        assert reduced.KAPPA == reduced.DECAY_EXPONENT
    with pytest.warns(APGIDeprecationWarning):
        assert bold.calibrate_kappa_meta_from_bold(2.0, 10.0) == bold.calibrate_r_l_from_bold(
            2.0, 10.0
        )
    with pytest.warns(APGIDeprecationWarning):
        assert APGIConfig(kappa_e=0.05).eta_pred_e == 0.05
    with pytest.warns(APGIDeprecationWarning):
        cal = calibrate_for_realistic_kappa(target_kappa_multiple=10.0)
    with pytest.warns(APGIDeprecationWarning):
        assert cal["target_kappa_multiple"] == cal["target_r_l"] == 10.0
    with pytest.warns(APGIDeprecationWarning):
        assert config_value({"kappa_e": 0.3}, "eta_pred_e", "kappa_e", 0.01) == 0.3


def test_renamed_result_keys_resolve_with_warning():
    from analysis.stability import check_stability
    from energy.bold_calibration import BOLDCalibrator

    result = check_stability({"lam": 0.2, "tau_theta": 20.0, "dt": 0.01}, verbose=False)
    constraints = result["constraints_satisfied"]
    assert "kappa_positive" not in constraints
    with pytest.warns(APGIDeprecationWarning):
        assert constraints["kappa_positive"] is constraints["decay_exponent_positive"]
    cal = BOLDCalibrator()
    r_l = cal.calibrate_from_trial(1.0, 2.5, 6.6)
    summary = cal.get_calibration_summary()
    assert summary["r_l_mean"] == pytest.approx(r_l)
    with pytest.warns(APGIDeprecationWarning):
        assert summary.get("kappa_mean") == summary["r_l_mean"]
    with pytest.warns(APGIDeprecationWarning):
        assert cal.calibrated_kappa == cal.calibrated_r_l
    with pytest.raises(KeyError):
        summary["no_such_key"]
