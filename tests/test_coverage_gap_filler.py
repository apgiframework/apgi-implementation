import numpy as np
import pytest
from pydantic import ValidationError as PydanticValidationError

from core.config_schema import APGIConfig, create_production_config
from core.validation import (
    ValidationError,
    format_constraint_summary,
    print_constraint_summary,
    validate_config,
    validate_parameter,
)
from energy.bold_calibration import BOLDCalibrator, calibrate_r_l_from_bold
from energy.calibration_utils import demonstrate_calibration_range
from hierarchy.coupling import (
    HierarchicalPrecisionNetwork,
    bidirectional_phase_coupling,
    bidirectional_threshold_cascade,
    bottom_up_threshold_cascade,
    estimate_hierarchy_levels,
    nonlinear_phase_amplitude_coupling,
    phase_locked_threshold,
    precision_coupling_ode,
    update_phase_kuramoto_full,
)
from hierarchy.multiscale import (
    aggregate_multiscale_signal,
    aggregate_multiscale_signal_phi,
    apply_reset_rule,
    bottom_up_cascade,
    build_timescales,
    estimate_hierarchy_levels_from_data,
    estimate_optimal_timescale_ratio,
    modulate_threshold,
    multiscale_weights,
    phase_signal,
    update_multiscale_feature,
)
from oscillation.kuramoto import HierarchicalKuramotoSystem, KuramotoOscillators
from reservoir.liquid_network import LiquidNetwork
from reservoir.liquid_state_machine import LiquidStateMachine
from stats.hurst import (
    dfa_analysis,
    estimate_beta_welch,
    estimate_hurst_dfa,
    estimate_hurst_robust,
    estimate_spectral_beta,
    hurst_from_slope,
    power_spectrum,
)
from stats.maturity_assessment import (
    assess_overall_maturity,
    format_maturity_assessment,
    get_maturity_rating,
    log_maturity_assessment,
    print_maturity_assessment,
)
from stats.spectral_extraction import (
    bootstrap_confidence_interval,
    compute_aic_bic,
    estimate_spectral_exponent_periodogram,
    estimate_spectral_exponent_welch,
    extract_1f_signature,
    robust_log_regression,
    validate_hierarchical_spectral_signature,
)
from stats.spectral_extraction import estimate_hurst_dfa as extract_dfa


def test_config_schema_gaps():
    # Trigger backward compatibility for beta_da
    cfg = APGIConfig(beta_da=1.5)
    assert cfg.beta == 1.5
    # Trigger alpha errors - Pydantic raises its own ValidationError
    with pytest.raises(PydanticValidationError):
        APGIConfig(alpha_e=0.0)
    with pytest.raises(PydanticValidationError):
        APGIConfig(alpha_i=0.0)
    # Trigger ignite_tau backward compatibility
    cfg = APGIConfig(tau_sigma=0.7)
    assert cfg.ignite_tau == 0.7


def test_core_validation_gaps():
    # Trigger various ValidationErrors in core/validation.py
    # Set dt very small to avoid dt stability error
    base_cfg = {"dt": 0.01}
    # timescale_k <= 1
    with pytest.raises(ValidationError, match="timescale_k must be > 1"):
        validate_config({**base_cfg, "use_hierarchical": True, "timescale_k": 0.5})
    # pi_min <= 0
    with pytest.raises(ValidationError, match="pi_min must be > 0"):
        validate_config({**base_cfg, "pi_min": 0})
    # pi_max <= pi_min
    with pytest.raises(ValidationError, match="pi_max must be > pi_min"):
        validate_config({**base_cfg, "pi_min": 100, "pi_max": 50})
    # eta_pred_i >= max_kappa
    # max_kappa = 2 / pi_max. If pi_max=100, max_kappa=0.02.
    with pytest.raises(ValidationError, match="eta_pred_i="):
        validate_config(
            {**base_cfg, "use_internal_predictions": True, "pi_max": 100, "eta_pred_i": 1.0}
        )
    # alpha_e/i errors
    with pytest.raises(ValidationError, match="alpha_e="):
        validate_config({**base_cfg, "variance_method": "ema", "alpha_e": 0})
    with pytest.raises(ValidationError, match="alpha_i="):
        validate_config({**base_cfg, "variance_method": "ema", "alpha_i": 0})
    # T_win error
    with pytest.raises(ValidationError, match="T_win="):
        validate_config({**base_cfg, "variance_method": "sliding_window", "T_win": 0})
    # T_win warning
    with pytest.warns(RuntimeWarning, match="T_win=2 is very small"):
        validate_config({**base_cfg, "variance_method": "sliding_window", "T_win": 2})
    # eps error
    with pytest.raises(ValidationError, match="eps must be in"):
        validate_config({**base_cfg, "eps": 0})
    # eta error
    with pytest.raises(ValidationError, match="eta must be in"):
        validate_config({**base_cfg, "eta": 0})
    # noise_std error
    with pytest.raises(ValidationError, match="noise_std must be"):
        validate_config({**base_cfg, "noise_std": -1})
    # c0 error
    with pytest.raises(ValidationError, match="c0="):
        validate_config({**base_cfg, "c0": -1})
    # delta error
    with pytest.raises(ValidationError, match="delta="):
        validate_config({**base_cfg, "delta": -1})
    # signal_log_nonlinearity type error
    with pytest.raises(ValidationError, match="signal_log_nonlinearity must be boolean"):
        validate_config({**base_cfg, "signal_log_nonlinearity": "yes"})
    # validate_parameter coverage
    validate_parameter("x", 0.5, "in (0, 1)", "§1")
    with pytest.raises(ValidationError):
        validate_parameter("x", 1.5, "in (0, 1)", "§1")
    validate_parameter("x", 5, ">= 5", "§1")
    with pytest.raises(ValidationError):
        validate_parameter("x", 4, ">= 5", "§1")
    validate_parameter("x", 5, "> 4", "§1")
    with pytest.raises(ValidationError):
        validate_parameter("x", 4, "> 4", "§1")
    validate_parameter("x", 4, "<= 4", "§1")
    with pytest.raises(ValidationError):
        validate_parameter("x", 5, "<= 4", "§1")
    validate_parameter("x", 4, "< 5", "§1")
    with pytest.raises(ValidationError):
        validate_parameter("x", 5, "< 5", "§1")
    # Summary formatting
    format_constraint_summary()
    print_constraint_summary()


def test_energy_bold_gaps():
    # BOLDCalibrator summary BEFORE calibration
    cal = BOLDCalibrator()
    assert cal.get_calibration_summary() == {"calibrated": False, "message": "No calibration data"}
    # calibrate_r_l_from_bold with bits_erased=0
    assert calibrate_r_l_from_bold(1.0, 0.0) == 0.0
    # BOLDCalibrator with estimated_bits=0
    assert cal.calibrate_from_trial(1.0, 1.1, 0.0) == 0.0
    # demonstrate_calibration_range
    demonstrate_calibration_range()


def test_hierarchy_gaps():
    # estimate_hierarchy_levels
    estimate_hierarchy_levels(0.01, 10.0)
    # precision_coupling_ode
    precision_coupling_ode(1.0, 1000.0, 0.1, 0.1, 1.1, 0.05, 0.1, 0.05, psi=lambda x: x)
    # phase_locked_threshold
    phase_locked_threshold(1.0, 1.1, 0.5, 0.1, phase_sensitivity=1.5)
    # nonlinear_phase_amplitude_coupling
    nonlinear_phase_amplitude_coupling(1.0, 1.1, 0.5, 0.1, nonlinearity="sigmoid")
    nonlinear_phase_amplitude_coupling(1.0, 1.1, 0.5, 0.1, nonlinearity="power")
    nonlinear_phase_amplitude_coupling(1.0, 1.1, 0.5, 0.1, nonlinearity="exponential")
    nonlinear_phase_amplitude_coupling(1.0, 1.1, 0.5, 0.1, nonlinearity="other")
    # bidirectional_phase_coupling
    bidirectional_phase_coupling(0.5, 0.6, 0.4, 0.1, 0.1, noise_std=0.1)
    # bottom_up_threshold_cascade
    bottom_up_threshold_cascade(1.0, 2.0, 1.0, 0.1)
    # bidirectional_threshold_cascade
    bidirectional_threshold_cascade(1.0, 2.0, 1.0, 2.0, 1.0)
    # update_phase_kuramoto_full
    update_phase_kuramoto_full(
        np.array([0.5, 0.6]), np.array([0.1, 0.1]), np.ones((2, 2)), 0.1, noise_std=0.1
    )
    # HierarchicalPrecisionNetwork
    hpn = HierarchicalPrecisionNetwork(n_levels=2)
    hpn.step(np.array([0.1, 0.1]))
    hpn.compute_thresholds(np.array([1.0, 1.0]), S_levels=np.array([2.0, 0.0]), kappa_up=0.1)


def test_stats_spectral_extraction_gaps():
    # robust_log_regression with low variance
    assert np.isnan(robust_log_regression(np.array([1, 1, 1]), np.array([1, 2, 3])))[0]
    # estimate_spectral_exponent_welch with fmin/fmax filtering
    sig = np.random.normal(0, 1, 100)
    estimate_spectral_exponent_welch(sig, fmin=0.1, fmax=0.4)
    # Too few points for mask
    assert np.isnan(estimate_spectral_exponent_welch(sig, fmin=0.49, fmax=0.5))[0]
    assert np.isnan(estimate_spectral_exponent_periodogram(sig, fmin=0.49, fmax=0.5))[0]
    # DFA gaps
    # Too few segments
    assert np.isnan(extract_dfa(np.random.normal(0, 1, 10), min_lag=5, max_lag=8))[0]
    # Bootstrap failure
    assert np.isnan(bootstrap_confidence_interval(np.array([1, 1]), lambda x: np.nan))[0]
    # compute_aic_bic error
    assert np.isnan(compute_aic_bic(0, 2, 10))[0]
    # extract_1f_signature with no methods
    with pytest.raises(ValueError, match="All spectral estimation methods failed"):
        extract_1f_signature(np.array([1, 1, 1]), methods=[])
    # extract_1f_signature with failed methods (triggering try-except)
    extract_1f_signature(np.random.normal(0, 1, 100), methods=["welch"], n_bootstrap=2)
    # Fallback CI
    extract_1f_signature(np.random.normal(0, 1, 100), compute_ci=False)
    # Hierarchical spectral gaps
    validate_hierarchical_spectral_signature(
        [np.random.normal(0, 1, 100), np.random.normal(0, 1, 100)], fmin=0.1, fmax=0.4
    )
    # Trigger exception in hierarchical signature
    validate_hierarchical_spectral_signature([np.array([1])])  # Too short


def test_reservoir_gaps():
    # LiquidNetwork
    net = LiquidNetwork(n_units=10)
    # Trigger some internal state update
    net.step(u=0.5)


def test_liquid_state_machine_gaps():
    lsm = LiquidStateMachine(N=10, M=2)
    lsm.reset_state()


def test_stats_hurst_gaps():
    # compute_hurst_exponent gaps
    sig = np.random.normal(0, 1, 100)
    estimate_spectral_beta([1, 2], [1, 1])
    estimate_beta_welch(sig)
    hurst_from_slope(1.0)
    power_spectrum([1.0], [1.0], [1.0])
    dfa_analysis(sig)
    estimate_hurst_dfa(sig)
    estimate_hurst_robust(sig)


def test_main_gaps():
    # create_production_config strict=False
    create_production_config(strict=False)


def test_hierarchy_multiscale_gaps():
    # Functions from multiscale
    build_timescales(1.0, 1.6, 3)
    estimate_optimal_timescale_ratio(np.random.normal(0, 1, 1000))
    estimate_hierarchy_levels_from_data(np.random.normal(0, 1, 1000))
    update_multiscale_feature(0.0, 1.0, 10.0)
    multiscale_weights(3, 1.6)
    aggregate_multiscale_signal([1, 2], [1, 1], [0.5, 0.5])
    aggregate_multiscale_signal_phi([1, 2], [1, 1], [0.5, 0.5])
    apply_reset_rule(1.0, 1.0)
    phase_signal(0.1, 10.0)
    modulate_threshold(1.0, 1.0, 0.0, 0.1)
    bottom_up_cascade(1.0, 2.0, 1.0, 0.1)
    bottom_up_cascade(1.0, 0.5, 1.0, 0.1)


def test_kuramoto_gaps():
    # Classes from kuramoto
    ks = HierarchicalKuramotoSystem(n_levels=3)
    ks.step(0.1)
    ks.apply_ignition_reset(1, broadcast=True)
    ks.get_phase_modulation_factor(1)
    ks.get_phase_modulation_factor(5)  # Out of range
    ko = KuramotoOscillators(n_levels=3)
    ko.get_synchronization_order()
    ko.get_phase_coherence()
    ko.get_history()


def test_spectral_model_gaps():
    # SpectralValidator gaps (from stats/spectral_model.py)
    from stats.spectral_model import SpectralValidator

    sv = SpectralValidator(n_levels=3)
    sv.validate_signal(np.random.normal(0, 1, 100))


def test_stats_maturity_gaps():
    # assess_overall_maturity gaps
    sig = np.random.normal(0, 1, 1000)
    levels = [np.random.normal(0, 1, 1000) for _ in range(2)]
    theta = [np.random.normal(1, 0.1, 1000) for _ in range(2)]
    phi = [np.random.uniform(0, 2 * np.pi, 1000) for _ in range(2)]
    pi = [np.random.uniform(1, 2, 1000) for _ in range(2)]
    score = assess_overall_maturity(
        sig, signal_levels=levels, theta_levels=theta, phi_levels=phi, pi_levels=pi
    )
    get_maturity_rating(95)
    get_maturity_rating(85)
    get_maturity_rating(75)
    get_maturity_rating(65)
    get_maturity_rating(55)
    get_maturity_rating(45)
    log_maturity_assessment(score)
    format_maturity_assessment(score)
    print_maturity_assessment(score)
