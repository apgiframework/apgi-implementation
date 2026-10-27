import warnings
from unittest.mock import patch

import numpy as np
import pytest

from core.config_schema import APGIConfig
from pipeline import APGIPipeline


def get_base_config():
    return APGIConfig().model_dump()


def test_pipeline_init_backward_compat():
    # Test beta_da and tau_sigma backward compatibility in pipeline init
    config = get_base_config()
    config.pop("beta", None)
    config.pop("ignite_tau", None)
    config["beta_da"] = 1.25
    config["tau_sigma"] = 0.7
    p = APGIPipeline(config)
    assert p.config["beta"] == 1.25
    assert p.config["ignite_tau"] == 0.7


def test_pipeline_init_non_strict_auto_adjust():
    # Test NE threshold instability auto-adjustment in non-strict mode
    config = get_base_config()
    config["ne_on_threshold"] = True
    config["ne_on_precision"] = False
    config["gamma_ne"] = 0.1
    config["kappa"] = 0.15
    config["strict_mode"] = False
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        p = APGIPipeline(config)
        assert any("threshold instability" in str(warning.message) for warning in w)
    # The warning is advisory only: the caller's parameters are left intact.
    # Silently rewriting gamma_ne/kappa meant a run could report parameters it
    # had not actually used.
    assert p.config["gamma_ne"] == 0.1
    assert p.config["kappa"] == 0.15
    assert p.config["ne_on_threshold"] is True


def test_pipeline_compute_errors_single_scale():
    # Test _compute_per_level_errors with use_hierarchical=False
    config = get_base_config()
    config["use_hierarchical"] = False
    p = APGIPipeline(config)
    z_e, z_i = p._compute_per_level_errors(0.5, 0.5)
    assert len(z_e) == 1
    assert len(z_i) == 1


def test_pipeline_reservoir_stochastic_ignition():
    # Test reservoir mode with stochastic ignition
    config = get_base_config()
    config["use_reservoir"] = True
    config["reservoir_as_threshold"] = True
    config["stochastic_ignition"] = True
    p = APGIPipeline(config)
    # Step multiple times to ensure we hit the stochastic path
    for _ in range(10):
        p.step(x_e=2.0, x_i=0.5)


def test_pipeline_reservoir_amplification():
    # Test reservoir suprathreshold amplification
    config = get_base_config()
    config["use_reservoir"] = True
    config["reservoir_amplification"] = 0.5
    p = APGIPipeline(config)
    p.step(x_e=2.0, x_i=0.5)  # High input to trigger amplification


def test_pipeline_resonance_ignition_reset():
    # Test resonance ignition reset logic
    config = get_base_config()
    config["hierarchical_mode"] = "full"
    config["use_resonance"] = True
    config["theta_0"] = 0.1  # Low threshold to trigger ignition
    p = APGIPipeline(config)
    # Force ignition via pipeline's own S (ODE-accumulated, not resonance.S)
    assert p.resonance_system is not None
    p.S = 10.0  # Pre-load high signal so the ODE step + threshold check fires
    res = p.step(x_e=10.0, x_i=10.0)
    assert res["B"] == 1
    # After ignition, pipeline S should be reset (reset_factor applied)
    assert p.S < 10.0


def test_pipeline_reset_factor_validation():
    # Test reset_factor validation in step
    config = get_base_config()
    p = APGIPipeline(config)
    p.config["reset_factor"] = 1.5  # Invalid
    with pytest.raises(ValueError, match=r"reset_factor.*must be in"):
        # Force ignition
        p.S = 10.0
        p.step(x_e=1.0, x_i=1.0)


def test_pipeline_config_passed_debug():
    # Test config_validation_passed debug log using mock
    from pipeline import logger as pipeline_logger

    config = get_base_config()
    config["strict_mode"] = True
    with patch.object(pipeline_logger, "debug") as mock_debug:
        APGIPipeline(config)
        mock_debug.assert_any_call("config_validation_passed", strict=True)


def test_pipeline_kuramoto_broadcast_ignition():
    # Test Kuramoto broadcast ignition reset
    config = get_base_config()
    config["use_kuramoto"] = True
    config["kuramoto_broadcast_ignition"] = True
    config["n_levels"] = 3
    p = APGIPipeline(config)
    # Force ignition
    p.resonance_system = None  # Avoid interference for this simple test
    p.S = 10.0
    p.theta = 0.1
    p.step(x_e=10.0, x_i=1.0)


def test_pipeline_bold_calibration():
    # Test BOLD calibration logic
    config = get_base_config()
    config["use_bold_calibration"] = True
    config["use_realistic_cost"] = True  # Needed for BOLD path
    config["enforce_landauer"] = True  # Trigger bold path in threshold.py
    # Seeded: with seed=None the SDE noise comes from the process-global generator,
    # so whether C = c1*S is positive depended on how many draws earlier tests made.
    config["seed"] = 0
    p = APGIPipeline(config)
    res = p.step(x_e=1.0, x_i=0.5)
    assert "C" in res
    assert res["C"] > 0
    # BOLD calibration adds bold_calibration dict to result with bold_signal_change
    assert "bold_calibration" in res
    assert "bold_signal_change" in res["bold_calibration"]


def test_pipeline_reservoir_energy_readout():
    # Test reservoir with energy readout
    config = get_base_config()
    config["use_reservoir"] = True
    config["reservoir_readout_method"] = "energy"
    p = APGIPipeline(config)
    p.step(x_e=1.0, x_i=0.5)


def test_pipeline_validate_method():
    # Test validate() method with realistic data to avoid Hurst errors
    config = get_base_config()
    p = APGIPipeline(config)
    # Insufficient data
    res = p.validate()
    assert res["status"] == "insufficient_data"
    # Fill history with some variability and enough points
    t = np.linspace(0, 100, 1000)
    p.history["theta"] = (1.0 + 0.1 * np.sin(t) + np.random.normal(0, 0.01, 1000)).tolist()
    res = p.validate()
    assert res["status"] == "success"


def test_pipeline_hierarchical_no_network():
    # Test use_hierarchical=True but hierarchical_network=None
    config = get_base_config()
    config["use_hierarchical"] = True
    config["use_hierarchical_precision_ode"] = False
    config["use_resonance"] = False
    p = APGIPipeline(config)
    p.step(x_e=1.0, x_i=0.5)


def test_pipeline_realistic_cost_false():
    # Test use_realistic_cost=False
    config = get_base_config()
    config["use_realistic_cost"] = False
    p = APGIPipeline(config)
    p.step(x_e=1.0, x_i=0.5)


def test_pipeline_active_inference_always():
    # Test active inference with ai_on_ignition_only=False
    config = get_base_config()
    config["use_active_inference"] = True
    config["ai_on_ignition_only"] = False
    config["ai_n_actions"] = 2
    config["ai_action_params"] = [[1, 0, 0], [0, 1, 0]]
    p = APGIPipeline(config)
    res = p.step(x_e=0.1, x_i=0.1)  # No ignition
    assert "ai_action_idx" in res


def test_pipeline_kuramoto_no_broadcast():
    # Test Kuramoto with broadcast_ignition=False (default)
    config = get_base_config()
    config["use_kuramoto"] = True
    config["kuramoto_broadcast_ignition"] = False
    p = APGIPipeline(config)
    # Force ignition
    p.S = 10.0
    p.theta = 0.1
    p.step(x_e=10.0, x_i=1.0)


def test_pipeline_ne_modulation_none():
    # Test when both ne_on_precision and ne_on_threshold are False
    config = get_base_config()
    config["ne_on_precision"] = False
    config["ne_on_threshold"] = False
    p = APGIPipeline(config)
    p.step(x_e=1.0, x_i=0.5)


def test_pipeline_reservoir_no_instance_error():
    # Test error when reservoir mode is on but reservoir is None
    config = get_base_config()
    config["use_reservoir"] = True
    config["reservoir_as_threshold"] = True
    p = APGIPipeline(config)
    p.reservoir = None  # Manually break it
    with pytest.raises(ValueError, match="reservoir not initialized"):
        p.step(x_e=1.0, x_i=0.5)


def test_pipeline_continuous_ode_no_drift():
    # Test use_continuous_threshold_ode=True but use_ode_refractory_drift=False
    config = get_base_config()
    config["use_continuous_threshold_ode"] = True
    config["use_ode_refractory_drift"] = False
    p = APGIPipeline(config)
    p.step(x_e=1.0, x_i=0.5)
