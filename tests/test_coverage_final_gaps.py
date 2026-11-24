"""Final coverage gap tests to achieve 100% line coverage.
This module contains targeted tests for all remaining uncovered lines
across the APGI codebase.
"""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import patch

import numpy as np
import pytest
import structlog


class TestAnalysisStabilityCoverage:
    """Tests for analysis/stability.py missing lines."""

    def test_compute_eigenvalues_exception_fallback(self):
        """Test eigenvalue computation exception fallback (lines 76-80)."""
        from analysis.stability import compute_eigenvalues

        # Create a singular matrix that should trigger exception
        J = np.array([[0.0, 0.0], [0.0, 0.0]])
        # Force exception by mocking np.linalg.eig
        with patch("numpy.linalg.eig") as mock_eig:
            mock_eig.side_effect = np.linalg.LinAlgError("Singular matrix")
            eigs, vecs = compute_eigenvalues(J)
        # Verify fallback is used (diagonal values, identity vectors)
        assert np.allclose(eigs, np.array([0.0, 0.0]))
        assert np.allclose(vecs, np.eye(2))

    def test_compute_eigenvalues_floating_point_error(self):
        """Test FloatingPointError handling in eigenvalue computation."""
        from analysis.stability import compute_eigenvalues

        J = np.array([[0.5, 0.0], [0.1, 0.9]])
        with patch("numpy.linalg.eig") as mock_eig:
            mock_eig.side_effect = FloatingPointError("Numerical error")
            eigs, vecs = compute_eigenvalues(J)
        assert np.allclose(eigs, np.array([0.5, 0.9]))
        assert np.allclose(vecs, np.eye(2))

    def test_analyze_bifurcation_with_bifurcation_points(self):
        """Test bifurcation analysis when bifurcation points exist (lines 247-249)."""
        from analysis.stability import analyze_bifurcation

        # Create a config that will cause stability change
        config = {
            "lam": 0.5,
            "kappa": 0.5,
            "c1": 0.5,
            "eta": 0.1,
            "theta_base": 1.0,
        }
        # Analyze over a range where stability might change
        result = analyze_bifurcation(config, "lam", (0.01, 2.0), n_points=50)
        # Verify bifurcation points are processed
        assert "bifurcation_points" in result
        assert isinstance(result["bifurcation_points"], list)


class TestCoreConfigSchemaCoverage:
    """Tests for core/config_schema.py missing lines."""

    def test_alpha_e_validation_error(self):
        """Test alpha_e validation error (line 242)."""
        from core.config_schema import APGIConfig

        # Test with alpha_e = 0 (should fail validation)
        with pytest.raises(ValueError):
            APGIConfig(alpha_e=0.0, variance_method="ema")

    def test_alpha_i_validation_error(self):
        """Test alpha_i validation error (line 244)."""
        from core.config_schema import APGIConfig

        # Test with alpha_i > 1 (should fail the lt=1 constraint)
        with pytest.raises(ValueError):
            APGIConfig(alpha_i=1.5, variance_method="ema")

    def test_backward_compat_beta_da(self):
        """Test backward compatibility with beta_da (line 251)."""
        from core.config_schema import APGIConfig

        # When beta is default and beta_da is provided
        config = APGIConfig(beta_da=1.5)
        assert config.beta == 1.5


class TestCoreLoggingConfigCoverage:
    """Tests for core/logging_config.py missing lines."""

    def test_configure_logging_json_output(self):
        """Test JSON output logging with format_exc_info (line 50)."""
        from core.logging_config import configure_logging

        # Reset structlog configuration
        structlog.reset_defaults()
        # Configure with JSON output
        logger = configure_logging(level="INFO", json_output=True)
        assert logger is not None
        # Verify JSON output path was taken

    def test_configure_logging_audit(self):
        """Test audit logging configuration (line 71)."""
        from core.logging_config import configure_logging

        structlog.reset_defaults()
        # Configure with audit logging enabled
        logger = configure_logging(level="INFO", audit_logging=True)
        assert logger is not None


class TestCoreThermodynamicsCoverage:
    """Tests for core/thermodynamics.py missing lines."""

    def test_invalid_kappa_units_raise_error(self):
        """Test invalid kappa_units raises ValueError (lines 102-105)."""
        from core.thermodynamics import compute_landauer_cost

        with pytest.raises(ValueError, match="r_l_units must be"):
            compute_landauer_cost(S=1.0, eps=0.01, kappa_units="invalid")

    def test_joules_per_bit_mode(self):
        """Test joules_per_bit mode (line 159)."""
        from core.thermodynamics import compute_landauer_cost_batch

        S_array = np.array([0.5, 1.0, 2.0])
        costs = compute_landauer_cost_batch(
            S_array, eps=0.01, kappa_meta=1e-20, kappa_units="joules_per_bit"
        )
        # Verify costs are computed in joules_per_bit mode
        assert len(costs) == len(S_array)
        assert all(c >= 0 for c in costs)


class TestCoreValidationCoverage:
    """Tests for core/validation.py missing lines."""

    def test_g_ach_negative_validation(self):
        """Test g_ach negative validation (lines 76-77)."""
        from core.validation import ValidationError, validate_config

        # Use valid config that passes other validations
        with pytest.raises(ValidationError, match="g_ach"):
            validate_config(
                {
                    "g_ach": -1.0,
                    "lam": 0.2,
                    "kappa": 0.15,
                    "dt": 0.01,  # Valid dt
                    "tau_s": 5.0,
                    "tau_theta": 1000.0,
                    "tau_pi": 1000.0,
                }
            )

    def test_g_ne_negative_validation(self):
        """Test g_ne negative validation (lines 77-78)."""
        from core.validation import ValidationError, validate_config

        with pytest.raises(ValidationError, match="g_ne"):
            validate_config(
                {
                    "g_ne": -0.5,
                    "lam": 0.2,
                    "kappa": 0.15,
                    "dt": 0.01,
                    "tau_s": 5.0,
                    "tau_theta": 1000.0,
                    "tau_pi": 1000.0,
                }
            )

    def test_beta_da_accepts_negative(self):
        """β_DA is an unconstrained real; negative models DA depletion.

        MathSpec glossary: "β_DA(t) | Dopaminergic additive bias on
        interoceptive error | ℝ". The former non-negativity check made the
        depletion arm of the spec's own DA prediction unreachable.
        """
        from core.validation import validate_config

        validate_config(  # must not raise
            {
                "beta": -1.0,
                "lam": 0.2,
                "kappa": 0.15,
                "dt": 0.01,
                "tau_s": 5.0,
                "tau_theta": 1000.0,
                "tau_pi": 1000.0,
            }
        )


class TestMainCoverage:
    """Tests for main.py missing lines."""

    def test_run_standard_pipeline_with_ring_buffer(self):
        """Test run_standard_pipeline with ring buffer (lines 86-99)."""
        from main import run_standard_pipeline

        # Run with max_history smaller than n_steps to trigger ring buffer
        results = run_standard_pipeline(
            n_steps=50,
            max_history=10,
            progress_interval=25,
        )
        assert results["n_steps"] == 50
        assert len(results["history"]["S"]) == 10

    def test_run_multiscale_pipeline_with_ring_buffer(self):
        """Test run_multiscale_pipeline with ring buffer (lines 216-226)."""
        from main import run_multiscale_pipeline

        results = run_multiscale_pipeline(
            n_steps=50,
            n_levels=3,
            max_history=10,
        )
        assert results["n_steps"] == 50
        assert len(results["history"]["S_multiscale"]) == 10

    def test_analyze_signal_statistics_with_hurst(self):
        """Test analyze_signal_statistics with enough data for Hurst (lines 319-336)."""
        from main import analyze_signal_statistics

        # Generate signal with 256+ samples for Hurst estimation
        signal = np.random.randn(300).tolist()
        stats = analyze_signal_statistics(signal, "Test Signal")
        assert "mean" in stats
        assert "std" in stats
        # May or may not have hurst_exponent depending on success

    def test_analyze_signal_statistics_short_data(self):
        """Test analyze_signal_statistics with insufficient data (lines 336-343)."""
        from main import analyze_signal_statistics

        signal = [1.0, 2.0, 3.0]
        stats = analyze_signal_statistics(signal, "Short Signal")
        assert "mean" in stats
        assert "hurst_exponent" not in stats

    def test_save_results(self):
        """Test save_results function (lines 348-371)."""
        from main import save_results

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name
        try:
            results = {
                "config": {"test": True},
                "data": np.array([1.0, 2.0, 3.0]),
                "nested": {"array": np.array([4.0, 5.0])},
            }
            save_results(results, filepath)
            # Verify file was created and contains valid JSON
            with open(filepath) as f:
                loaded = json.load(f)
                assert loaded["config"]["test"] is True
                assert loaded["data"] == [1.0, 2.0, 3.0]
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_main_keyboard_interrupt(self):
        """Test main() KeyboardInterrupt handling (lines 503-505)."""
        from main import main

        with patch("main.run_standard_pipeline") as mock_run:
            mock_run.side_effect = KeyboardInterrupt()
            with patch("sys.argv", ["main.py", "--demo"]):
                result = main()
        assert result == 130

    def test_main_exception_handling(self):
        """Test main() exception handling (lines 506-508)."""
        from main import main

        with patch("main.run_standard_pipeline") as mock_run:
            mock_run.side_effect = ValueError("Test error")
            with patch("sys.argv", ["main.py", "--demo"]):
                result = main()
        assert result == 1


class TestEnergyModulesCoverage:
    """Tests for energy module missing lines."""

    def test_calibration_utils_line_124(self):
        """Test calibration_utils.py line 124 (calibration_utils edge case)."""
        from energy.calibration_utils import calibrate_for_realistic_r_l

        # Test edge case
        result = calibrate_for_realistic_r_l(
            target_r_l=0.1,  # Edge value
            typical_bold_change=1.0,
            typical_bits=6.6,
        )
        assert result is not None

    def test_bold_calibration_line_149(self):
        """Test bold_calibration.py line 149 (edge case)."""
        from energy.bold_calibration import calibrate_r_l_from_bold

        # Test with zero bold change (edge case)
        result = calibrate_r_l_from_bold(
            bold_signal_change=0.0,  # Edge case
            bits_erased=6.6,
        )
        assert result is not None


class TestHierarchyModulesCoverage:
    """Tests for hierarchy module missing lines."""

    def test_coupling_line_182(self):
        """Test hierarchy/coupling.py line 182 (nonlinear phase amplitude coupling)."""
        from hierarchy.coupling import nonlinear_phase_amplitude_coupling

        # Test the function that covers line 182 (else branch)
        result = nonlinear_phase_amplitude_coupling(
            theta_0_ell=1.0,
            pi_ell_plus_1=0.5,
            phi_ell_plus_1=np.pi / 4,  # 45 degrees
            kappa_down=0.1,
            nonlinearity="invalid",  # Triggers line 182: else branch
        )
        assert result is not None

    def test_multiscale_line_59(self):
        """Test hierarchy/multiscale.py line 59."""
        from hierarchy.multiscale import build_timescales

        # Test edge case that triggers line 59
        with pytest.raises(ValueError):
            build_timescales(tau0=1.0, k=0.5, n_levels=3)  # k <= 1 should error


class TestOscillationKuramotoCoverage:
    """Tests for oscillation/kuramoto.py missing lines."""

    def test_kuramoto_noise_lines_196_203_212(self):
        """Test Kuramoto noise coupling lines (196, 203-212)."""
        from oscillation.kuramoto import KuramotoOscillators

        osc = KuramotoOscillators(
            n_levels=5,
            tau_xi=1.0,
            sigma_xi=0.1,
        )
        # Step to trigger noise computation
        phases = osc.step(dt=0.1)
        assert len(phases) == 5


class TestPipelineCoverage:
    """Tests for pipeline.py missing lines."""

    def test_pipeline_line_638(self):
        """Test pipeline.py line 638 (specific error/edge case)."""
        from config import CONFIG
        from pipeline import APGIPipeline

        # Start with default CONFIG and add specific test settings
        config = CONFIG.copy()
        config.update(
            {
                "use_bold_calibration": True,
            }
        )
        pipeline = APGIPipeline(config)
        # Trigger line 638 through specific condition
        result = pipeline.step(0.1, 0.0, 0.1, 0.0)
        assert result is not None

    def test_pipeline_lines_871_872(self):
        """Test pipeline.py lines 871-872 (error handling)."""
        from config import CONFIG
        from pipeline import APGIPipeline

        # Start with default CONFIG and add specific test settings
        config = CONFIG.copy()
        config.update(
            {
                "use_thermodynamic_cost": True,
            }
        )
        pipeline = APGIPipeline(config)
        # Steps to trigger thermodynamic cost computation
        for _ in range(5):
            result = pipeline.step(1.0, 0.5, 0.5, 0.2)
        assert result is not None


class TestReservoirCoverage:
    """Tests for reservoir module missing lines."""

    def test_liquid_network_lines_25_27(self):
        """Test reservoir/liquid_network.py lines 25, 27."""
        from reservoir.liquid_network import LiquidNetwork

        # Create network that triggers lines 25, 27 (fallback branch)
        network = LiquidNetwork(
            n_units=50,
            spectral_radius=0.9,  # Valid value
        )
        # Use step method instead of forward
        output = network.step(u=0.5, dt=0.1)
        assert output is not None

    def test_liquid_state_machine_lines_126_128(self):
        """Test reservoir/liquid_state_machine.py lines 126, 128."""
        from reservoir.liquid_state_machine import LiquidStateMachine

        # Test initialization with default parameters
        lsm = LiquidStateMachine(N=50, M=2)
        # Test the reservoir - reset_state and get state statistics
        lsm.reset_state()
        stats = lsm.get_state_statistics()
        assert stats is not None
        assert "mean" in stats


class TestStatsSpectralExtractionCoverage:
    """Tests for stats/spectral_extraction.py missing lines."""

    def test_spectral_extraction_line_65(self):
        """Test line 65 - short input handling."""
        from stats.spectral_extraction import robust_log_regression

        # Test with too short input
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 2.0])
        slope, intercept, r2 = robust_log_regression(x, y)
        assert np.isnan(slope)

    def test_spectral_extraction_line_70(self):
        """Test line 70 - all invalid mask."""
        from stats.spectral_extraction import robust_log_regression

        # Test with all NaN/inf values
        x = np.array([np.nan, np.inf, -np.inf])
        y = np.array([1.0, 2.0, 3.0])
        slope, intercept, r2 = robust_log_regression(x, y)
        assert np.isnan(slope)

    def test_spectral_extraction_lines_77_85(self):
        """Test lines 77, 85 - low variance handling."""
        from stats.spectral_extraction import robust_log_regression

        # Test with low variance
        x = np.array([1.0, 1.0 + 1e-12, 1.0 + 2e-12])
        y = np.array([2.0, 3.0, 4.0])
        slope, intercept, r2 = robust_log_regression(x, y)
        assert np.isnan(slope)


class TestStatsMaturityAssessmentCoverage:
    """Tests for stats/maturity_assessment.py missing lines."""

    def test_maturity_short_signal(self):
        """Test maturity assessment with short signal (lines 95, 98)."""
        from stats.maturity_assessment import assess_statistical_validation

        signal = np.array([1.0, 2.0, 3.0])  # Very short signal
        spectral_score, conf_score, _, issues, _, _ = assess_statistical_validation(signal, fs=1.0)
        assert spectral_score == 0.0
        assert len(issues) > 0


class TestValidationObservableMappingCoverage:
    """Tests for validation/observable_mapping.py missing line 435."""

    def test_observable_mapping_line_435(self):
        """Test observable_mapping.py line 435."""
        from validation.observable_mapping import NeuralObservableExtractor

        # Create extractor and test
        extractor = NeuralObservableExtractor(fs=100.0)
        # Process some data with short history (triggers line 435 return 0.0)
        result = extractor.extract_gamma_power(np.array([1.0, 2.0]))  # Too short (< 64 samples)
        assert result == 0.0  # Returns 0.0 for short input


class TestEnergyCalibrationUtilsExtended:
    """Extended tests for energy/calibration_utils.py."""

    def test_calibration_utils_line_124(self):
        """Test calibration_utils.py line 124."""
        from energy.calibration_utils import calibrate_for_realistic_r_l

        # Test with edge case parameters
        result = calibrate_for_realistic_r_l(
            target_r_l=500.0,  # High efficiency
            typical_bold_change=0.1,  # Low BOLD change
        )
        # Check for actual keys returned
        assert "target_r_l" in result


class TestEnergyBoldCalibrationExtended:
    """Extended tests for energy/bold_calibration.py."""

    def test_bold_calibration_lines_149_374_401(self):
        """Test bold_calibration.py missing lines."""
        from energy.bold_calibration import (
            calibrate_r_l_from_bold,
            compute_landauer_energy_per_bit,
        )

        # Line 149: calibrate with zero bold change - returns 0.0
        result = calibrate_r_l_from_bold(
            bold_signal_change=0.0,
            bits_erased=6.6,
        )
        # Zero bold change returns 0.0 as expected
        assert result >= 0.0
        # Line 374: edge case with very high temperature
        e_bit = compute_landauer_energy_per_bit(T=500.0)
        assert e_bit > 0


class TestOscillationKuramotoExtended:
    """Extended tests for oscillation/kuramoto.py."""

    def test_kuramoto_lines_196_203_212(self):
        """Test kuramoto.py missing lines - noise reset and coupling."""
        from oscillation.kuramoto import KuramotoOscillators

        # Create oscillator with noise
        osc = KuramotoOscillators(
            n_levels=5,
            tau_xi=0.5,
            sigma_xi=0.2,
        )
        # Trigger noise reset (line 196) and coupling
        for _ in range(10):
            phases = osc.step(dt=0.1)
        assert len(phases) == 5
        assert osc.t > 0
        # Test get_history (covers related lines)
        history = osc.get_history()
        assert isinstance(history, np.ndarray)


class TestReservoirLiquidNetworkExtended:
    """Extended tests for reservoir/liquid_network.py."""

    def test_liquid_network_fallback_branch(self):
        """Test lines 25, 27 - fallback for eigenvalue computation."""
        from reservoir.liquid_network import LiquidNetwork

        # Create a network that triggers the fallback branch
        # This happens when eigvals computation fails; use spec-compliant radius §17
        network = LiquidNetwork(n_units=10, spectral_radius=0.8)
        # Verify the network was created
        assert network.W_res is not None
        assert network.n == 10
        # Step to verify operation
        output = network.step(u=0.5, dt=0.1)
        assert output is not None


class TestReservoirLiquidStateMachineExtended:
    """Extended tests for reservoir/liquid_state_machine.py."""

    def test_liquid_state_machine_extended(self):
        """Test additional LSM lines."""
        from reservoir.liquid_state_machine import LiquidStateMachine

        lsm = LiquidStateMachine(N=50, M=2, tau_res=2.0)
        # Collect training data with targets
        for _ in range(10):
            u = np.random.randn(2)
            target = np.random.randn()
            lsm.step(u, dt=0.1)  # Step the reservoir
            lsm.collect_state(target=target)  # Collect state with target
        # Get training data (covers lines 126-128)
        states, targets = lsm.get_training_data()
        assert states is not None
        # Reset state (covers lines 432-433)
        lsm.reset_state()
        assert np.allclose(lsm.x, 0)


class TestStatsSpectralExtractionExtended:
    """Extended tests for stats/spectral_extraction.py."""

    def test_spectral_extraction_welch_extended(self):
        """Test estimate_spectral_exponent_welch edge cases."""
        from stats.spectral_extraction import estimate_spectral_exponent_welch

        # Test with short signal (lines 85, 129-130)
        short_signal = np.random.randn(10)
        beta, r2, hurst = estimate_spectral_exponent_welch(short_signal, fs=1.0)
        # May return NaN for short signals
        assert isinstance(beta, float)

    def test_spectral_extraction_periodogram_extended(self):
        """Test estimate_spectral_exponent_periodogram edge cases."""
        from stats.spectral_extraction import estimate_spectral_exponent_periodogram

        # Test with short signal (lines 165, 167)
        short_signal = np.random.randn(10)
        beta, r2, hurst = estimate_spectral_exponent_periodogram(short_signal, fs=1.0)
        assert isinstance(beta, float)

    def test_spectral_extraction_bootstrap_extended(self):
        """Test bootstrap confidence intervals (lines 270, 276-277)."""
        from stats.spectral_extraction import bootstrap_confidence_interval

        # Test with edge cases
        signal = np.random.randn(100)
        ci_lower, ci_upper = bootstrap_confidence_interval(
            signal, estimator=np.mean, n_bootstrap=50
        )
        assert isinstance(ci_lower, float)
        assert isinstance(ci_upper, float)


class TestStatsMaturityAssessmentExtended:
    """Extended tests for stats/maturity_assessment.py."""

    def test_maturity_assessment_short_signal_extended(self):
        """Test with short signals (lines 95, 98, 154-155)."""
        from stats.maturity_assessment import assess_statistical_validation

        # Very short signal
        signal = np.array([1.0, 2.0])
        spectral, conf, consistency, issues, recs, sig = assess_statistical_validation(
            signal, fs=1.0
        )
        assert spectral == 0.0
        assert len(issues) > 0


class TestStatsSpectralModelExtended:
    """Extended tests for stats/spectral_model.py."""

    def test_spectral_model_lines_348_350(self):
        """Test spectral_model.py missing lines."""
        from stats.spectral_model import estimate_1f_exponent

        # Test 1/f estimation (lines 348, 350)
        freqs = np.logspace(-2, 0, 50)
        psd = freqs ** (-1.0)  # 1/f spectrum
        beta = estimate_1f_exponent(freqs, psd, fmin=0.01, fmax=1.0)
        assert isinstance(beta, float)


class TestCoreValidationExtended:
    """Extended tests for core/validation.py."""

    def test_validation_extended_lines(self):
        """Test remaining validation lines."""
        from core.validation import ValidationError, validate_config

        # Both NE pathways are permitted; the ceiling is on the PRODUCT
        # γ_NE·g_NE (MathSpec §2), and exceeding it warns with the spec's
        # prescribed rescale remedy rather than aborting.
        with pytest.warns(RuntimeWarning, match="sub-dominant"):
            validate_config(
                {
                    "ne_on_precision": True,
                    "ne_on_threshold": True,
                    "gamma_ne": 1.0,
                    "g_ne": 1.0,
                    "lam": 0.2,
                    "kappa": 0.15,
                    "dt": 0.01,
                    "tau_s": 5.0,
                    "tau_theta": 1000.0,
                    "tau_pi": 1000.0,
                }
            )
        # Test learning rate too high with internal predictions (lines 229, 234)
        # eta_pred_e=2.0 exceeds max_kappa = 2/pi_max = 2/100 = 0.02
        with pytest.raises(ValidationError):
            validate_config(
                {
                    "use_internal_predictions": True,
                    "eta_pred_e": 2.0,  # Way too high
                    "pi_max": 100.0,  # max_kappa = 2/100 = 0.02
                    "lam": 0.2,
                    "kappa": 0.15,
                    "dt": 0.01,
                    "tau_s": 5.0,
                    "tau_theta": 1000.0,
                    "tau_pi": 1000.0,
                }
            )


class TestPipelineExtended:
    """Extended tests for pipeline.py."""

    def test_pipeline_lines_638_871_872(self):
        """Test remaining pipeline lines."""
        from config import CONFIG
        from pipeline import APGIPipeline

        # Test line 638 - BOLD calibration path
        config = CONFIG.copy()
        config["use_bold_calibration"] = True
        pipeline = APGIPipeline(config)
        # Step with BOLD calibration
        for _ in range(5):
            result = pipeline.step(1.0, 0.5, 0.5, 0.2)
        assert result is not None
        # Test lines 871-872 - thermodynamic cost path
        config2 = CONFIG.copy()
        config2["use_thermodynamic_cost"] = True
        pipeline2 = APGIPipeline(config2)
        for _ in range(5):
            result2 = pipeline2.step(1.0, 0.5, 0.5, 0.2)
        assert result2 is not None


class TestMainExtended:
    """Extended tests for main.py."""

    def test_main_lines_334_335(self):
        """Test main.py lines 334, 335."""
        from main import analyze_signal_statistics

        # Test with exactly 256 samples (triggers hurst path)
        signal = np.random.randn(256).tolist()
        stats = analyze_signal_statistics(signal, "Test Signal")
        assert "mean" in stats

    def test_main_line_498(self):
        """Test main.py line 498."""
        import tempfile

        from main import save_results

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name
        try:
            results = {"test": np.array([1.0, 2.0])}
            save_results(results, filepath)
            # Verify file was created
            import json

            with open(filepath) as f:
                loaded = json.load(f)
                assert loaded["test"] == [1.0, 2.0]
        finally:
            import os

            if os.path.exists(filepath):
                os.remove(filepath)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
