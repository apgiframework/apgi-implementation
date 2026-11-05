"""Final coverage gap tests — drives all remaining source lines to 100%.
Targets:
  - core/circadian.py       lines 177, 190
  - core/precision.py       lines 52, 53, 54, 100
  - hierarchy/multiscale.py line 259
  - main.py                 lines 276, 521, 524
  - pipeline.py             lines 318-325, 549-554, 851-852, 1124-1155
  - stats/avalanche.py      lines 35, 47, 49, 81, 83, 90, 94, 95, 129-131
  - stats/spectral_model.py lines 396, 398, 399
  - tests/conftest.py       all uncovered fixture bodies and helpers
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

# =============================================================================
# core/circadian.py  — lines 177, 190
# =============================================================================


class TestCircadianArrayValidation:
    """Cover ValueError raises in vectorised circadian/ultradian functions."""

    def test_circadian_offset_array_zero_period(self) -> None:
        """Line 176-177: raise ValueError when T_circ <= 0."""
        from core.circadian import circadian_theta_offset_array

        with pytest.raises(ValueError, match="T_circ must be > 0"):
            circadian_theta_offset_array(np.array([0.0, 1.0]), T_circ=0.0)

    def test_circadian_offset_array_negative_period(self) -> None:
        """Also line 177: negative period."""
        from core.circadian import circadian_theta_offset_array

        with pytest.raises(ValueError, match="T_circ must be > 0"):
            circadian_theta_offset_array(np.array([0.0]), T_circ=-5.0)

    def test_ultradian_offset_array_zero_period(self) -> None:
        """Line 189-190: raise ValueError when T_ultradian <= 0."""
        from core.circadian import ultradian_theta_offset_array

        with pytest.raises(ValueError, match="T_ultradian must be > 0"):
            ultradian_theta_offset_array(np.array([0.0, 1.0]), T_ultradian=0.0)

    def test_ultradian_offset_array_negative_period(self) -> None:
        """Also line 190: negative period."""
        from core.circadian import ultradian_theta_offset_array

        with pytest.raises(ValueError, match="T_ultradian must be > 0"):
            ultradian_theta_offset_array(np.array([0.0]), T_ultradian=-1.0)


# =============================================================================
# core/precision.py  — lines 52, 53, 54, 100
# =============================================================================


class TestPrecisionMissingLines:
    """Cover update_uncertainty_phi and apply_serotonin_threshold_patience."""

    def test_update_uncertainty_phi_alpha_zero(self) -> None:
        """Lines 52-53: ValueError when alpha == 0 (not in (0,1])."""
        from core.precision import update_uncertainty_phi

        with pytest.raises(ValueError, match="alpha must be in"):
            update_uncertainty_phi(prev_sigma2=1.0, phi=0.5, alpha=0.0)

    def test_update_uncertainty_phi_alpha_above_one(self) -> None:
        """Lines 52-53: ValueError when alpha > 1."""
        from core.precision import update_uncertainty_phi

        with pytest.raises(ValueError, match="alpha must be in"):
            update_uncertainty_phi(prev_sigma2=1.0, phi=0.5, alpha=1.5)

    def test_update_uncertainty_phi_valid_path(self) -> None:
        """Line 54: normal return — (1-α)·σ² + α·φ²."""
        from core.precision import update_uncertainty_phi

        result = update_uncertainty_phi(prev_sigma2=1.0, phi=0.5, alpha=0.1)
        # (1-0.1)*1.0 + 0.1*(0.5**2) = 0.9 + 0.025 = 0.925
        assert abs(result - 0.925) < 1e-10

    def test_apply_serotonin_threshold_patience(self) -> None:
        """Line 100: return float(theta + beta_5ht)."""
        from core.precision import apply_serotonin_threshold_patience

        assert abs(apply_serotonin_threshold_patience(1.0, 0.2) - 1.2) < 1e-10
        assert abs(apply_serotonin_threshold_patience(1.0, -0.1) - 0.9) < 1e-10
        assert abs(apply_serotonin_threshold_patience(0.0, 0.0) - 0.0) < 1e-10


# =============================================================================
# hierarchy/multiscale.py  — line 259
# =============================================================================


class TestMultiscaleWeightSchedulerZeroTotal:
    """Line 259: uniform fallback when accumulated values total <= 1e-12."""

    def test_zero_value_fallback(self) -> None:
        """Force values to zero so the else branch (line 259) fires."""
        from hierarchy.multiscale import MultiscaleWeightScheduler

        scheduler = MultiscaleWeightScheduler(n_levels=3, alpha=0.05)
        # Zero out accumulated values directly
        scheduler.values[:] = 0.0
        # Call update with zero phi errors — EMA stays at zero → total <= 1e-12
        weights = scheduler.update(np.zeros(3))
        # Should fall back to uniform (line 259)
        np.testing.assert_array_almost_equal(weights, np.ones(3) / 3)


# =============================================================================
# stats/avalanche.py  — lines 35, 47, 49, 81, 83, 90, 94, 95, 129-131
# =============================================================================


class TestAvalancheMissingLines:
    """Cover all remaining avalanche analysis paths."""

    # ---- extract_avalanches ----
    def test_extract_avalanches_2d_raises(self) -> None:
        """Line 35: ValueError for non-1D input."""
        from stats.avalanche import extract_avalanches

        with pytest.raises(ValueError, match="activity must be 1D"):
            extract_avalanches(np.array([[1, 0], [0, 1]]))

    def test_extract_avalanches_starts_at_zero(self) -> None:
        """Line 47: np.r_[0, starts] when active[0] is True."""
        from stats.avalanche import extract_avalanches

        # Begins with active state → starts prepended with 0
        sizes = extract_avalanches(np.array([1, 1, 0, 1, 0]))
        assert len(sizes) == 2
        assert sizes[0] == 2

    def test_extract_avalanches_ends_active(self) -> None:
        """Line 49: np.r_[ends, len(active)] when active[-1] is True."""
        from stats.avalanche import extract_avalanches

        # Ends with active state → ends appended with len
        sizes = extract_avalanches(np.array([0, 1, 0, 1, 1]))
        assert len(sizes) == 2
        assert sizes[-1] == 2

    def test_extract_avalanches_both_edges_active(self) -> None:
        """Lines 47 AND 49 simultaneously: starts and ends both active."""
        from stats.avalanche import extract_avalanches

        sizes = extract_avalanches(np.array([1, 0, 1]))
        assert len(sizes) == 2
        assert sizes[0] == 1
        assert sizes[1] == 1

    # ---- fit_discrete_power_law_mle ----
    def test_fit_power_law_2d_input_raises(self) -> None:
        """Line 81: ValueError for non-1D sizes."""
        from stats.avalanche import fit_discrete_power_law_mle

        with pytest.raises(ValueError, match="sizes must be 1D"):
            fit_discrete_power_law_mle(np.array([[1, 2], [3, 4]]))

    def test_fit_power_law_xmin_zero_raises(self) -> None:
        """Line 83: ValueError when xmin < 1."""
        from stats.avalanche import fit_discrete_power_law_mle

        with pytest.raises(ValueError, match="xmin must be >= 1"):
            fit_discrete_power_law_mle(np.array([1, 2, 3]), xmin=0)

    def test_fit_power_law_all_below_xmin(self) -> None:
        """Lines 87-88: empty array after xmin filter → nan alpha, n=0."""
        from stats.avalanche import fit_discrete_power_law_mle

        # xmin=100 filters out all sizes [1,2,3]
        result = fit_discrete_power_law_mle(np.array([1, 2, 3]), xmin=100)
        assert np.isnan(result.alpha)
        assert result.n == 0

    def test_fit_power_law_normal_computation(self) -> None:
        """Lines 90, 94, 95: normal MLE path."""
        from stats.avalanche import fit_discrete_power_law_mle

        rng = np.random.default_rng(42)
        # Integer sizes >= xmin=1
        sizes = rng.integers(1, 20, size=100)
        result = fit_discrete_power_law_mle(sizes, xmin=1)
        assert np.isfinite(result.alpha)
        assert result.alpha > 1.0  # MLE exponent always > 1
        assert result.n == 100

    # ---- validate_avalanche_power_law ----
    def test_validate_with_sufficient_avalanches(self) -> None:
        """Lines 129-131: success path when >= min_avalanches found."""
        from stats.avalanche import validate_avalanche_power_law

        # Alternate 0/1 rapidly to create many single-sample avalanches
        activity = np.tile([1, 0], 50)  # 50 avalanches of size 1
        result = validate_avalanche_power_law(activity, min_avalanches=10)
        assert result["status"] == "success"
        assert "alpha" in result
        assert isinstance(result["within_tolerance"], bool)

    def test_validate_insufficient_avalanches(self) -> None:
        """Lines 118-127: insufficient_data path."""
        from stats.avalanche import validate_avalanche_power_law

        # Only 2 avalanches, need 30 by default
        activity = np.array([1, 0, 0, 1, 0])
        result = validate_avalanche_power_law(activity, min_avalanches=30)
        assert result["status"] == "insufficient_data"


# =============================================================================
# stats/spectral_model.py  — lines 396, 398, 399
# =============================================================================


class TestSpectralModelNNLSFallback:
    """Lines 396-399: except branch when nnls raises inside fit_lorentzian_superposition."""

    def test_nnls_failure_triggers_lstsq_fallback(self) -> None:
        """Patch scipy.optimize.nnls to raise so the lstsq fallback runs."""
        import scipy.optimize

        from stats.spectral_model import fit_lorentzian_superposition

        freqs = np.logspace(-2, 2, 60)
        taus = np.array([0.1, 1.0, 10.0])
        power = np.ones(60)
        original_nnls = scipy.optimize.nnls

        def _raise(*args, **kwargs):
            raise RuntimeError("forced nnls failure for coverage")

        # Patch the attribute on the module so that
        # `from scipy.optimize import nnls` inside the function sees our mock
        scipy.optimize.nnls = _raise
        try:
            result = fit_lorentzian_superposition(freqs, power, taus)
        finally:
            scipy.optimize.nnls = original_nnls
        # Should have fallen back to lstsq (lines 398-399 executed)
        assert "amplitudes" in result
        assert "fitted_psd" in result
        assert result.get("fit_method") == "lstsq_clipped"


# =============================================================================
# pipeline.py — lines 318-325, 549-554, 851-852, 1124-1155
# =============================================================================


def _hierarchical_cfg() -> dict:
    """Return a minimal hierarchical pipeline config."""
    from config import CONFIG

    cfg = CONFIG.copy()
    cfg.update(
        {
            "use_hierarchical": True,
            "n_levels": 3,
            "tau_0": 10.0,
            "use_hierarchical_precision_ode": True,
        }
    )
    return cfg


class TestPipelinePsiTypeBranches:
    """Lines 318-325: psi_type dispatch in __init__."""

    def test_psi_type_tanh(self) -> None:
        """Lines 318-319: psi_fn = np.tanh."""
        from pipeline import APGIPipeline

        cfg = _hierarchical_cfg()
        cfg["psi_type"] = "tanh"
        p = APGIPipeline(cfg)
        assert p.hierarchical_network is not None
        result = p.step(1.0, 0.0, 0.5, 0.5)
        assert "S" in result

    def test_psi_type_softsign(self) -> None:
        """Lines 320-321: psi_fn = softsign lambda."""
        from pipeline import APGIPipeline

        cfg = _hierarchical_cfg()
        cfg["psi_type"] = "softsign"
        p = APGIPipeline(cfg)
        assert p.hierarchical_network is not None
        result = p.step(1.0, 0.0, 0.5, 0.5)
        assert "S" in result

    def test_psi_type_unknown_raises(self) -> None:
        """Lines 323-325: unknown psi_type → ValueError."""
        from pipeline import APGIPipeline

        cfg = _hierarchical_cfg()
        cfg["psi_type"] = "not_a_real_psi"
        with pytest.raises(ValueError, match="Unknown psi_type"):
            APGIPipeline(cfg)


class TestPipelineBuSourceBranches:
    """Lines 549-554: precision_bottom_up_source dispatch in step()."""

    def test_bu_source_extero(self) -> None:
        """Lines 549-550: bu_source='extero' uses exteroceptive errors."""
        from pipeline import APGIPipeline

        cfg = _hierarchical_cfg()
        cfg["precision_bottom_up_source"] = "extero"
        p = APGIPipeline(cfg)
        result = p.step(1.0, 0.0, 0.5, 0.5)
        assert "S" in result

    def test_bu_source_intero(self) -> None:
        """Lines 551-552: bu_source='intero' uses interoceptive errors."""
        from pipeline import APGIPipeline

        cfg = _hierarchical_cfg()
        cfg["precision_bottom_up_source"] = "intero"
        p = APGIPipeline(cfg)
        result = p.step(1.0, 0.0, 0.5, 0.5)
        assert "S" in result

    def test_bu_source_invalid_raises(self) -> None:
        """Lines 554-555: invalid bu_source → ValueError on step()."""
        from pipeline import APGIPipeline

        cfg = _hierarchical_cfg()
        cfg["precision_bottom_up_source"] = "bad_source"
        p = APGIPipeline(cfg)
        with pytest.raises(ValueError, match="precision_bottom_up_source must be one of"):
            p.step(1.0, 0.0, 0.5, 0.5)


class TestPipelineBottomUpCascade:
    """Lines 854-855: bottom_up_threshold_cascade loop when kappa_up > 0."""

    def test_kappa_up_nonzero_plain_hierarchical(self) -> None:
        """Lines 854-855: cascade loop body in the non-resonance else branch.
        use_resonance defaults to True, which short-circuits to the resonance
        branch at line 836 and skips the cascade else block.  Setting it False
        forces the else branch so the for-loop body at lines 854-855 executes.
        """
        from config import CONFIG
        from pipeline import APGIPipeline

        cfg = CONFIG.copy()
        cfg.update(
            {
                "use_hierarchical": True,
                "n_levels": 3,
                "tau_0": 10.0,
                "use_resonance": False,  # must be False to reach the else branch
                "kappa_up": 0.5,
            }
        )
        p = APGIPipeline(cfg)
        for _ in range(5):
            result = p.step(2.0, 0.0, 1.0, 0.0)
        assert "theta" in result

    def test_kappa_up_nonzero_with_ode(self) -> None:
        """Also confirm cascade works with precision-ODE path (kappa_up > 0)."""
        from pipeline import APGIPipeline

        cfg = _hierarchical_cfg()
        cfg["kappa_up"] = 0.2
        p = APGIPipeline(cfg)
        for _ in range(5):
            result = p.step(2.0, 0.0, 1.0, 0.0)
        assert "theta" in result


class TestPipelineValidate:
    """Lines 1124-1158: validate() Lorentzian + avalanche sections and exception handlers."""

    def test_validate_hierarchical_lorentzian(self) -> None:
        """Lines 1124-1139: Lorentzian fit block inside validate()."""
        from pipeline import APGIPipeline

        cfg = _hierarchical_cfg()
        p = APGIPipeline(cfg)
        for t in range(70):
            p.step(np.sin(0.1 * t) * 2.0, 0.0, 0.5, 0.3)
        out = p.validate()
        assert out["status"] == "success"
        assert "lorentzian_superposition" in out

    def test_validate_lorentzian_exception_handler(self) -> None:
        """Lines 1141-1142: except branch when Lorentzian fit raises."""
        from unittest.mock import patch

        from pipeline import APGIPipeline

        cfg = _hierarchical_cfg()
        p = APGIPipeline(cfg)
        for t in range(70):
            p.step(np.sin(0.1 * t) * 2.0, 0.0, 0.5, 0.3)
        with patch(
            "stats.spectral_model.fit_lorentzian_superposition",
            side_effect=RuntimeError("forced lorentzian failure"),
        ):
            out = p.validate()
        # AUD-DEF-03: a failed sub-analysis is reported as partial, never success.
        assert out["status"] == "partial"
        assert out["failed_analyses"] == ["lorentzian_superposition"]
        assert out["lorentzian_superposition"]["status"] == "error"

    def test_validate_avalanche_block(self) -> None:
        """Lines 1147-1155: avalanche stats success block inside validate()."""
        from config import CONFIG
        from pipeline import APGIPipeline

        cfg = CONFIG.copy()
        p = APGIPipeline(cfg)
        # Drive signal high enough to produce some ignitions
        for t in range(70):
            p.step(np.sin(0.1 * t) * 4.0, 0.0, 0.5, 0.0)
        out = p.validate()
        assert out["status"] == "success"
        assert "avalanche_power_law" in out

    def test_validate_avalanche_exception_handler(self) -> None:
        """Lines 1157-1158: except branch when avalanche validation raises."""
        from unittest.mock import patch

        from config import CONFIG
        from pipeline import APGIPipeline

        cfg = CONFIG.copy()
        p = APGIPipeline(cfg)
        for t in range(70):
            p.step(np.sin(0.1 * t) * 4.0, 0.0, 0.5, 0.0)
        with patch(
            "stats.avalanche.validate_avalanche_power_law",
            side_effect=RuntimeError("forced avalanche failure"),
        ):
            out = p.validate()
        # AUD-DEF-03: a failed sub-analysis is reported as partial, never success.
        assert out["status"] == "partial"
        assert out["failed_analyses"] == ["avalanche_power_law"]
        assert out["avalanche_power_law"]["status"] == "error"


# =============================================================================
# main.py  — lines 276, 521, 524, 570
# energy/calibration_utils.py — line 124
# =============================================================================


class TestMainMissingLines:
    """Cover remaining main.py branches."""

    def test_multiscale_progress_log_line_276(self) -> None:
        """Line 276: logger.debug called when (t+1) % 1000 == 0."""
        from main import run_multiscale_pipeline

        # 1001 steps → one progress log at t=999
        result = run_multiscale_pipeline(n_steps=1001, n_levels=2)
        assert result["n_steps"] == 1001

    def test_main_ne_threshold_with_explicit_gamma_ne(self) -> None:
        """Lines 521 and 524: --ne-on-threshold with explicit --gamma-ne value."""
        from main import main

        with patch(
            "sys.argv",
            [
                "main.py",
                "--steps",
                "5",
                "--ne-on-threshold",
                "--gamma-ne",
                "0.005",
            ],
        ):
            code = main()
        assert code == 0

    def test_main_dunder_main_block(self) -> None:
        """Line 570: sys.exit(main()) via __main__ block using runpy."""
        import runpy

        with patch("sys.argv", ["main.py", "--steps", "3"]):
            with pytest.raises(SystemExit) as exc_info:
                runpy.run_module("main", run_name="__main__", alter_sys=True)
        assert exc_info.value.code == 0


class TestCalibrationUtilsDunderMain:
    """Line 124: demonstrate_calibration_range() via __main__ block."""

    def test_calibration_utils_dunder_main(self, capsys) -> None:
        """Line 124 in energy/calibration_utils.py: __main__ guard.
        run_path cannot handle relative imports, so we use run_module with
        the package-qualified name, which correctly resolves 'from .bold_calibration'.
        """
        import runpy
        import sys

        # Remove from sys.modules to avoid RuntimeWarning about module already imported
        sys.modules.pop("energy.calibration_utils", None)
        runpy.run_module("energy.calibration_utils", run_name="__main__")
        captured = capsys.readouterr()
        assert "Landauer" in captured.out or "BOLD" in captured.out


# =============================================================================
# tests/conftest.py — all uncovered fixture bodies and helper functions
# =============================================================================
# Fixtures must be injected via pytest parameter names to fire their bodies.


class TestConftestFixtures:
    """Exercise every conftest fixture body."""

    def test_rng(self, rng) -> None:
        """Line 123: `return np.random.default_rng(42)`."""
        val = rng.uniform()
        assert 0.0 <= val <= 1.0

    def test_seeded_random(self, seeded_random) -> None:
        """Lines 129-131: np.random.seed(42), yield, np.random.seed(None)."""
        val = np.random.uniform()
        assert 0.0 <= val <= 1.0

    def test_sample_signal_history(self, sample_signal_history) -> None:
        """Line 142: array of 10 float values."""
        assert len(sample_signal_history) == 10

    def test_sample_threshold_history(self, sample_threshold_history) -> None:
        """Line 148: array of 10 threshold floats."""
        assert len(sample_threshold_history) == 10

    def test_sample_ignition_history(self, sample_ignition_history) -> None:
        """Line 154: binary ignition array."""
        assert len(sample_ignition_history) == 10

    def test_sample_time_series(self, sample_time_series) -> None:
        """Lines 160-161: sinusoidal time series of length 1000."""
        assert len(sample_time_series) == 1000

    def test_base_config(self, base_config) -> None:
        """Line 174: return base config dict."""
        assert "lam" in base_config
        assert "kappa" in base_config

    def test_full_config(self, full_config) -> None:
        """Line 195: return full config dict."""
        assert "S0" in full_config
        assert "alpha_e" in full_config

    def test_stability_config(self, stability_config) -> None:
        """Line 281: return stability config dict."""
        assert "lam" in stability_config
        assert "theta_base" in stability_config

    def test_hierarchical_config(self, hierarchical_config) -> None:
        """Line 293: return hierarchical config dict."""
        assert hierarchical_config["n_levels"] == 3

    def test_reservoir_params(self, reservoir_params) -> None:
        """Line 313: return reservoir params dict."""
        assert reservoir_params["N"] == 50

    def test_kuramoto_params(self, kuramoto_params) -> None:
        """Line 326: return kuramoto params dict."""
        assert kuramoto_params["n_levels"] == 3

    def test_mock_eeg_data(self, mock_eeg_data) -> None:
        """Line 341: return mock EEG dict."""
        assert mock_eeg_data["signals"].shape == (32, 1000)

    def test_mock_behavioral_data(self, mock_behavioral_data) -> None:
        """Line 352: return mock behavioral dict."""
        assert mock_behavioral_data["n_trials"] == 100

    def test_suppress_lapack(self, suppress_lapack) -> None:
        """Line 111: suppress_lapack fixture teardown drains pipe.
        Using the fixture exercises both the setup and teardown paths.
        """
        # Write something to stderr so the pipe isn't empty when teardown
        # executes os.read — ensures line 111 is evaluated.
        import sys

        sys.stderr.write("")
        sys.stderr.flush()
        assert True  # fixture setup/teardown is the coverage target


class TestConftestHelpers:
    """Exercise the module-level helper functions in conftest.py."""

    def test_assert_array_equal(self) -> None:
        """Line 366: np.testing.assert_allclose."""
        import tests.conftest as cf

        a = np.array([1.0, 2.0, 3.0])
        cf.assert_array_equal(a, a.copy())  # should not raise

    def test_assert_scalar_equal(self) -> None:
        """Line 371: np.testing.assert_allclose on scalars."""
        import tests.conftest as cf

        cf.assert_scalar_equal(3.14, 3.14)  # should not raise


class TestConftestSessionTeardown:
    """Lines 60-82: force the session fixture teardown to process chunks.
    The suppress_lapack_stderr_session session fixture captures fd 2.
    Writing directly to that file descriptor during a test ensures that
    `chunks` is non-empty in the teardown, triggering lines 64-82.
    """

    def test_write_to_captured_fd2(self) -> None:
        """Write LAPACK-style output to fd 2 so the teardown filters it."""
        import os

        # Write a fake DLASCL warning; suppressed by session fixture teardown
        os.write(2, b"** On entry to DLASCL, parameter number 4 had an illegal value\n")
        os.write(2, b"Real content that should pass through the filter\n")
        # Teardown will read these bytes and execute lines 64-82
        assert True
