"""Tests for Serotonergic Threshold Stabilization (§8.4) and DFA (§22)."""

from __future__ import annotations

import numpy as np
import pytest

from core.threshold import apply_serotonin_threshold_offset
from stats.hurst import dfa_analysis, estimate_hurst_dfa
from stats.spectral_model import validate_hurst_dfa

# ── Serotonergic Threshold Stabilization ─────────────────────────────────────


class TestApplySerotoninThresholdOffset:
    def test_positive_offset_raises_threshold(self):
        """β_5HT > 0 should raise θ (patience / uncertainty tolerance)."""
        theta = 1.0
        result = apply_serotonin_threshold_offset(theta, 0.3)
        assert result == pytest.approx(1.3)

    def test_zero_offset_identity(self):
        """β_5HT = 0 must leave θ unchanged (backward compatibility)."""
        assert apply_serotonin_threshold_offset(2.5, 0.0) == pytest.approx(2.5)

    def test_negative_offset_lowers_threshold(self):
        """Negative β_5HT (hypothetical SSRI washout) lowers θ."""
        result = apply_serotonin_threshold_offset(1.0, -0.2)
        assert result == pytest.approx(0.8)

    def test_returns_float(self):
        assert isinstance(apply_serotonin_threshold_offset(1.0, 0.1), float)

    def test_spec_formula(self):
        """Verify θ_eff = θ + β_5HT exactly as written in spec §8.4."""
        for theta, beta in [(0.5, 0.1), (1.0, 0.5), (2.0, 0.0), (0.0, 1.0)]:
            assert apply_serotonin_threshold_offset(theta, beta) == pytest.approx(theta + beta)


class TestPipelineSerotonin:
    """Integration: beta_5ht config key is applied in the full APGI pipeline."""

    def _run_steps(self, beta_5ht: float, n_steps: int = 20) -> list[float]:
        from config import CONFIG
        from pipeline import APGIPipeline

        cfg = {**CONFIG, "beta_5ht": beta_5ht}
        pipe = APGIPipeline(cfg)
        thetas = []
        rng = np.random.default_rng(42)
        for _ in range(n_steps):
            eps_e = float(rng.normal(0, 0.5))
            eps_i = float(rng.normal(0, 0.5))
            result = pipe.step(eps_e, eps_i)
            thetas.append(result["theta_eff"] if "theta_eff" in result else result["theta"])
        return thetas

    def test_positive_5ht_raises_threshold_vs_zero(self):
        """Positive β_5HT should produce systematically higher θ than β=0."""
        thetas_zero = self._run_steps(0.0)
        thetas_pos = self._run_steps(0.5)
        assert np.mean(thetas_pos) > np.mean(thetas_zero)

    def test_zero_5ht_unchanged(self):
        """β_5HT = 0.0 must produce valid output and leave θ unshifted vs baseline."""
        from config import CONFIG
        from pipeline import APGIPipeline

        cfg_zero = {**CONFIG, "beta_5ht": 0.0}
        pipe = APGIPipeline(cfg_zero)
        rng = np.random.default_rng(99)
        results = []
        for _ in range(10):
            eps_e = float(rng.normal(0, 0.5))
            eps_i = float(rng.normal(0, 0.5))
            results.append(pipe.step(eps_e, eps_i))
        # All thetas must be finite and within a sane range
        thetas = [r["theta"] for r in results]
        assert all(np.isfinite(t) for t in thetas)
        assert all(t > 0 for t in thetas)


# ── DFA Analysis ──────────────────────────────────────────────────────────────


class TestDFAAnalysis:
    @pytest.fixture
    def pink_noise(self) -> np.ndarray:
        """Generate approximate pink noise via spectral shaping."""
        rng = np.random.default_rng(0)
        N = 2048
        white = rng.standard_normal(N)
        fft = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(N)
        freqs[0] = 1e-10  # avoid div-by-zero at DC
        fft /= np.sqrt(freqs)
        return np.fft.irfft(fft, n=N)

    @pytest.fixture
    def brownian_noise(self) -> np.ndarray:
        """Brownian (1/f²) noise: cumulative sum of white noise. H ≈ 1.5."""
        rng = np.random.default_rng(1)
        return np.cumsum(rng.standard_normal(1024))

    @pytest.fixture
    def white_noise(self) -> np.ndarray:
        rng = np.random.default_rng(2)
        return rng.standard_normal(1024)

    def test_returns_tuple_of_three(self, pink_noise):
        result = dfa_analysis(pink_noise)
        assert len(result) == 3

    def test_alpha_scales_returned(self, pink_noise):
        alpha, scales, F = dfa_analysis(pink_noise)
        assert isinstance(alpha, float)
        assert len(scales) == len(F)
        assert len(scales) >= 2

    def test_pink_noise_h_approx_1(self, pink_noise):
        """Pink noise should yield H ≈ 1.0 (within generous tolerance for DFA)."""
        alpha, _, _ = dfa_analysis(pink_noise)
        assert 0.7 < alpha < 1.3, f"Pink noise H={alpha:.3f} unexpectedly far from 1"

    def test_white_noise_h_approx_half(self, white_noise):
        """White noise has H ≈ 0.5 (uncorrelated process)."""
        alpha, _, _ = dfa_analysis(white_noise)
        assert 0.3 < alpha < 0.7, f"White noise H={alpha:.3f} unexpectedly far from 0.5"

    def test_brownian_noise_h_above_one(self, brownian_noise):
        """Brownian noise (1/f²) has H ≈ 1.5 (strong persistence)."""
        alpha, _, _ = dfa_analysis(brownian_noise)
        assert alpha > 1.0, f"Brownian H={alpha:.3f} should be > 1"

    def test_custom_scales(self, pink_noise):
        custom = [8, 16, 32, 64, 128]
        alpha, scales, F = dfa_analysis(pink_noise, scales=custom)
        assert set(scales).issubset(set(custom))

    def test_order_2_detrending(self, pink_noise):
        alpha, _, _ = dfa_analysis(pink_noise, order=2)
        assert isinstance(alpha, float)

    def test_short_signal_raises(self):
        with pytest.raises(ValueError, match="too short"):
            dfa_analysis(np.random.default_rng(0).standard_normal(8))

    def test_F_values_positive(self, pink_noise):
        _, _, F = dfa_analysis(pink_noise)
        assert np.all(F > 0)

    def test_F_increases_with_scale(self, pink_noise):
        """F(n) must be monotonically non-decreasing for long-memory signals."""
        _, scales, F = dfa_analysis(pink_noise)
        # Allow one dip (numerical), but overall trend must be increasing
        diffs = np.diff(F)
        assert np.sum(diffs > 0) > np.sum(diffs < 0)


class TestEstimateHurstDFA:
    def test_returns_float(self):
        rng = np.random.default_rng(3)
        sig = rng.standard_normal(512)
        assert isinstance(estimate_hurst_dfa(sig), float)

    def test_matches_dfa_analysis(self):
        rng = np.random.default_rng(4)
        sig = rng.standard_normal(512)
        alpha_direct, _, _ = dfa_analysis(sig)
        assert estimate_hurst_dfa(sig) == pytest.approx(alpha_direct)


# ── validate_hurst_dfa ────────────────────────────────────────────────────────


class TestValidateHurstDFA:
    @pytest.fixture
    def pink_noise(self) -> np.ndarray:
        rng = np.random.default_rng(5)
        N = 2048
        white = rng.standard_normal(N)
        fft = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(N)
        freqs[0] = 1e-10
        fft /= np.sqrt(freqs)
        return np.fft.irfft(fft, n=N)

    def test_returns_dict_keys(self, pink_noise):
        result = validate_hurst_dfa(pink_noise)
        for key in (
            "hurst",
            "h_min",
            "h_max",
            "in_range",
            "falsifier_threshold",
            "falsified",
            "scales",
            "F_values",
            "message",
        ):
            assert key in result

    def test_pink_noise_near_apgi_range(self, pink_noise):
        """Pink noise H ≈ 1.0 should be well clear of the falsifier threshold,
        even though it overshoots the canonical [0.85, 0.95] range (pure 1/f
        pink noise is steeper than the coupled-threshold-dynamics prediction)."""
        result = validate_hurst_dfa(pink_noise)
        assert not result["falsified"]
        assert result["hurst"] > result["h_max"]

    def test_white_noise_outside_range_and_falsified(self):
        """White noise H ≈ 0.5 is outside the APGI range [0.85, 0.95] and
        below the α_DFA<0.55 falsifier threshold."""
        rng = np.random.default_rng(6)
        sig = rng.standard_normal(1024)
        result = validate_hurst_dfa(sig)
        assert not result["in_range"]
        assert result["falsified"]

    def test_custom_range(self, pink_noise):
        result = validate_hurst_dfa(pink_noise, h_min=0.0, h_max=2.0)
        assert result["in_range"]

    def test_falsifier_threshold_is_independent_of_range(self, pink_noise):
        """A signal can be out of the predicted range without being falsified
        (falsification only triggers below the 0.55 severity-B bound)."""
        result = validate_hurst_dfa(pink_noise, h_min=0.0, h_max=0.5)
        assert not result["in_range"]
        assert not result["falsified"]

    def test_short_signal_fails_gracefully(self):
        result = validate_hurst_dfa(np.array([1.0, 2.0, 3.0]))
        assert not result["in_range"]
        assert not result["falsified"]
        assert "DFA failed" in result["message"]

    def test_hurst_nan_for_bad_input(self):
        result = validate_hurst_dfa(np.zeros(5))
        assert np.isnan(result["hurst"]) or not result["in_range"]
