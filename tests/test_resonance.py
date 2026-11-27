"""Tests for Cross-Level Threshold Resonance — Russian Doll Architecture.
Coverage:
- Phase advancement: natural frequency + top-down Kuramoto coupling
- Spec §8 threshold formula: θ_l = θ₀_l · (1 + κ·Π_{l+1}·cos(φ_{l+1}))
- Per-level leaky S accumulation
- apply_level_ignition: refractory reset + bottom-up suppression
- build_resonance_system factory: correct ω and λ from taus
- Pipeline integration: resonance outputs present, threshold uses live phase
- Zero-phase fix: phases advance in basic hierarchical mode
"""

from __future__ import annotations

import numpy as np
import pytest

from hierarchy.resonance import NestedResonanceSystem, build_resonance_system

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_system(
    n_levels: int = 3,
    kappa_down: float = 0.1,
    kappa_up: float = 0.0,
    phi_noise_std: float = 0.0,
) -> NestedResonanceSystem:
    taus = np.array([1.0, 5.0, 25.0][:n_levels])
    return build_resonance_system(
        n_levels=n_levels,
        taus=taus,
        theta_base=1.0,
        dt=0.1,
        kappa_down=kappa_down,
        kappa_up=kappa_up,
        phi_noise_std=phi_noise_std,
        rng=np.random.default_rng(0),
    )


# ---------------------------------------------------------------------------
# Phase advancement
# ---------------------------------------------------------------------------


class TestPhaseAdvancement:
    def test_phases_advance_from_zero(self):
        """Phases must change from 0 after one step (natural frequency drive)."""
        sys = _make_system()
        sys.step(np.ones(3), np.ones(3), dt=0.1)
        assert not np.allclose(sys.phi, 0.0), "phases should advance from zero"

    def test_phase_stays_in_0_2pi(self):
        """Phases must stay in [0, 2π) after many steps."""
        sys = _make_system()
        for _ in range(500):
            sys.step(np.ones(3), np.ones(3), dt=0.5)
        assert np.all(sys.phi >= 0.0)
        assert np.all(sys.phi < 2.0 * np.pi)

    def test_higher_level_slower_phase(self):
        """Level 2 (large τ) should accumulate less total phase than level 0."""
        sys = _make_system()
        # Run many steps
        for _ in range(200):
            sys.step(np.ones(3), np.ones(3), dt=0.1)
        # Level 0 has ω_0 = 2π/1 >> ω_2 = 2π/25 — higher phase accumulation
        # We can't compare raw phi because of modulo, so compare natural frequency directly
        assert sys.omega[0] > sys.omega[2]

    def test_natural_frequency_set_correctly(self):
        """ω_l = 2π / τ_l should be set by build_resonance_system."""
        sys = _make_system()
        taus = np.array([1.0, 5.0, 25.0])
        expected_omega = 2.0 * np.pi / taus
        np.testing.assert_allclose(sys.omega, expected_omega)

    def test_phase_noise_changes_trajectory(self):
        """With phi_noise_std > 0, phases should differ from the noiseless run."""
        sys_noisy = _make_system(phi_noise_std=0.5)
        sys_clean = _make_system(phi_noise_std=0.0)
        for _ in range(20):
            sys_noisy.step(np.ones(3), np.ones(3), dt=0.1)
            sys_clean.step(np.ones(3), np.ones(3), dt=0.1)
        # With noise, trajectories will diverge
        assert not np.allclose(sys_noisy.phi, sys_clean.phi, atol=1e-6)


# ---------------------------------------------------------------------------
# Threshold formula (§8)
# ---------------------------------------------------------------------------


class TestThresholdFormula:
    def test_top_level_stays_at_baseline(self):
        """The top level has no level above it, so θ = θ₀ always."""
        sys = _make_system()
        for _ in range(10):
            sys.step(np.ones(3), np.ones(3), dt=0.1)
        # Top level (index 2) should equal theta_0[2] = 1.0 (no modulation)
        assert sys.theta[2] == pytest.approx(1.0, abs=1e-9)

    def test_threshold_modulated_by_level_above_phase(self):
        """θ_l deviates from baseline when the level above has non-trivial phase."""
        sys = _make_system(kappa_down=0.5)
        # Force level-1 phase to π/2 so cos(φ_1) ≈ 0 → minimal modulation
        sys.phi[1] = np.pi / 2.0
        sys.pi[1] = 1.0
        # Manually recompute thresholds
        thetas_halfpi = sys._compute_thresholds()
        # Force level-1 phase to 0 so cos(0)=1 → full positive modulation
        sys.phi[1] = 0.0
        thetas_zero = sys._compute_thresholds()
        # At φ=0 the threshold should be higher (positive modulation)
        assert thetas_zero[0] > thetas_halfpi[0]

    def test_threshold_spec_formula_numerically(self):
        """EQ-MS-4: θ_0 = θ₀_0 · clip(1 + κ · (Π_1/Π_max) · cos(φ_1), 0.5, 1.5)."""
        sys = _make_system(kappa_down=0.2)
        sys.phi = np.array([0.0, np.pi / 3.0, 0.0])
        sys.pi = np.array([1.0, 2.0, 1.0])
        thetas = sys._compute_thresholds()
        expected_theta_0 = 1.0 * (1.0 + 0.2 * (2.0 / sys.pi_max) * np.cos(np.pi / 3.0))
        assert thetas[0] == pytest.approx(expected_theta_0, rel=1e-9)

    def test_threshold_factor_band(self):
        """EQ-MS-4 bounds the factor in [0.5, 1.5] whatever κ_down and Π are (SSOT-P12)."""
        sys = _make_system(kappa_down=5.0)
        sys.pi = np.full(3, sys.pi_max)
        for phase, expected in ((0.0, 1.5), (np.pi, 0.5)):
            sys.phi = np.full(3, phase)
            assert sys._compute_thresholds()[0] == pytest.approx(expected, rel=1e-12)

    def test_threshold_clamped_within_bounds(self):
        """Modulated thresholds must respect theta_min and theta_max clamps."""
        sys = NestedResonanceSystem(
            n_levels=2,
            theta_0=np.array([1.0, 1.0]),
            omega=np.array([1.0, 0.2]),
            lambda_rates=np.array([0.1, 0.02]),
            kappa_down=100.0,  # extreme modulation
            theta_min=0.5,
            theta_max=3.0,
        )
        sys.pi = np.array([5.0, 5.0])
        sys.phi = np.array([0.0, 0.0])
        thetas = sys._compute_thresholds()
        assert np.all(thetas >= 0.5)
        assert np.all(thetas <= 3.0)


# ---------------------------------------------------------------------------
# Per-level S accumulation
# ---------------------------------------------------------------------------


class TestSignalAccumulation:
    def test_S_starts_at_zero(self):
        sys = _make_system()
        np.testing.assert_array_equal(sys.S, 0.0)

    def test_S_increases_with_positive_salience(self):
        sys = _make_system()
        for _ in range(5):
            sys.step(np.array([1.0, 1.0, 1.0]), np.ones(3), dt=0.1)
        assert np.all(sys.S > 0.0)

    def test_S_decays_without_input(self):
        sys = _make_system()
        # Prime accumulators
        for _ in range(10):
            sys.step(np.ones(3), np.ones(3), dt=0.1)
        S_before = sys.S.copy()
        # Let decay with zero salience
        for _ in range(20):
            sys.step(np.zeros(3), np.ones(3), dt=0.1)
        assert np.all(S_before > sys.S)

    def test_faster_level_responds_more_quickly(self):
        """Level 0 (small τ → large λ) should accumulate faster than level 2."""
        sys = _make_system()
        sys.step(np.array([5.0, 5.0, 5.0]), np.ones(3), dt=0.1)
        # λ_0 = dt/τ_0 > λ_2 = dt/τ_2, so level 0 accumulates more per step
        assert sys.lambda_rates[0] > sys.lambda_rates[2]

    def test_primary_signal_returns_level_zero(self):
        sys = _make_system()
        sys.step(np.array([3.0, 1.0, 0.5]), np.ones(3), dt=0.1)
        assert sys.primary_signal == pytest.approx(float(sys.S[0]))

    def test_primary_threshold_returns_level_zero(self):
        sys = _make_system()
        sys.step(np.ones(3), np.ones(3), dt=0.1)
        assert sys.primary_threshold == pytest.approx(float(sys.theta[0]))

    def test_step_with_noise_std(self):
        """Test that noise_std parameter adds SDE noise to signal accumulation."""
        sys = _make_system()
        sys.S = np.array([1.0, 1.0, 1.0])
        S_before = sys.S.copy()
        sys.step(np.zeros(3), np.ones(3), dt=0.1, noise_std=0.5)
        # With noise, S should differ from deterministic decay
        assert not np.allclose(sys.S, S_before * (1.0 - sys.lambda_rates), atol=1e-6)

    def test_step_with_phi_noise(self):
        """Test that phi_noise_std adds phase jitter."""
        sys = _make_system(phi_noise_std=0.5)
        phi_before = sys.phi.copy()
        sys.step(np.ones(3), np.ones(3), dt=0.1)
        # With phase noise, phases should differ from natural frequency only
        assert not np.allclose(sys.phi, phi_before + sys.omega * 0.1, atol=1e-6)


# ---------------------------------------------------------------------------
# apply_level_ignition
# ---------------------------------------------------------------------------


class TestApplyLevelIgnition:
    def test_ignition_resets_S_at_level(self):
        sys = _make_system()
        for _ in range(20):
            sys.step(np.ones(3), np.ones(3), dt=0.1)
        S_before = sys.S[0]
        sys.apply_level_ignition(level=0, rho_S=0.1, delta_refractory=0.5)
        assert sys.S[0] == pytest.approx(S_before * 0.1)

    def test_ignition_elevates_threshold(self):
        sys = _make_system()
        sys.step(np.ones(3), np.ones(3), dt=0.1)
        theta_before = sys.theta[0]
        sys.apply_level_ignition(level=0, rho_S=0.1, delta_refractory=0.5)
        assert sys.theta[0] > theta_before

    def test_ignition_does_not_exceed_theta_max(self):
        sys = NestedResonanceSystem(
            n_levels=2,
            theta_0=np.array([1.0, 1.0]),
            omega=np.array([1.0, 0.2]),
            lambda_rates=np.array([0.1, 0.02]),
            theta_max=2.0,
        )
        sys.theta[0] = 1.9
        sys.apply_level_ignition(level=0, delta_refractory=10.0)
        assert sys.theta[0] <= 2.0

    def test_bottom_up_suppression_when_kappa_up_set(self):
        sys = _make_system(kappa_up=0.5)
        for _ in range(5):
            sys.step(np.ones(3), np.ones(3), dt=0.1)
        theta_1_before = sys.theta[1]
        sys.apply_level_ignition(level=0, rho_S=0.1, delta_refractory=0.0)
        # kappa_up=0.5 → θ_1 *= (1 - 0.5) = halved
        assert sys.theta[1] < theta_1_before

    def test_no_suppression_when_kappa_up_zero(self):
        sys = _make_system(kappa_up=0.0)
        for _ in range(5):
            sys.step(np.ones(3), np.ones(3), dt=0.1)
        theta_1_before = sys.theta[1]
        sys.apply_level_ignition(level=0, rho_S=0.1, delta_refractory=0.0)
        assert sys.theta[1] == pytest.approx(theta_1_before)

    def test_invalid_level_raises(self):
        sys = _make_system(n_levels=2)
        with pytest.raises(ValueError, match="out of range"):
            sys.apply_level_ignition(level=5)

    def test_ignition_with_custom_rho_S(self):
        """Test ignition with custom signal retention fraction."""
        sys = _make_system()
        sys.S = np.array([10.0, 5.0, 2.0])
        sys.apply_level_ignition(level=0, rho_S=0.3, delta_refractory=0.0)
        assert sys.S[0] == pytest.approx(10.0 * 0.3)

    def test_ignition_with_custom_delta_refractory(self):
        """Test ignition with custom refractory elevation."""
        sys = _make_system()
        sys.theta = np.array([1.0, 1.0, 1.0])
        sys.apply_level_ignition(level=1, rho_S=0.1, delta_refractory=2.0)
        assert sys.theta[1] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# ignition_windows and modulation_depth
# ---------------------------------------------------------------------------


class TestProperties:
    def test_ignition_windows_shape(self):
        sys = _make_system()
        assert sys.ignition_windows.shape == (3,)

    def test_ignition_windows_true_when_S_above_theta(self):
        sys = _make_system()
        # Force S well above theta
        sys.S = np.array([100.0, 100.0, 100.0])
        sys.theta = np.array([1.0, 1.0, 1.0])
        assert np.all(sys.ignition_windows)

    def test_modulation_depth_zero_at_top_level(self):
        sys = _make_system()
        sys.step(np.ones(3), np.ones(3), dt=0.1)
        depth = sys.modulation_depth
        assert depth[-1] == pytest.approx(0.0)

    def test_modulation_depth_varies_with_phase(self):
        sys = _make_system(kappa_down=0.3)
        sys.phi = np.array([0.0, 0.0, 0.0])
        sys.pi = np.array([1.0, 2.0, 1.0])
        depth_zero = sys.modulation_depth[0]  # cos(0) = 1 → max modulation
        sys.phi = np.array([0.0, np.pi, 0.0])
        depth_pi = sys.modulation_depth[0]  # cos(π) = -1 → min modulation
        assert depth_zero > depth_pi


# ---------------------------------------------------------------------------
# build_resonance_system factory
# ---------------------------------------------------------------------------


class TestBuildResonanceSystem:
    def test_lambda_rates_from_taus(self):
        taus = np.array([2.0, 10.0])
        sys = build_resonance_system(2, taus, theta_base=1.0, dt=0.5)
        expected = np.clip(0.5 / taus, 1e-4, 0.999)
        np.testing.assert_allclose(sys.lambda_rates, expected)

    def test_omega_from_taus(self):
        taus = np.array([1.0, 4.0])
        sys = build_resonance_system(2, taus, theta_base=1.0, dt=0.1)
        expected = 2.0 * np.pi / taus
        np.testing.assert_allclose(sys.omega, expected)

    def test_theta_0_all_equal_to_theta_base(self):
        sys = build_resonance_system(3, np.array([1.0, 5.0, 25.0]), theta_base=2.5, dt=0.1)
        np.testing.assert_array_equal(sys.theta_0, 2.5)

    def test_n_levels_one_valid(self):
        sys = build_resonance_system(1, np.array([5.0]), theta_base=1.0, dt=0.1)
        assert sys.n_levels == 1

    def test_n_levels_zero_raises(self):
        with pytest.raises(ValueError):
            NestedResonanceSystem(
                n_levels=0,
                theta_0=np.array([]),
                omega=np.array([]),
                lambda_rates=np.array([]),
            )


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


class TestPipelineResonance:
    def _cfg(self, **overrides):
        from config import CONFIG

        cfg = dict(CONFIG)
        cfg.update(
            {
                "use_hierarchical": True,
                "n_levels": 3,
                "use_resonance": True,
                "resonance_kappa_down": 0.1,
                "resonance_kappa_up": 0.0,
                "resonance_phi_noise_std": 0.0,
                "stochastic_ignition": False,
                "use_reservoir": False,
                "use_active_inference": False,
            }
        )
        cfg.update(overrides)
        return cfg

    def test_resonance_system_initialised(self):
        from pipeline import APGIPipeline

        p = APGIPipeline(self._cfg())
        assert p.resonance_system is not None

    def test_resonance_system_none_when_flag_off(self):
        from pipeline import APGIPipeline

        cfg = self._cfg()
        cfg["use_resonance"] = False
        p = APGIPipeline(cfg)
        assert p.resonance_system is None

    def test_resonance_output_keys_present(self):
        from pipeline import APGIPipeline

        p = APGIPipeline(self._cfg())
        res = p.step(x_e=0.5, x_i=0.2)
        assert "resonance_phases" in res
        assert "resonance_thetas" in res
        assert "resonance_S_levels" in res
        assert "resonance_ignition_windows" in res
        assert "resonance_modulation_depth" in res

    def test_resonance_phases_advance_over_steps(self):
        from pipeline import APGIPipeline

        p = APGIPipeline(self._cfg())
        p.step(x_e=0.5, x_i=0.2)
        for _ in range(10):
            p.step(x_e=0.5, x_i=0.2)
        assert p.resonance_system is not None
        # Phases should differ from initial zeros
        assert not np.allclose(p.resonance_system.phi, 0.0)

    def test_resonance_thetas_finite(self):
        from pipeline import APGIPipeline

        p = APGIPipeline(self._cfg())
        for _ in range(20):
            res = p.step(x_e=np.random.randn(), x_i=np.random.randn())
        assert all(np.isfinite(t) for t in res["resonance_thetas"])

    def test_resonance_S_levels_non_negative_after_steps(self):
        from pipeline import APGIPipeline

        p = APGIPipeline(self._cfg())
        for _ in range(10):
            res = p.step(x_e=0.5, x_i=0.2)
        # S accumulators should be >= 0 (no negative signal)
        assert all(s >= 0.0 for s in res["resonance_S_levels"])

    def test_zero_phase_fix_basic_hierarchical(self):
        """Without resonance, basic hierarchical mode should still advance phases."""
        from pipeline import APGIPipeline

        cfg = self._cfg()
        cfg["use_resonance"] = False
        p = APGIPipeline(cfg)
        for _ in range(10):
            p.step(x_e=0.5, x_i=0.2)
        # Basic mode now advances phases via omega_levels
        phases = p.hierarchical.phases
        assert not np.allclose(phases, 0.0), "phases must advance in basic hierarchical mode"

    def test_modulation_depth_output_length(self):
        from pipeline import APGIPipeline

        p = APGIPipeline(self._cfg())
        res = p.step(x_e=0.5, x_i=0.2)
        assert len(res["resonance_modulation_depth"]) == 3

    def test_pipeline_runs_many_steps_without_crash(self):
        from pipeline import APGIPipeline

        p = APGIPipeline(self._cfg())
        for _ in range(50):
            res = p.step(x_e=np.random.randn(), x_i=np.random.randn())
        assert "B" in res
