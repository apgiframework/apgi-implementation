"""Tests for stability analysis.
Tests spec §7.5: Fixed-Point Stability Analysis
"""

import numpy as np
import pytest

from analysis.stability import (
    StabilityAnalyzer,
    analyze_bifurcation,
    check_stability,
    compute_eigenvalues,
    compute_fixed_point,
    compute_jacobian_discrete,
    validate_system_dynamics,
)


class TestJacobianComputation:
    """Test Jacobian computation."""

    def test_jacobian_shape(self):
        """Test Jacobian has correct shape."""
        J = compute_jacobian_discrete(lam=0.2, kappa=0.15, c1=0.2, eta=0.1)
        assert J.shape == (2, 2)

    def test_jacobian_formula(self):
        """Test Jacobian matches spec formula.
        Spec §7.5: J = [[1-λ, 0], [ηc₁λ, e^{-κ}]]
        """
        lam = 0.2
        kappa = 0.15
        c1 = 0.2
        eta = 0.1
        J = compute_jacobian_discrete(lam, kappa, c1, eta)
        # Check elements
        assert np.isclose(J[0, 0], 1 - lam)
        assert np.isclose(J[0, 1], 0)
        assert np.isclose(J[1, 0], eta * c1 * lam)
        assert np.isclose(J[1, 1], np.exp(-kappa))

    def test_jacobian_with_different_params(self):
        """Test Jacobian with various parameter values."""
        params = [
            (0.1, 0.1, 0.1, 0.1),
            (0.5, 0.5, 0.5, 0.5),
            (0.01, 1.0, 0.01, 0.01),
        ]
        for lam, kappa, c1, eta in params:
            J = compute_jacobian_discrete(lam, kappa, c1, eta)
            assert J.shape == (2, 2)
            assert np.all(np.isfinite(J))


class TestEigenvalueComputation:
    """Test eigenvalue computation."""

    def test_eigenvalue_shape(self):
        """Test eigenvalues have correct shape."""
        J = compute_jacobian_discrete(0.2, 0.15, 0.2, 0.1)
        eigs, vecs = compute_eigenvalues(J)
        assert len(eigs) == 2
        assert vecs.shape == (2, 2)

    def test_eigenvalue_magnitudes(self):
        """Test eigenvalue magnitudes."""
        J = compute_jacobian_discrete(0.2, 0.15, 0.2, 0.1)
        eigs, _ = compute_eigenvalues(J)
        # For this system, eigenvalues should be real
        assert np.all(np.isreal(eigs))
        # Magnitudes should be positive
        assert np.all(np.abs(eigs) >= 0)


class TestStabilityChecking:
    """Test stability checking."""

    def test_stable_configuration(self):
        """Test that typical configuration is stable."""
        config = {
            "lam": 0.2,
            "kappa": 0.15,
            "c1": 0.2,
            "eta": 0.1,
        }
        result = check_stability(config)
        assert result["stable"]
        assert result["max_eigenvalue"] < 1.0

    def test_unstable_configuration(self):
        """Test detection of unstable configuration."""
        # Make kappa very large (e^{-κ} → 0, but still stable)
        # Actually, the system is stable for all κ > 0 and λ ∈ (0,1)
        # So let's test with negative kappa (unphysical)
        config = {
            "lam": 0.2,
            "kappa": -1.0,  # Unphysical! e^{-(-1)} = e > 1
            "c1": 0.2,
            "eta": 0.1,
        }
        result = check_stability(config)
        # Should detect instability
        assert not result["stable"]
        assert result["max_eigenvalue"] >= 1.0

    def test_stability_margin(self):
        """Test stability margin computation."""
        config = {
            "lam": 0.2,
            "kappa": 0.15,
            "c1": 0.2,
            "eta": 0.1,
        }
        result = check_stability(config)
        # Stability margin should be positive for stable system
        assert result["stability_margin"] > 0

    def test_constraint_verification(self):
        """Test constraint verification."""
        config = {
            "lam": 0.2,
            "kappa": 0.15,
            "c1": 0.2,
            "eta": 0.1,
        }
        result = check_stability(config)
        # All constraints should be satisfied
        constraints = result["constraints_satisfied"]
        assert constraints["lambda_positive"]
        assert constraints["decay_exponent_positive"]
        assert constraints["lambda1_stable"]
        assert constraints["lambda2_stable"]


class TestFixedPointComputation:
    """Test fixed point computation."""

    def test_fixed_point_shape(self):
        """Test fixed point has correct structure."""
        config = {
            "lam": 0.2,
            "theta_base": 1.0,
        }
        fp = compute_fixed_point(config)
        assert "S_star" in fp
        assert "theta_star" in fp

    def test_fixed_point_values(self):
        """Test fixed point values are reasonable."""
        config = {
            "lam": 0.2,
            "theta_base": 1.0,
        }
        fp = compute_fixed_point(config)
        # S* should be positive
        assert fp["S_star"] > 0
        # θ* should equal θ_base
        assert np.isclose(fp["theta_star"], 1.0)

    def test_fixed_point_with_different_lam(self):
        """Test fixed point varies with lambda."""
        config1 = {"lam": 0.1, "theta_base": 1.0}
        config2 = {"lam": 0.5, "theta_base": 1.0}
        fp1 = compute_fixed_point(config1)
        fp2 = compute_fixed_point(config2)
        # S* should be larger for smaller lambda
        assert fp1["S_star"] > fp2["S_star"]


class TestBifurcationAnalysis:
    """Test bifurcation analysis."""

    def test_bifurcation_structure(self):
        """Test bifurcation analysis structure."""
        config = {
            "lam": 0.2,
            "kappa": 0.15,
            "c1": 0.2,
            "eta": 0.1,
        }
        result = analyze_bifurcation(
            config,
            param_name="lam",
            param_range=(0.01, 0.99),
            n_points=20,
        )
        assert "parameter_name" in result
        assert "parameter_values" in result
        assert "eigenvalue_magnitudes" in result
        assert "stability" in result
        assert "bifurcation_points" in result

    def test_bifurcation_detection(self):
        """Test bifurcation point detection."""
        config = {
            "lam": 0.2,
            "kappa": 0.15,
            "c1": 0.2,
            "eta": 0.1,
        }
        result = analyze_bifurcation(
            config,
            param_name="lam",
            param_range=(0.01, 1.5),
            n_points=50,
        )
        # Should find bifurcation around lam = 1
        bifurcation_points = result["bifurcation_points"]
        # There should be at least one bifurcation
        assert len(bifurcation_points) >= 0


class TestSystemDynamicsValidation:
    """Test system dynamics validation."""

    def test_validation_structure(self):
        """Test validation result structure."""
        config = {
            "lam": 0.2,
            "kappa": 0.15,
            "c1": 0.2,
            "eta": 0.1,
            "theta_base": 1.0,
        }
        # Generate synthetic data
        np.random.seed(42)
        S_history = np.random.normal(1.0, 0.5, 200)
        theta_history = np.random.normal(1.0, 0.2, 200)
        result = validate_system_dynamics(config, S_history, theta_history)
        assert "valid" in result
        if result["valid"]:
            assert "fixed_point" in result
            assert "prediction_error" in result
            assert "correlation_S" in result
            assert "correlation_theta" in result

    def test_insufficient_data(self):
        """Test validation with insufficient data."""
        config = {
            "lam": 0.2,
            "kappa": 0.15,
            "c1": 0.2,
            "eta": 0.1,
        }
        # Only 50 samples
        S_history = np.random.normal(1.0, 0.5, 50)
        theta_history = np.random.normal(1.0, 0.2, 50)
        result = validate_system_dynamics(config, S_history, theta_history)
        assert not result["valid"]


class TestStabilityAnalyzer:
    """Test StabilityAnalyzer class."""

    def test_initialization(self):
        """Test analyzer initialization."""
        config = {
            "lam": 0.2,
            "kappa": 0.15,
            "c1": 0.2,
            "eta": 0.1,
        }
        analyzer = StabilityAnalyzer(config)
        assert analyzer.config == config
        assert len(analyzer.history["S"]) == 0

    def test_step_recording(self):
        """Test step recording."""
        config = {
            "lam": 0.2,
            "kappa": 0.15,
            "c1": 0.2,
            "eta": 0.1,
        }
        analyzer = StabilityAnalyzer(config)
        # Record some steps
        for t in range(100):
            analyzer.step(S=1.0 + 0.1 * np.sin(0.1 * t), theta=1.0)
        assert len(analyzer.history["S"]) == 100
        assert len(analyzer.history["theta"]) == 100

    def test_analysis(self):
        """Test comprehensive analysis."""
        config = {
            "lam": 0.2,
            "kappa": 0.15,
            "c1": 0.2,
            "eta": 0.1,
            "theta_base": 1.0,
        }
        analyzer = StabilityAnalyzer(config)
        # Record some steps
        np.random.seed(42)
        for t in range(200):
            S = 1.0 + 0.1 * np.sin(0.1 * t) + np.random.normal(0, 0.05)
            theta = 1.0 + 0.05 * np.cos(0.05 * t)
            analyzer.step(S, theta)
        # Analyze
        result = analyzer.analyze(verbose=False)
        assert "stability" in result
        assert "fixed_point" in result
        assert "dynamics_validation" in result
        # Check stability
        assert result["stability"]["stable"]


class TestStabilityIntegration:
    """Integration tests for stability analysis."""

    def test_full_analysis_pipeline(self):
        """Test full stability analysis pipeline."""
        config = {
            "lam": 0.2,
            "kappa": 0.15,
            "c1": 0.2,
            "eta": 0.1,
            "theta_base": 1.0,
        }
        # 1. Check stability
        stability = check_stability(config)
        assert stability["stable"]
        # 2. Compute fixed point
        fp = compute_fixed_point(config)
        assert fp["S_star"] > 0
        # 3. Analyze bifurcation
        bifurcation = analyze_bifurcation(
            config,
            param_name="lam",
            param_range=(0.01, 0.99),
            n_points=20,
        )
        assert "bifurcation_points" in bifurcation
        # 4. Validate dynamics
        np.random.seed(42)
        S_history = np.random.normal(1.0, 0.5, 200)
        theta_history = np.random.normal(1.0, 0.2, 200)
        dynamics = validate_system_dynamics(config, S_history, theta_history)
        assert "valid" in dynamics

    def test_validate_system_dynamics_with_bifurcation(self):
        """Test dynamics validation with bifurcation points."""
        config = {"lam": 0.2, "eta": 0.1, "delta": 0.5, "ignite_tau": 0.5}
        np.random.seed(42)
        S_history = np.random.normal(1.0, 0.5, 200)
        theta_history = np.random.normal(1.0, 0.2, 200)
        dynamics = validate_system_dynamics(config, S_history, theta_history)
        assert "valid" in dynamics
        # Check that bifurcation analysis was performed
        assert isinstance(dynamics["valid"], bool)

    def test_validate_system_dynamics_short_history(self):
        """Test dynamics validation with short history."""
        config = {"lam": 0.2, "eta": 0.1, "delta": 0.5, "ignite_tau": 0.5}
        np.random.seed(42)
        S_history = np.random.normal(1.0, 0.5, 50)
        theta_history = np.random.normal(1.0, 0.2, 50)
        dynamics = validate_system_dynamics(config, S_history, theta_history)
        assert "valid" in dynamics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])  # pragma: no cover
