"""Comprehensive unit tests for analysis/stability.py module.
Tests cover:
- compute_jacobian_discrete function
- compute_eigenvalues function
- check_stability function
- compute_fixed_point function
- analyze_bifurcation function
- validate_system_dynamics function
- StabilityAnalyzer class
"""

from __future__ import annotations

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


class TestComputeJacobianDiscrete:
    """Tests for compute_jacobian_discrete function."""

    def test_basic_computation(self):
        """Should compute Jacobian correctly."""
        result = compute_jacobian_discrete(
            lam=0.2,
            kappa=0.15,
            c1=0.2,
            eta=0.1,
        )
        expected = np.array(
            [
                [1.0 - 0.2, 0.0],
                [0.1 * 0.2 * 0.2, np.exp(-0.15)],
            ]
        )
        np.testing.assert_array_almost_equal(result, expected)

    def test_diagonal_values(self):
        """Should have correct diagonal values."""
        result = compute_jacobian_discrete(
            lam=0.3,
            kappa=0.2,
            c1=0.2,
            eta=0.1,
        )
        assert result[0, 0] == 0.7  # 1 - 0.3
        assert pytest.approx(result[1, 1], rel=1e-7) == np.exp(-0.2)


class TestComputeEigenvalues:
    """Tests for compute_eigenvalues function."""

    def test_diagonal_matrix(self):
        """Should return diagonal elements as eigenvalues."""
        J = np.array([[2.0, 0.0], [0.0, 3.0]])
        eigs, vecs = compute_eigenvalues(J)
        assert len(eigs) == 2
        assert pytest.approx(eigs[0], abs=1e-10) in [2.0, 3.0]
        assert pytest.approx(eigs[1], abs=1e-10) in [2.0, 3.0]


class TestCheckStability:
    """Tests for check_stability function."""

    def test_stable_config(self):
        """Should return stable for valid config."""
        config = {"lam": 0.2, "kappa": 0.15, "c1": 0.2, "eta": 0.1}
        result = check_stability(config)
        assert result["stable"] is True
        assert result["max_eigenvalue"] < 1.0

    def test_constraints_satisfied(self):
        """Should check all constraints."""
        config = {"lam": 0.2, "kappa": 0.15, "c1": 0.2, "eta": 0.1}
        result = check_stability(config)
        assert result["constraints_satisfied"]["lambda_positive"] is True
        assert result["constraints_satisfied"]["decay_exponent_positive"] is True

    def test_verbose_output(self, capsys):
        """Should print detailed output when verbose=True."""
        config = {"lam": 0.2, "kappa": 0.15, "c1": 0.2, "eta": 0.1}
        check_stability(config, verbose=True)  # noqa: F841
        captured = capsys.readouterr()
        assert "APGI Fixed-Point Stability Analysis" in captured.out
        assert "Parameters:" in captured.out
        assert "Jacobian" in captured.out
        assert "Eigenvalues:" in captured.out
        assert "Stability:" in captured.out

    def test_verbose_shows_lambda_values(self, capsys):
        """Verbose output should show lambda values."""
        config = {"lam": 0.2, "kappa": 0.15, "c1": 0.2, "eta": 0.1}
        check_stability(config, verbose=True)
        captured = capsys.readouterr()
        assert "λ (integration rate)" in captured.out
        assert "d = γ_θ·dt (decay exp.)" in captured.out

    def test_verbose_shows_constraint_verification(self, capsys):
        """Verbose output should show constraint verification."""
        config = {"lam": 0.2, "kappa": 0.15, "c1": 0.2, "eta": 0.1}
        check_stability(config, verbose=True)
        captured = capsys.readouterr()
        assert "Constraint Verification" in captured.out
        assert "λ > 0" in captured.out
        assert "d > 0" in captured.out


class TestComputeFixedPoint:
    """Tests for compute_fixed_point function."""

    def test_basic_computation(self):
        """Should compute fixed point."""
        config = {"lam": 0.2, "theta_base": 1.0}
        result = compute_fixed_point(config)
        assert "S_star" in result
        assert "theta_star" in result
        assert result["theta_star"] == 1.0


class TestAnalyzeBifurcation:
    """Tests for analyze_bifurcation function."""

    def test_bifurcation_analysis(self):
        """Should analyze bifurcation."""
        config = {"lam": 0.2, "kappa": 0.15, "c1": 0.2, "eta": 0.1}
        result = analyze_bifurcation(
            config,
            param_name="lam",
            param_range=(0.1, 0.5),
            n_points=10,
        )
        assert "parameter_name" in result
        assert "parameter_values" in result
        assert "stability" in result
        assert result["parameter_name"] == "lam"


class TestValidateSystemDynamics:
    """Tests for validate_system_dynamics function."""

    def test_validation_with_data(self):
        """Should validate dynamics."""
        config = {"lam": 0.2, "kappa": 0.15, "c1": 0.2, "eta": 0.1}
        S_history = np.random.normal(1.0, 0.1, 200)
        theta_history = np.random.normal(1.0, 0.1, 200)
        result = validate_system_dynamics(config, S_history, theta_history)
        assert result["valid"] is True
        assert "correlation_S" in result
        assert "correlation_theta" in result

    def test_insufficient_data(self):
        """Should return invalid for insufficient data."""
        config = {"lam": 0.2, "kappa": 0.15, "c1": 0.2, "eta": 0.1}
        S_history = np.random.normal(1.0, 0.1, 50)
        theta_history = np.random.normal(1.0, 0.1, 50)
        result = validate_system_dynamics(config, S_history, theta_history)
        assert result["valid"] is False


class TestStabilityAnalyzer:
    """Tests for StabilityAnalyzer class."""

    def test_initialization(self):
        """Should initialize correctly."""
        config = {"lam": 0.2, "kappa": 0.15, "c1": 0.2, "eta": 0.1}
        analyzer = StabilityAnalyzer(config)
        assert analyzer.config == config
        assert len(analyzer.history["S"]) == 0
        assert len(analyzer.history["theta"]) == 0

    def test_step(self):
        """Should record state."""
        config = {"lam": 0.2, "kappa": 0.15, "c1": 0.2, "eta": 0.1}
        analyzer = StabilityAnalyzer(config)
        analyzer.step(S=1.0, theta=1.0)
        assert len(analyzer.history["S"]) == 1
        assert len(analyzer.history["theta"]) == 1
        assert analyzer.history["S"][0] == 1.0
        assert analyzer.history["theta"][0] == 1.0

    def test_analyze(self):
        """Should perform comprehensive analysis."""
        config = {"lam": 0.2, "kappa": 0.15, "c1": 0.2, "eta": 0.1}
        analyzer = StabilityAnalyzer(config)
        # Add some history
        for _ in range(100):
            analyzer.step(S=np.random.normal(1.0, 0.1), theta=np.random.normal(1.0, 0.1))
        result = analyzer.analyze()
        assert "stability" in result
        assert "fixed_point" in result
        assert "dynamics_validation" in result

    def test_analyze_with_verbose(self):
        """Should print verbose output."""
        config = {"lam": 0.2, "kappa": 0.15, "c1": 0.2, "eta": 0.1}
        analyzer = StabilityAnalyzer(config)
        for _ in range(100):
            analyzer.step(S=np.random.normal(1.0, 0.1), theta=np.random.normal(1.0, 0.1))
        result = analyzer.analyze(verbose=True)
        assert "stability" in result

    def test_analyze_insufficient_history(self):
        """Should handle insufficient history gracefully."""
        config = {"lam": 0.2, "kappa": 0.15, "c1": 0.2, "eta": 0.1}
        analyzer = StabilityAnalyzer(config)
        # Add only a few steps
        for _ in range(10):
            analyzer.step(S=1.0, theta=1.0)
        result = analyzer.analyze()
        assert "stability" in result
        assert "fixed_point" in result
        assert "dynamics_validation" in result
        assert result["dynamics_validation"]["valid"] is False
        assert "reason" in result["dynamics_validation"]


class TestAnalyzeBifurcationExtended:
    """Extended tests for bifurcation analysis."""

    def test_bifurcation_finds_transition_points(self):
        """Should find bifurcation points when stability changes."""
        config = {"lam": 0.2, "kappa": 0.15, "c1": 0.2, "eta": 0.1}
        result = analyze_bifurcation(
            config,
            param_name="kappa",
            param_range=(0.01, 1.0),
            n_points=50,
        )
        assert "bifurcation_points" in result
        assert isinstance(result["bifurcation_points"], list)

    def test_bifurcation_stable_region(self):
        """Should identify stable region."""
        config = {"lam": 0.2, "kappa": 0.15, "c1": 0.2, "eta": 0.1}
        result = analyze_bifurcation(
            config,
            param_name="lam",
            param_range=(0.1, 0.9),
            n_points=20,
        )
        assert "stable_region" in result
        assert "min" in result["stable_region"]
        assert "max" in result["stable_region"]

    def test_bifurcation_parameter_values_length(self):
        """Should have correct number of parameter values."""
        config = {"lam": 0.2, "kappa": 0.15, "c1": 0.2, "eta": 0.1}
        n_points = 30
        result = analyze_bifurcation(
            config,
            param_name="eta",
            param_range=(0.01, 0.5),
            n_points=n_points,
        )
        assert len(result["parameter_values"]) == n_points
        assert len(result["stability"]) == n_points
        assert len(result["eigenvalue_magnitudes"]) == n_points


class TestValidateSystemDynamicsExtended:
    """Extended tests for system dynamics validation."""

    def test_dynamics_validates_correlations(self):
        """Should compute and return correlations."""
        config = {"lam": 0.2, "kappa": 0.15, "c1": 0.2, "eta": 0.1}
        # Generate correlated data
        S_history = np.linspace(0, 1, 200)
        theta_history = np.linspace(0, 1, 200)
        result = validate_system_dynamics(config, S_history, theta_history)
        assert result["valid"] is True
        assert "correlation_S" in result
        assert "correlation_theta" in result
        assert "mean_correlation" in result
        assert "linearization_valid" in result

    def test_dynamics_with_uncorrelated_data(self):
        """Should handle uncorrelated data."""
        config = {"lam": 0.2, "kappa": 0.15, "c1": 0.2, "eta": 0.1}
        # Generate uncorrelated random data
        np.random.seed(42)
        S_history = np.random.randn(200)
        theta_history = np.random.randn(200)
        result = validate_system_dynamics(config, S_history, theta_history)
        assert result["valid"] is True
        assert isinstance(result["prediction_error"], float)

    def test_dynamics_fixed_point_values(self):
        """Should compute fixed point values."""
        config = {"lam": 0.2, "kappa": 0.15, "c1": 0.2, "eta": 0.1}
        S_history = np.ones(200) * 2.0
        theta_history = np.ones(200) * 1.5
        # Suppress numpy divide by zero warnings for constant arrays
        with np.errstate(divide="ignore", invalid="ignore"):
            result = validate_system_dynamics(config, S_history, theta_history)
        assert result["valid"] is True
        assert "fixed_point" in result
        assert "S_star" in result["fixed_point"]
        assert "theta_star" in result["fixed_point"]
