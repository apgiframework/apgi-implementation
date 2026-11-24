"""Pytest configuration and shared fixtures for APGI test suite.
This module provides:
- Common fixtures for numpy arrays and random seeds
- Configuration fixtures for all APGI components
- Mock fixtures for external dependencies
"""

from __future__ import annotations

import warnings
from collections.abc import Generator
from typing import Any

import numpy as np
import pytest

# Suppress LAPACK/BLAS warnings during tests (DLASCL parameter warnings)
warnings.filterwarnings("ignore", message=".*On entry to DLASCL.*")
warnings.filterwarnings("ignore", message=".*On entry to DLASCL.*", category=RuntimeWarning)

# =============================================================================
# LAPACK Stderr Suppression
# =============================================================================
# LAPACK's Fortran routines write warnings directly to stderr, bypassing
# Python's warning system. We suppress these at the file descriptor level.


@pytest.fixture(scope="session", autouse=True)
def suppress_lapack_stderr_session() -> Generator[None, None, None]:
    """Suppress LAPACK stderr output during entire test session.
    LAPACK's DLASCL routine writes parameter warnings directly to stderr (fd 2)
    when matrices have extreme values. We redirect the file descriptor at OS level
    for the entire test session to capture all output.
    """
    import os
    import sys

    # Save original stderr file descriptor
    original_stderr_fd = os.dup(2)
    # Create a pipe to capture stderr
    pipe_read, pipe_write = os.pipe()
    os.dup2(pipe_write, 2)
    os.close(pipe_write)
    yield
    # Restore original stderr
    os.dup2(original_stderr_fd, 2)
    os.close(original_stderr_fd)
    # Read and filter captured output (read in chunks to handle large output)
    chunks = []
    while True:
        chunk = os.read(pipe_read, 65536)
        if not chunk:
            break
        chunks.append(chunk)  # pragma: no cover  # only reached when LAPACK writes stderr
    os.close(pipe_read)
    if chunks:  # pragma: no cover  # only reached when LAPACK writes stderr
        captured = b"".join(chunks).decode("utf-8", errors="replace")
        # Filter out LAPACK messages and write remaining content
        lines = captured.split("\n")
        skip_next = False
        for line in lines:
            # Match "** On entry to DLASCL" or "** On entry to DLASCL parameter number"
            if "** On entry to DLASCL" in line:
                skip_next = True
                continue
            # Skip lines with "parameter number X had an illegal value"
            if skip_next and ("had an illegal value" in line or "parameter number" in line):
                skip_next = False
                continue
            # Also filter standalone DLASCL-related messages
            if "DLASCL" in line or "Dlascl" in line:
                continue
            if line:
                sys.stderr.write(line + "\n")
        sys.stderr.flush()


@pytest.fixture
def suppress_lapack() -> Generator[None, None, None]:
    """Suppress LAPACK DLASCL warnings from stderr.
    Use this fixture in tests that intentionally trigger LAPACK edge cases
    (e.g., singular matrices, duplicate frequencies) that cause DLASCL warnings.
    Example:
        def test_edge_case(suppress_lapack):
            with suppress_lapack:
                result = function_that_triggers_lapack_warning()
    """
    import os

    original_stderr_fd = os.dup(2)
    pipe_read, pipe_write = os.pipe()
    os.dup2(pipe_write, 2)
    os.close(pipe_write)
    try:
        yield
    finally:
        os.dup2(original_stderr_fd, 2)
        os.close(original_stderr_fd)
        # Drain and discard the pipe
        while os.read(pipe_read, 65536):  # pragma: no cover
            pass
        os.close(pipe_read)


# =============================================================================
# Random Seed Fixtures
# =============================================================================


@pytest.fixture
def rng() -> np.random.Generator:
    """Provide a seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def seeded_random() -> Generator[None, None, None]:
    """Seed numpy random for deterministic tests."""
    np.random.seed(42)
    yield
    np.random.seed(None)


# =============================================================================
# Basic Array Fixtures
# =============================================================================


@pytest.fixture
def sample_signal_history() -> np.ndarray:
    """Provide a sample signal history array."""
    return np.array([0.5, 0.8, 1.2, 1.5, 1.0, 0.7, 0.9, 1.1, 1.3, 1.0])


@pytest.fixture
def sample_threshold_history() -> np.ndarray:
    """Provide a sample threshold history array."""
    return np.array([1.0, 1.05, 1.02, 0.98, 1.0, 1.03, 0.99, 1.01, 0.97, 1.0])


@pytest.fixture
def sample_ignition_history() -> np.ndarray:
    """Provide a sample binary ignition history."""
    return np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 0])


@pytest.fixture
def sample_time_series() -> np.ndarray:
    """Provide a longer sample time series for spectral analysis."""
    t = np.linspace(0, 10, 1000)
    return (
        np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t) + 0.1 * np.random.randn(1000)
    )


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def base_config() -> dict[str, Any]:
    """Provide a valid base configuration dictionary."""
    return {
        "lam": 0.2,
        "kappa": 0.15,
        "ignite_tau": 0.5,
        "dt": 0.1,
        "tau_s": 5.0,
        "tau_theta": 1000.0,
        "tau_pi": 1000.0,
        "pi_min": 1e-4,
        "pi_max": 1e4,
        "eps": 1e-8,
        "eta": 0.1,
        "noise_std": 0.01,
        "ne_on_precision": False,
        "ne_on_threshold": False,
    }


@pytest.fixture
def full_config() -> dict[str, Any]:
    """Provide a complete configuration with all parameters."""
    return {
        # Initial states
        "S0": 0.0,
        "theta_0": 1.0,
        "theta_base": 1.0,
        "sigma2_e0": 1.0,
        "sigma2_i0": 1.0,
        # Numerical stability
        "eps": 1e-8,
        "pi_min": 1e-4,
        "pi_max": 1e4,
        # EMA variance update
        "alpha_e": 0.05,
        "alpha_i": 0.05,
        # Variance estimation
        "variance_method": "ema",
        "T_win": 50,
        # Neuromodulation
        "g_ach": 1.0,
        "g_ne": 1.0,
        "beta": 0.0,
        "beta_da": 0.0,
        "ne_on_precision": True,
        "ne_on_threshold": False,
        "gamma_ne": 0.1,
        # Threshold decay
        "kappa": 0.15,
        # Signal accumulation
        "lam": 0.2,
        "signal_log_nonlinearity": True,
        "use_canonical_discrete_mode": False,
        # Threshold update
        "eta": 0.1,
        "delta": 0.5,
        # Post-ignition reset
        "reset_factor": 0.5,
        # Cost-value model
        "use_realistic_cost": True,
        "c0": 0.0,
        "c1": 0.2,
        "c2": 0.5,
        "v1": 0.5,
        "v2": 0.5,
        # Ignition dynamics
        "ignite_tau": 0.5,
        "tau_sigma": 0.5,
        "stochastic_ignition": False,
        # Continuous-time parameters
        "tau_s": 5.0,
        "dt": 0.5,
        "noise_std": 0.01,
        # Generative model dynamics
        "use_internal_predictions": True,
        "eta_pred_e": 0.01,
        "eta_pred_i": 0.01,
        # Multi-scale
        "timescale_k": 1.6,
        # Thermodynamic constraints
        "use_thermodynamic_cost": False,
        "k_boltzmann": 1.38e-23,
        "T_env": 310.0,
        "kappa_meta": 1.0,
        # Reservoir layer
        "use_reservoir": False,
        "reservoir_size": 100,
        "reservoir_tau": 1.0,
        "reservoir_spectral_radius": 0.9,
        "reservoir_input_scale": 0.1,
        "reservoir_readout_method": "linear",
        "reservoir_amplification": 0.0,
        "reservoir_ridge_alpha": 1e-6,
        # Kuramoto oscillators
        "use_kuramoto": False,
        "kuramoto_tau_xi": 1.0,
        "kuramoto_sigma_xi": 0.1,
        "kuramoto_reset_amount": np.pi,
        # Observable mapping
        "use_observable_mapping": False,
        # Stability analysis
        "use_stability_analysis": False,
    }


@pytest.fixture
def stability_config() -> dict[str, Any]:
    """Provide configuration optimized for stability testing."""
    return {
        "lam": 0.2,
        "kappa": 0.15,
        "c1": 0.2,
        "eta": 0.1,
        "theta_base": 1.0,
    }


@pytest.fixture
def hierarchical_config() -> dict[str, Any]:
    """Provide configuration for hierarchical testing."""
    return {
        "n_levels": 3,
        "tau_0": 1.0,
        "timescale_k": 1.6,
        "C_down": 0.1,
        "C_up": 0.05,
        "tau_pi": 1000.0,
        "kappa_down": 0.1,
        "kappa_up": 0.05,
    }


# =============================================================================
# Component Fixtures
# =============================================================================


@pytest.fixture
def reservoir_params() -> dict[str, Any]:
    """Provide parameters for reservoir initialization."""
    return {
        "N": 50,
        "M": 2,
        "tau_res": 1.0,
        "spectral_radius": 0.9,
        "input_scale": 0.1,
        "seed": 42,
    }


@pytest.fixture
def kuramoto_params() -> dict[str, Any]:
    """Provide parameters for Kuramoto oscillator initialization."""
    return {
        "n_levels": 3,
        "tau_xi": 1.0,
        "sigma_xi": 0.1,
    }


# =============================================================================
# Mock Data Fixtures
# =============================================================================


@pytest.fixture
def mock_eeg_data() -> dict[str, Any]:
    """Provide mock EEG data structure."""
    return {
        "signals": np.random.randn(32, 1000),
        "fs": 100.0,
        "channels": [f"EEG{i}" for i in range(32)],
        "duration": 10.0,
    }


@pytest.fixture
def mock_behavioral_data() -> dict[str, Any]:
    """Provide mock behavioral data."""
    return {
        "rt": np.random.lognormal(0.5, 0.2, 100),
        "accuracy": np.random.binomial(1, 0.8, 100),
        "n_trials": 100,
    }


# =============================================================================
# Test Helpers
# =============================================================================


def assert_array_equal(a: np.ndarray, b: np.ndarray, rtol: float = 1e-7) -> None:
    """Assert two arrays are equal within tolerance."""
    np.testing.assert_allclose(a, b, rtol=rtol)


def assert_scalar_equal(a: float, b: float, rtol: float = 1e-7) -> None:
    """Assert two scalars are equal within tolerance."""
    np.testing.assert_allclose(a, b, rtol=rtol)
