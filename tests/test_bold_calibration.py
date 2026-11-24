"""Tests for BOLD calibration and energy conversion module."""

import numpy as np
import pytest

from energy.bold_calibration import (
    BOLDCalibrator,
    bold_signal_to_energy,
    calibrate_r_l_from_bold,
    compute_landauer_energy_per_bit,
    estimate_bits_from_bold_energy,
    validate_energy_against_landauer,
)


def test_compute_landauer_energy_per_bit():
    """Test Landauer energy per bit calculation."""
    # At body temperature (310K)
    e_min = compute_landauer_energy_per_bit(310.0)
    assert e_min > 0
    assert np.isclose(e_min, 1.380649e-23 * 310.0 * np.log(2.0))
    # At room temperature (300K)
    e_min_room = compute_landauer_energy_per_bit(300.0)
    assert e_min_room < e_min  # Lower temperature = lower minimum energy


def test_bold_signal_to_energy():
    """Test BOLD signal to energy conversion."""
    # Test with default parameters
    energy = bold_signal_to_energy(2.5)  # 2.5% BOLD change
    assert energy > 0
    # Test with custom parameters
    energy_custom = bold_signal_to_energy(
        2.5,
        baseline_energy=2e-9,
        conversion_factor=2e-6,
        tissue_volume=2.0,
    )
    assert energy_custom > energy  # Larger volume = more energy


def test_calibrate_kappa_meta_from_bold():
    """Test κ_meta calibration from BOLD signal."""
    # Test with typical values
    kappa = calibrate_r_l_from_bold(
        bold_signal_change=2.5,  # 2.5% BOLD change
        bits_erased=6.6,  # Typical bits for S=1.0, eps=0.01
    )
    assert kappa > 0
    # Test edge cases
    kappa_zero = calibrate_r_l_from_bold(2.5, 0.0)
    assert kappa_zero == 0.0
    # Test that κ_meta >= Landauer minimum (at least 1x)
    e_min_per_bit = compute_landauer_energy_per_bit()
    assert kappa / e_min_per_bit >= 1.0


def test_estimate_bits_from_bold_energy():
    """Test bits estimation from BOLD energy."""
    # Test round-trip: calibrate κ, then estimate bits
    bold_change = 2.5
    true_bits = 6.6
    kappa = calibrate_r_l_from_bold(bold_change, true_bits)
    estimated_bits = estimate_bits_from_bold_energy(bold_change, kappa)
    # Should be close to original (within 10% due to numerical precision)
    assert np.isclose(estimated_bits, true_bits, rtol=0.1)
    # Test edge cases
    zero_bits = estimate_bits_from_bold_energy(2.5, 0.0)
    assert zero_bits == 0.0


def test_validate_energy_against_landauer():
    """Test energy validation against Landauer's principle."""
    # Test case that satisfies Landauer (energy > minimum)
    validation = validate_energy_against_landauer(
        measured_energy=2e-19,  # ~10× Landauer minimum (1.96e-20)
        bits_erased=6.6,
    )
    assert bool(validation["satisfied"])
    assert validation["ratio"] >= 1.0
    # Test case that violates Landauer (energy < minimum)
    validation = validate_energy_against_landauer(
        measured_energy=1e-22,  # 0.1× Landauer minimum
        bits_erased=6.6,
    )
    assert not bool(validation["satisfied"])
    assert validation["ratio"] < 1.0
    assert validation["violation_j"] > 0
    # Test edge case: no bits erased
    validation = validate_energy_against_landauer(
        measured_energy=1e-20,
        bits_erased=0.0,
    )
    assert bool(validation["satisfied"])
    assert validation["message"] == "No information to erase"


def test_bold_calibrator():
    """Test BOLDCalibrator class."""
    calibrator = BOLDCalibrator()
    # Calibrate from multiple trials
    trials = [
        (1.0, 2.5, 6.6, 1.0),
        (1.2, 3.0, 8.0, 1.0),
        (0.8, 2.0, 5.0, 1.0),
    ]
    for baseline, ignition, bits, duration in trials:
        kappa = calibrator.calibrate_from_trial(baseline, ignition, bits, duration)
        assert kappa > 0
    # Get summary
    summary = calibrator.get_calibration_summary()
    assert summary["calibrated"] is True
    assert summary["n_trials"] == 3
    assert summary["r_l_mean"] > 0
    assert summary["spike_factor_mean"] > 1.0  # Spike factor > 1
    # Test validation with calibrator
    validation = calibrator.validate_against_landauer(1e-20, 6.6)
    assert "satisfied" in validation


def test_energy_with_ignition_spike():
    """Test energy computation with ignition spike."""
    from energy.bold_calibration import compute_energy_with_ignition_spike

    result = compute_energy_with_ignition_spike(
        baseline_bold=1.0,
        ignition_bold=2.5,
        duration=1.0,
    )
    assert "baseline_energy_j" in result
    assert "spike_energy_j" in result
    assert "total_energy_j" in result
    assert "spike_factor" in result
    # Spike factor should be ignition_bold / baseline_bold
    assert np.isclose(result["spike_factor"], 2.5 / 1.0)
    # Total energy should be positive
    assert result["total_energy_j"] > 0


def test_thermodynamic_cost_with_calibrated_kappa():
    """Test thermodynamic cost computation with calibrated κ."""
    from core.thermodynamics import compute_landauer_cost

    # Test with dimensionless κ (legacy mode)
    cost_dimensionless = compute_landauer_cost(
        S=1.0,
        eps=0.01,
        kappa_meta=1.0,
        kappa_units="dimensionless",
    )
    assert cost_dimensionless > 0
    # Test with calibrated κ in J/bit
    kappa_calibrated = 3.21e-21  # Landauer minimum at body temp
    cost_calibrated = compute_landauer_cost(
        S=1.0,
        eps=0.01,
        kappa_meta=kappa_calibrated,
        kappa_units="joules_per_bit",
    )
    assert cost_calibrated > 0
    # For S=1.0, eps=0.01, bits ≈ 6.6
    # Cost should be approximately κ * bits
    bits = np.log2(1.0 / 0.01)
    expected_cost = kappa_calibrated * bits
    assert np.isclose(cost_calibrated, expected_cost, rtol=0.01)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
