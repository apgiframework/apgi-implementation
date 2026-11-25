"""Tests for energy/calibration_utils.py - BOLD calibration utilities."""

from __future__ import annotations

import pytest

from energy.bold_calibration import DEFAULT_TISSUE_VOLUME, BOLDCalibrator
from energy.calibration_utils import (
    calibrate_for_realistic_r_l,
    create_realistic_calibrator,
    demonstrate_calibration_range,
)


class TestCalibrateForRealisticKappa:
    """Tests for calibrate_for_realistic_r_l function."""

    def test_default_parameters(self):
        """Should calibrate with default parameters."""
        result = calibrate_for_realistic_r_l()
        assert "target_r_l" in result
        assert "target_energy_j_per_bit" in result
        assert "calibrated_conversion_factor" in result
        assert "typical_bold_change" in result
        assert "typical_bits" in result
        assert "landauer_minimum_j_per_bit" in result
        assert "temperature_k" in result
        assert result["target_r_l"] == 1000.0
        assert result["typical_bold_change"] == 2.0
        assert result["typical_bits"] == 6.6
        assert result["temperature_k"] == 310.0

    def test_target_kappa_calculation(self):
        """Should calculate target kappa correctly."""
        result = calibrate_for_realistic_r_l(target_r_l=1000.0)
        # kappa should be 1000x Landauer minimum
        landauer = result["landauer_minimum_j_per_bit"]
        expected_kappa = 1000.0 * landauer
        assert result["target_energy_j_per_bit"] == pytest.approx(expected_kappa, rel=1e-10)

    def test_custom_efficiency(self):
        """Should handle custom efficiency levels."""
        result = calibrate_for_realistic_r_l(target_r_l=100.0)
        assert result["target_r_l"] == 100.0
        # Lower efficiency should give lower kappa
        result_1000 = calibrate_for_realistic_r_l(target_r_l=1000.0)
        assert result["target_energy_j_per_bit"] < result_1000["target_energy_j_per_bit"]

    def test_calibrated_conversion_factor(self):
        """Should calculate calibrated conversion factor."""
        result = calibrate_for_realistic_r_l(
            target_r_l=1000.0,
            typical_bold_change=2.0,
            typical_bits=6.6,
        )
        # Energy = conversion_factor * bold_change * volume
        # Required energy = kappa * bits
        # So conversion_factor = kappa * bits / (bold_change * volume)
        expected_energy = result["target_energy_j_per_bit"] * 6.6
        expected_factor = expected_energy / (2.0 * DEFAULT_TISSUE_VOLUME)
        assert result["calibrated_conversion_factor"] == pytest.approx(expected_factor, rel=1e-10)

    def test_custom_bold_change(self):
        """Should handle custom BOLD change values."""
        result = calibrate_for_realistic_r_l(
            target_r_l=1000.0,
            typical_bold_change=3.0,
        )
        # Higher BOLD change should result in lower conversion factor
        result_default = calibrate_for_realistic_r_l(typical_bold_change=2.0)
        assert (
            result["calibrated_conversion_factor"] < result_default["calibrated_conversion_factor"]
        )

    def test_custom_bits(self):
        """Should handle custom bit values."""
        result = calibrate_for_realistic_r_l(
            target_r_l=1000.0,
            typical_bits=10.0,
        )
        # More bits should result in higher conversion factor
        result_default = calibrate_for_realistic_r_l(typical_bits=6.6)
        assert (
            result["calibrated_conversion_factor"] > result_default["calibrated_conversion_factor"]
        )

    def test_custom_temperature(self):
        """Should handle custom temperature."""
        result_300 = calibrate_for_realistic_r_l(T=300.0)
        result_310 = calibrate_for_realistic_r_l(T=310.0)
        # Higher temperature should give higher Landauer minimum
        assert result_310["landauer_minimum_j_per_bit"] > result_300["landauer_minimum_j_per_bit"]

    def test_minimal_efficiency(self):
        """Should handle minimal efficiency (1x Landauer)."""
        result = calibrate_for_realistic_r_l(target_r_l=1.0)
        assert result["target_r_l"] == 1.0
        assert result["target_energy_j_per_bit"] == result["landauer_minimum_j_per_bit"]

    def test_high_efficiency(self):
        """Should handle high inefficiency (10^6x Landauer)."""
        result = calibrate_for_realistic_r_l(target_r_l=1e6)
        assert result["target_r_l"] == 1e6
        assert result["target_energy_j_per_bit"] == pytest.approx(
            1e6 * result["landauer_minimum_j_per_bit"], rel=1e-10
        )


class TestCreateRealisticCalibrator:
    """Tests for create_realistic_calibrator function."""

    def test_default_creation(self):
        """Should create calibrator with defaults."""
        calibrator, calibration = create_realistic_calibrator()
        assert isinstance(calibrator, BOLDCalibrator)
        assert isinstance(calibration, dict)
        assert calibration["target_r_l"] == 1000.0

    def test_custom_efficiency(self):
        """Should create calibrator with custom efficiency."""
        calibrator, calibration = create_realistic_calibrator(target_efficiency=500.0)
        assert calibration["target_r_l"] == 500.0
        assert calibrator.conversion_factor == calibration["calibrated_conversion_factor"]

    def test_custom_kwargs(self):
        """Should pass additional kwargs to calibration."""
        calibrator, calibration = create_realistic_calibrator(
            target_efficiency=1000.0,
            typical_bold_change=3.0,
            typical_bits=8.0,
        )
        assert calibration["typical_bold_change"] == 3.0
        assert calibration["typical_bits"] == 8.0

    def test_calibrator_configuration(self):
        """Should configure calibrator correctly."""
        calibrator, calibration = create_realistic_calibrator()
        # Verify calibrator was created with calibrated parameters
        assert calibrator.conversion_factor == calibration["calibrated_conversion_factor"]
        assert calibrator.tissue_volume == DEFAULT_TISSUE_VOLUME
        assert calibrator.ignition_spike_factor == 1.075
        assert calibration["temperature_k"] == calibrator.T

    def test_very_efficient_system(self):
        """Should handle very efficient systems."""
        calibrator, calibration = create_realistic_calibrator(target_efficiency=10.0)
        assert calibration["target_r_l"] == 10.0
        assert isinstance(calibrator, BOLDCalibrator)

    def test_very_inefficient_system(self):
        """Should handle very inefficient biological systems."""
        calibrator, calibration = create_realistic_calibrator(target_efficiency=1e6)
        assert calibration["target_r_l"] == 1e6
        assert isinstance(calibrator, BOLDCalibrator)


class TestDemonstrateCalibrationRange:
    """Tests for demonstrate_calibration_range function."""

    def test_demonstration_output(self, capsys):
        """Should print calibration demonstration."""
        demonstrate_calibration_range()
        captured = capsys.readouterr()
        output = captured.out
        # Check header
        assert "BOLD Calibration for Different Biological Efficiency Levels" in output
        # Check efficiency levels
        assert "Theoretical minimum" in output
        assert "Highly efficient synthetic" in output
        assert "Optimized biological" in output
        assert "Typical neural" in output
        assert "Inefficient biological" in output
        assert "Pathological" in output
        # Check that kappa values are printed
        assert "R_L = 1" in output or "R_L = 1×" in output
        assert "R_L = 10" in output or "R_L = 10×" in output
        assert "R_L = 100" in output or "R_L = 100×" in output
        assert "R_L = 1000" in output or "R_L = 1000×" in output
        assert "R_L = 10000" in output or "R_L = 10000×" in output
        assert "R_L = 1000000" in output or "R_L = 1000000×" in output
        # Check units
        assert "J/bit" in output
        assert "J/%/cm³" in output
        # Check footer note
        assert "Default parameters produce" in output
        assert "Landauer" in output

    def test_efficiency_levels_order(self, capsys):
        """Should print efficiency levels in ascending order."""
        demonstrate_calibration_range()
        captured = capsys.readouterr()
        output = captured.out
        # Find positions of efficiency markers
        pos_1 = output.find("1× Landauer")
        pos_10 = output.find("10× Landauer")
        pos_100 = output.find("100× Landauer")
        pos_1000 = output.find("1000× Landauer")
        pos_10000 = output.find("10000× Landauer")
        pos_1000000 = output.find("1000000× Landauer")
        # All should be found
        assert pos_1 != -1
        assert pos_10 != -1
        assert pos_100 != -1
        assert pos_1000 != -1
        assert pos_10000 != -1
        assert pos_1000000 != -1
        # Should be in ascending order
        assert pos_1 < pos_10 < pos_100 < pos_1000 < pos_10000 < pos_1000000

    def test_energy_per_bit_values_increase(self, capsys):
        """Should show increasing energy-per-bit values (formerly κ_meta)."""
        demonstrate_calibration_range()
        captured = capsys.readouterr()
        output = captured.out
        # Extract energy-per-bit values (scientific notation after "Energy per bit:")
        import re

        kappa_values = re.findall(r"Energy per bit:\s+([\d.eE+-]+)", output)
        # Should find 6 values
        assert len(kappa_values) == 6
        # Convert to floats and verify ascending order
        kappa_floats = [float(v) for v in kappa_values]
        for i in range(len(kappa_floats) - 1):
            assert kappa_floats[i] < kappa_floats[i + 1]

    def test_main_execution(self):
        """Should execute without errors when run as main."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "energy.calibration_utils"], capture_output=True, text=True
        )
        assert result.returncode == 0
        assert "BOLD Calibration for Different Biological Efficiency Levels" in result.stdout
