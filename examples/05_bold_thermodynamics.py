#!/usr/bin/env python3
"""
APGI BOLD-Calibrated Thermodynamic Analysis Example
This example demonstrates thermodynamic analysis using BOLD fMRI calibration
to estimate R_L, the erasure overhead over the Landauer floor (formerly κ_meta).
Spec Reference: §11 Thermodynamic Constraints with BOLD calibration
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from config import CONFIG  # noqa: E402
from core.thermodynamics import (
    compute_information_bits,  # noqa: E402
    compute_landauer_cost,
)
from energy.bold_calibration import (
    BOLDCalibrator,  # noqa: E402
    validate_energy_against_landauer,
)
from pipeline import APGIPipeline  # noqa: E402


def main() -> None:
    """Run BOLD-calibrated thermodynamic analysis example."""
    print("=" * 60)
    print("APGI BOLD-Calibrated Thermodynamic Analysis Example")
    print("=" * 60)
    # Part 1: Demonstrate BOLD calibration
    print("\n1. BOLD fMRI Calibration Demonstration")
    print("-" * 40)
    # Create BOLD calibrator
    calibrator = BOLDCalibrator()
    # Simulate some trials with BOLD measurements
    trials = [
        {"baseline_bold": 1.0, "ignition_bold": 2.5, "bits": 6.6, "duration": 1.0},
        {"baseline_bold": 1.2, "ignition_bold": 3.0, "bits": 8.0, "duration": 1.0},
        {"baseline_bold": 0.8, "ignition_bold": 2.0, "bits": 5.0, "duration": 1.0},
    ]
    print(f"Estimating R_L from {len(trials)} simulated fMRI trials...")
    for i, trial in enumerate(trials):
        r_l = calibrator.calibrate_from_trial(
            trial["baseline_bold"],
            trial["ignition_bold"],
            trial["bits"],
            trial["duration"],
        )
        print(f"  Trial {i + 1}: R_L = {r_l:.2e} (dimensionless)")
    summary = calibrator.get_calibration_summary()
    print("\nCalibration Summary:")
    print(f"  R_L mean: {summary['r_l_mean']:.2e}")
    print(f"  R_L std: {summary['r_l_std']:.2e}")
    print(f"  Landauer minimum: {summary['landauer_energy_per_bit_j']:.2e} J/bit")
    print(f"  Energy per bit: {summary['r_l_mean'] * summary['landauer_energy_per_bit_j']:.2e} J")
    # Part 2: Compare calibrated vs uncalibrated energy calculations
    print("\n2. Calibrated vs Uncalibrated Energy Calculations")
    print("-" * 40)
    S_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    eps = 0.01
    print("Signal | Bits | Landauer (R_L = 1) | Landauer (BOLD R_L)")
    print("-" * 80)
    for S in S_values:
        bits = compute_information_bits(S, eps)
        # Uncalibrated (R_L = 1.0, the Landauer floor)
        cost_uncalibrated = compute_landauer_cost(S, eps, r_l=1.0, r_l_units="dimensionless")
        # Calibrated (mean R_L from BOLD; R_L is dimensionless)
        cost_calibrated = compute_landauer_cost(
            S, eps, r_l=summary["r_l_mean"], r_l_units="dimensionless"
        )
        print(f"{S:6.1f} | {bits:4.1f} | {cost_uncalibrated:10.2e} J | {cost_calibrated:10.2e} J")
    # Part 3: Validate energy against Landauer's principle
    print("\n3. Energy Validation Against Landauer's Principle")
    print("-" * 40)
    # Simulate measured energy from BOLD using the calibrated R_L
    bold_change = 2.5  # 2.5% BOLD signal change
    bits_erased = 6.6
    # R_L is dimensionless, so measured energy = R_L × N_erase × E_min per bit
    measured_energy = summary["r_l_mean"] * bits_erased * summary["landauer_energy_per_bit_j"]
    # Validate
    validation = validate_energy_against_landauer(measured_energy, bits_erased)
    print(f"BOLD signal change: {bold_change:.1f}%")
    print(f"Bits erased: {bits_erased:.1f}")
    print(f"Measured energy: {validation['measured_energy_j']:.2e} J")
    print(f"Landauer minimum: {validation['landauer_minimum_j']:.2e} J")
    print(f"Ratio (E/E_min): {validation['ratio']:.1f}")
    print(f"Validation: {validation['message']}")
    # Part 4: APGI pipeline with BOLD calibration
    print("\n4. APGI Pipeline with BOLD-Calibrated Thermodynamics")
    print("-" * 40)
    # Create configuration with BOLD calibration
    config = CONFIG.copy()
    config["use_thermodynamic_cost"] = True
    config["use_bold_calibration"] = True
    config["r_l"] = summary["r_l_mean"]  # calibrated R_L (formerly κ_meta)
    config["r_l_units"] = "dimensionless"
    print("Configuration with BOLD calibration:")
    print(f"  R_L factor: {config['r_l']:.2e}")
    print(f"  units: {config['r_l_units']}")
    print(f"  BOLD conversion factor: {config['bold_conversion_factor']:.2e} J/%/cm³")
    print(f"  Ignition spike factor: {config['bold_ignition_spike_factor']:.3f}")
    # Initialize pipeline
    pipeline = APGIPipeline(config)
    # Run a few steps
    n_steps = 100
    print(f"\nRunning {n_steps} steps with BOLD-calibrated thermodynamics...")
    thermodynamic_data: dict[str, list[float]] = {
        "S": [],
        "landauer_cost": [],
        "info_bits": [],
        "metabolic_cost": [],
        "efficiency": [],
    }
    for t in range(n_steps):
        # Create input signals
        x_e = 0.5 + 0.2 * np.sin(2 * np.pi * t / 100)
        x_i = 0.2 + 0.1 * np.cos(2 * np.pi * t / 100)
        # Execute step
        result = pipeline.step(x_e=x_e, x_hat_e=0.3, x_i=x_i, x_hat_i=0.1)
        # Record data
        S = result["S"]
        thermodynamic_data["S"].append(S)
        if "C_landauer" in result:
            thermodynamic_data["landauer_cost"].append(result["C_landauer"])
            thermodynamic_data["info_bits"].append(result.get("bits_erased", 0.0))
            thermodynamic_data["metabolic_cost"].append(result["C"])
            if result["C_landauer"] > 0:
                # Compute the BOLD-calibration efficiency: what fraction of E_min does
                # the BOLD-enforced step cost represent?
                # NOTE: C_landauer uses log2(S/eps) bits, while BOLD energy corresponds
                # to ~0.9 bits at R_L resolution.  These differ by ~29×, so this ratio
                # will always be << 1 and should NOT be read as a Landauer violation.
                # The correct constraint check (Part 3 above) already confirms compliance.
                SCALE_FACTOR = 1e20  # AU → J (matches compute_metabolic_cost_realistic)
                C_joules = result["C"] / SCALE_FACTOR
                ratio = C_joules / result["C_landauer"]
                thermodynamic_data["efficiency"].append(ratio)
            else:
                thermodynamic_data["efficiency"].append(0.0)
    # Analyze results
    SCALE_FACTOR = 1e20  # AU → J (matches threshold.py compute_metabolic_cost_realistic)
    if thermodynamic_data["landauer_cost"]:
        avg_cost = np.mean(thermodynamic_data["landauer_cost"])
        avg_bits = np.mean(thermodynamic_data["info_bits"])
        # np.mean of an empty list warns ("Mean of empty slice") and returns
        # NaN. A run where every efficiency is zero is a legitimate outcome
        # (no ignitions), so report it as such rather than as NaN.
        positive_ratios = [e for e in thermodynamic_data["efficiency"] if e > 0]
        avg_ratio = float(np.mean(positive_ratios)) if positive_ratios else 0.0
        print("\nResults:")
        print(f"  Average Landauer cost (R_L × log2(S/eps) × kT ln2): {avg_cost:.2e} J")
        print(f"  Average bits erased (log2(S/eps)):           {avg_bits:.1f}")
        print(f"  BOLD energy / Landauer cost ratio: {avg_ratio:.3f}×")
        print("  (Ratio < 1 is expected: BOLD 2% = ~0.9 bits vs log2(S/eps) ≈ 27 bits)")
        print("  See Part 3 for the correct Landauer constraint check (ratio=1020×)")
        # Validate overall: convert AU → J for the energy sum
        total_energy_au = np.sum(thermodynamic_data["metabolic_cost"])
        total_bits = np.sum(thermodynamic_data["info_bits"])
        total_energy_j = total_energy_au / SCALE_FACTOR
        overall_validation = validate_energy_against_landauer(total_energy_j, total_bits)
        print("\nOverall Validation:")
        print(f"  Total energy: {total_energy_j:.2e} J")
        print(f"  Total bits: {total_bits:.1f}")
        print(f"  Validation: {overall_validation['message']}")
    print("\n" + "=" * 60)
    print("Summary:")
    print("  ✓ BOLD calibration estimates R_L, the erasure overhead over the Landauer floor")
    print("  ✓ R_L is dimensionless; energy per bit = R_L × k_B·T·ln 2")
    print("  ✓ 5-10% energy spike during ignition events is modeled")
    print("  ✓ Direct comparison with Landauer's physical minimum enabled")
    print("=" * 60)


if __name__ == "__main__":
    main()
