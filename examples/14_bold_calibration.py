#!/usr/bin/env python3
"""
Final BOLD Calibration Example
Demonstrates the complete workflow for calibrating APGI energy module
using BOLD fMRI data to produce an energy per bit erased (R_L × Landauer floor).
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
from energy.bold_calibration import compute_landauer_energy_per_bit  # noqa: E402
from energy.calibration_utils import calibrate_for_realistic_r_l  # noqa: E402
from pipeline import APGIPipeline  # noqa: E402


def main() -> None:
    """Demonstrate complete BOLD calibration workflow."""
    print("=" * 70)
    print("APGI BOLD Calibration: Complete Workflow")
    print("=" * 70)
    print("\nStep 1: Understand the Physics")
    print("-" * 40)
    # Landauer's minimum at body temperature
    e_min = compute_landauer_energy_per_bit(310.0)
    print(f"Landauer minimum (310K): {e_min:.2e} J/bit")
    print("This is the ABSOLUTE minimum energy to erase 1 bit")
    print("\nBiological systems are MUCH less efficient:")
    print("  - Typical neural computation: 10^3-10^4 × Landauer minimum")
    print("  - This reflects metabolic overhead, not just information theory")
    print("\nStep 2: Calibrate Conversion Parameters")
    print("-" * 40)
    # Calibrate for typical neural efficiency (1000× Landauer)
    calibration = calibrate_for_realistic_r_l(
        target_r_l=1000.0,
        typical_bold_change=2.0,
        typical_bits=6.6,
    )
    print(f"Target: R_L = {calibration['target_r_l']:.0f}× Landauer")
    print(f"Energy per bit = {calibration['target_energy_j_per_bit']:.2e} J/bit")
    print(
        f"Calibrated conversion factor: {calibration['calibrated_conversion_factor']:.2e} J/%/cm³"
    )
    print("\nStep 3: Create Calibrated Configuration")
    print("-" * 40)
    config = CONFIG.copy()
    config["use_thermodynamic_cost"] = True
    config["use_bold_calibration"] = True
    config["r_l"] = calibration["target_energy_j_per_bit"]
    config["r_l_units"] = "joules_per_bit"
    config["bold_conversion_factor"] = calibration["calibrated_conversion_factor"]
    print("APGI Configuration with BOLD calibration:")
    print(f"  R_L factor: {config['r_l']:.2e} J/bit")
    print(f"  units: {config['r_l_units']}")
    print(f"  BOLD conversion: {config['bold_conversion_factor']:.2e} J/%/cm³")
    print(f"  Efficiency: {float(config['r_l']) / e_min:.0f}× Landauer minimum")  # type: ignore[arg-type]
    print("\nStep 4: Compare Energy Calculations")
    print("-" * 40)
    S_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    eps = 0.01
    print("Signal | Bits | Energy (Legacy) | Energy (BOLD-calibrated) | Ratio")
    print("-" * 90)
    for S in S_values:
        bits = compute_information_bits(S, eps)
        # Legacy calculation (dimensionless R_L = 1.0)
        energy_legacy = compute_landauer_cost(S, eps, r_l=1.0, r_l_units="dimensionless")
        # BOLD-calibrated calculation
        energy_calibrated = compute_landauer_cost(
            S,
            eps,
            r_l=float(config["r_l"]),
            r_l_units="joules_per_bit",  # type: ignore[arg-type]
        )
        ratio = energy_calibrated / energy_legacy if energy_legacy > 0 else 0
        print(
            f"{S:6.1f} | {bits:4.1f} | {energy_legacy:12.2e} J | {energy_calibrated:20.2e} J | {ratio:8.0f}×"
        )
    print("\nStep 5: Run APGI Pipeline with Calibration")
    print("-" * 40)
    pipeline = APGIPipeline(config)
    # Run a short simulation
    n_steps = 50
    print(f"Running {n_steps} steps of APGI with BOLD-calibrated thermodynamics...")
    energy_data = []
    for t in range(n_steps):
        # Create varying input signals
        x_e = 0.5 + 0.3 * np.sin(2 * np.pi * t / 50)
        x_i = 0.2 + 0.1 * np.cos(2 * np.pi * t / 50)
        result = pipeline.step(x_e=x_e, x_hat_e=0.3, x_i=x_i, x_hat_i=0.1)
        if "C_landauer" in result:
            energy_data.append(
                {
                    "S": result["S"],
                    "energy_j": result["C_landauer"],
                    "bits": result.get("bits_erased", 0.0),
                    "metabolic_cost": result["C"],
                }
            )
    if energy_data:
        avg_energy: float = float(np.mean([d["energy_j"] for d in energy_data]))
        avg_bits: float = float(np.mean([d["bits"] for d in energy_data]))
        print("\nResults:")
        print(f"  Average energy per ignition: {avg_energy:.2e} J")
        print(f"  Average bits per ignition: {avg_bits:.1f}")
        print(f"  Average energy per bit: {avg_energy / max(avg_bits, 1e-6):.2e} J/bit")
        print(f"  Efficiency: {(avg_energy / max(avg_bits, 1e-6)) / e_min:.0f}× Landauer")
    print("\nStep 6: Validate Against Physical Limits")
    print("-" * 40)
    from energy.bold_calibration import validate_energy_against_landauer  # noqa: E402

    # Simulate a measurement
    measured_energy = 1e-17  # 10 attojoules (realistic neural energy)
    bits_erased = 6.6
    validation = validate_energy_against_landauer(measured_energy, bits_erased)
    print(f"Measured energy: {validation['measured_energy_j']:.2e} J")
    print(f"Landauer minimum: {validation['landauer_minimum_j']:.2e} J")
    print(f"Ratio: {validation['ratio']:.1f}")
    print(f"Validation: {validation['message']}")
    print("\n" + "=" * 70)
    print("SUMMARY: Energy Module Refinements")
    print("=" * 70)
    print("\n✅ ACCOMPLISHED:")
    print("1. Added BOLD-to-Joule conversion module")
    print("2. r_l can be given in physical units (J/bit) instead of as dimensionless R_L")
    print("3. Models 5-10% energy spike during ignition events")
    print("4. Enables direct comparison with Landauer's physical minimum")
    print("5. Provides empirical grounding in fMRI neuroscience")
    print("\n🔧 KEY FEATURES:")
    print("• Two modes: dimensionless R_L (legacy) or J/bit energy per bit (calibrated)")
    print("• BOLD calibration from fMRI data")
    print("• Energy validation against thermodynamic limits")
    print("• Configurable efficiency levels (1× to 10^6× Landauer)")
    print("\n📊 TYPICAL VALUES:")
    print(f"• Landauer minimum (310K): {e_min:.2e} J/bit")
    print(f"• Typical neural energy per bit: {calibration['target_energy_j_per_bit']:.2e} J/bit")
    print(f"• Typical overhead R_L: {calibration['target_r_l']:.0f}× Landauer")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
