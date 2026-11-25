#!/usr/bin/env python3
"""
APGI Thermodynamic Analysis Example
This example demonstrates thermodynamic analysis using Landauer's principle
to ground metabolic cost in information theory.
Spec Reference: §11 Thermodynamic Constraints
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from config import CONFIG
from core.thermodynamics import compute_information_bits, compute_landauer_cost
from pipeline import APGIPipeline


def main() -> None:
    """Run thermodynamic analysis example."""
    print("=" * 60)
    print("APGI Thermodynamic Analysis Example")
    print("=" * 60)
    # Create configuration with thermodynamics
    print("\n1. Creating configuration with thermodynamics...")
    config = CONFIG.copy()
    config["use_thermodynamic_cost"] = True
    # Use simple cost model for clearer thermodynamic analysis
    config["use_realistic_cost"] = False
    config["c1"] = 1.0  # Simple linear cost coefficient
    config["c2"] = 0.0  # No ignition cost for this example
    print("   ✓ Configuration created")
    print(f"     - Boltzmann constant: {float(config['k_boltzmann']):.2e} J/K")  # type: ignore[arg-type]
    print(
        f"     - Environment temperature: {float(config['T_env']):.1f} K ({float(config['T_env']) - 273:.1f}°C)"  # type: ignore[arg-type]
    )
    print(f"     - Erasure-overhead factor R_L: {float(config['r_l']):.2f}")  # type: ignore[arg-type]
    print("     - Cost model: C(t) = c1 * S(t) (simple linear)")
    # Initialize pipeline
    print("\n2. Initializing APGI pipeline...")
    pipeline = APGIPipeline(config)
    print("   ✓ Pipeline initialized")
    # Run simulation
    print("\n3. Running simulation for thermodynamic analysis...")
    n_steps = 500
    thermodynamic_data: dict[str, list[float]] = {
        "S": [],
        "landauer_cost": [],
        "info_bits": [],
        "metabolic_cost": [],
        "efficiency": [],
        "temperature": [],
    }
    for t in range(n_steps):
        # Create input signals
        x_e = 0.5 + 0.2 * np.sin(2 * np.pi * t / 100)
        x_i = 0.2 + 0.1 * np.cos(2 * np.pi * t / 100)
        # Execute step
        result = pipeline.step(x_e=x_e, x_hat_e=0.3, x_i=x_i, x_hat_i=0.1)
        # Record signal
        S = result["S"]
        thermodynamic_data["S"].append(S)
        # Compute Landauer cost
        landauer = compute_landauer_cost(
            S=S,
            eps=float(config["eps"]),  # type: ignore[arg-type]
            k_b=float(config["k_boltzmann"]),  # type: ignore[arg-type]
            T_env=float(config["T_env"]),  # type: ignore[arg-type]
            r_l=float(config["r_l"]),  # type: ignore[arg-type]
        )
        thermodynamic_data["landauer_cost"].append(landauer)
        # Compute information bits
        bits = compute_information_bits(S, float(config["eps"]))  # type: ignore[arg-type]
        thermodynamic_data["info_bits"].append(bits)
        # Record metabolic cost
        thermodynamic_data["metabolic_cost"].append(result["C"])
        # Compute efficiency (C / E_min) - note: C is in arbitrary units, not Joules
        # To make the ratio meaningful, we need to convert Landauer cost to AU using the same scale factor
        # The scale factor 1e20 is used in core/threshold.py to convert Joules to neural-scale AU
        if landauer > 0:
            # Convert Joules to neural-scale AU (matches core/threshold.py)
            scale_factor = 1e20
            landauer_au = landauer * scale_factor
            efficiency = result["C"] / landauer_au
            thermodynamic_data["efficiency"].append(efficiency)
        else:
            thermodynamic_data["efficiency"].append(0.0)
        # Note: Temperature estimation disabled - requires C in Joules but our C is arbitrary units
        # To enable, convert metabolic cost to Joules using a scaling factor
        thermodynamic_data["temperature"].append(float(config["T_env"]))  # type: ignore[arg-type]
    print(f"   ✓ Simulation completed ({n_steps} steps)")
    # Display Landauer's principle
    print("\n4. Landauer's Principle (Spec §11):")
    print("   Formula: E_min = κ_meta · N_erase · k_B · T_env · ln(2)")
    print("   where N_erase ≈ log₂(S / ε)")
    # Display thermodynamic results
    print("\n5. Thermodynamic Analysis Results:")
    S_array = np.array(thermodynamic_data["S"])
    landauer_array = np.array(thermodynamic_data["landauer_cost"])
    bits_array = np.array(thermodynamic_data["info_bits"])
    cost_array = np.array(thermodynamic_data["metabolic_cost"])
    efficiency_array = np.array(thermodynamic_data["efficiency"])
    print("   Signal (S):")
    print(f"     Mean:  {np.mean(S_array):.4f}")
    print(f"     Std:   {np.std(S_array):.4f}")
    print(f"     Range: [{np.min(S_array):.4f}, {np.max(S_array):.4f}]")
    print("   Information bits (N_erase):")
    print(f"     Mean:  {np.mean(bits_array):.4f}")
    print(f"     Std:   {np.std(bits_array):.4f}")
    print(f"     Range: [{np.min(bits_array):.4f}, {np.max(bits_array):.4f}]")
    print("   Landauer cost (E_min):")
    print(f"     Mean:  {np.mean(landauer_array):.4e} J")
    print(f"     Std:   {np.std(landauer_array):.4e} J")
    print(f"     Range: [{np.min(landauer_array):.4e}, {np.max(landauer_array):.4e}] J")
    print("   Metabolic cost (C):")
    print(f"     Mean:  {np.mean(cost_array):.4f}")
    print(f"     Std:   {np.std(cost_array):.4f}")
    print(f"     Range: [{np.min(cost_array):.4f}, {np.max(cost_array):.4f}]")
    print("   Metabolic efficiency (C / E_min):")
    print(f"     Mean:  {np.mean(efficiency_array):.4f}")
    print(f"     Std:   {np.std(efficiency_array):.4f}")
    print(f"     Range: [{np.min(efficiency_array):.4f}, {np.max(efficiency_array):.4f}]")
    print("   Note: C is in arbitrary units, not Joules. Ratio shows cost excess over minimum.")
    # Validate thermodynamic constraint
    print("\n6. Thermodynamic Constraint Validation:")
    print("   Constraint: C(t) ≥ κ_meta · N_erase(t) · k_B · T_env · ln(2)")
    # Convert Landauer cost to AU for comparison with metabolic cost
    scale_factor = 1e20  # Convert Joules to neural-scale AU (matches core/threshold.py)
    landauer_array_au = landauer_array * scale_factor
    # Check constraint satisfaction
    constraint_satisfied: bool = bool(np.all(cost_array >= landauer_array_au))
    violations: int = int(np.sum(cost_array < landauer_array_au))
    print(f"   Constraint satisfied: {constraint_satisfied}")
    print(f"   Violations: {violations} / {n_steps}")
    print(f"   Satisfaction rate: {(1 - violations / n_steps) * 100:.1f}%")
    if constraint_satisfied:
        print("   ✓ Thermodynamic constraint satisfied")
    else:
        print("   ⚠ Some constraint violations detected")
        print("   (This may indicate suboptimal metabolic efficiency)")
    # Display energy statistics
    print("\n7. Energy Statistics:")
    total_landauer: float = np.sum(landauer_array_au)  # Use AU-scaled Landauer for comparison
    total_metabolic: float = np.sum(cost_array)
    print(f"   Total Landauer cost (AU): {total_landauer:.4e}")
    print(f"   Total metabolic cost (AU): {total_metabolic:.4e}")
    # A run with no ignitions has zero Landauer cost; the efficiency ratio is
    # then undefined rather than infinite. Report that instead of dividing.
    if total_landauer > 0:
        print(f"   Average efficiency: {total_metabolic / total_landauer:.4f}")
    else:
        print("   Average efficiency: n/a (no bits erased — zero Landauer cost)")
    # Display information-theoretic analysis
    print("\n8. Information-Theoretic Analysis:")
    total_bits: float = np.sum(bits_array)
    avg_bits_per_step = total_bits / n_steps
    print(f"   Total information bits: {total_bits:.2f}")
    print(f"   Average bits per step: {avg_bits_per_step:.4f}")
    print(f"   Information rate: {avg_bits_per_step / float(config['dt']):.4f} bits/ms")  # type: ignore[arg-type]
    # Display cost-benefit analysis
    print("\n9. Cost-Benefit Analysis:")
    # Find high-signal periods
    high_signal_mask = S_array > np.mean(S_array)
    low_signal_mask = S_array <= np.mean(S_array)
    if np.any(high_signal_mask):
        print("   High-signal periods (S > mean):")
        print(f"     Average Landauer cost: {np.mean(landauer_array[high_signal_mask]):.4e} J")
        print(f"     Average metabolic cost: {np.mean(cost_array[high_signal_mask]):.4f}")
        print(f"     Average efficiency: {np.mean(efficiency_array[high_signal_mask]):.4f}")
    if np.any(low_signal_mask):
        print("   Low-signal periods (S ≤ mean):")
        print(f"     Average Landauer cost: {np.mean(landauer_array[low_signal_mask]):.4e} J")
        print(f"     Average metabolic cost: {np.mean(cost_array[low_signal_mask]):.4f}")
        print(f"     Average efficiency: {np.mean(efficiency_array[low_signal_mask]):.4f}")
    print("\n" + "=" * 60)
    print("Thermodynamic analysis example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
