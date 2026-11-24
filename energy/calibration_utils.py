"""Utilities for calibrating BOLD-to-energy conversion parameters.
This module helps set calibration parameters to produce realistic erasure
overheads R_L in the range of biological systems (10^3-10^9 × Landauer minimum).
R_L is dimensionless (Notation Appendix §11); κ_meta is deprecated.
"""

from __future__ import annotations

from typing import Any

from core.deprecation import RenamedKeyDict, warn_renamed

from .bold_calibration import DEFAULT_TISSUE_VOLUME, compute_landauer_energy_per_bit


def calibrate_for_realistic_r_l(
    target_r_l: float = 1000.0,
    typical_bold_change: float = 2.0,
    typical_bits: float = 6.6,
    T: float = 310.0,
) -> dict:
    """Calibrate parameters to produce a realistic erasure overhead R_L.
    Biological systems are highly inefficient compared to Landauer's minimum.
    This function sets calibration parameters so that BOLD energy sits at
    R_L × the Landauer floor.
    Args:
        target_r_l: Desired R_L, a multiple of the Landauer minimum
            (e.g., 1000 for 1000× Landauer minimum; formerly ``target_kappa_multiple``)
        typical_bold_change: Typical BOLD signal change in percent
        typical_bits: Typical bits erased during ignition
        T: Temperature for Landauer calculation
    Returns:
        Dictionary with calibrated parameters
    """
    # Landauer minimum per bit
    e_min_per_bit = compute_landauer_energy_per_bit(T)
    # Energy per bit at the target overhead, in J/bit (R_L × floor)
    energy_per_bit = target_r_l * e_min_per_bit
    # Required energy from BOLD for these bits
    required_energy = energy_per_bit * typical_bits
    # Calibrate conversion factor to produce this energy
    # Energy = conversion_factor * bold_change * tissue_volume
    # Assuming baseline_energy is negligible for calibration
    calibrated_factor = required_energy / (typical_bold_change * DEFAULT_TISSUE_VOLUME)
    return RenamedKeyDict(
        {
            "target_r_l": target_r_l,
            "target_energy_j_per_bit": energy_per_bit,
            "calibrated_conversion_factor": calibrated_factor,
            "typical_bold_change": typical_bold_change,
            "typical_bits": typical_bits,
            "landauer_minimum_j_per_bit": e_min_per_bit,
            "temperature_k": T,
        },
        {
            "target_kappa_multiple": "target_r_l",
            "target_kappa_j_per_bit": "target_energy_j_per_bit",
        },
    )


def calibrate_for_realistic_kappa(target_kappa_multiple: float = 1000.0, **kwargs: Any) -> dict:
    """Deprecated alias of :func:`calibrate_for_realistic_r_l` (κ_meta is retired)."""
    warn_renamed("calibrate_for_realistic_kappa", "calibrate_for_realistic_r_l")
    return calibrate_for_realistic_r_l(target_r_l=target_kappa_multiple, **kwargs)


def create_realistic_calibrator(
    target_efficiency: float = 1000.0,
    **kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    """Create a BOLDCalibrator with realistic parameters.
    Args:
        target_efficiency: Desired efficiency as multiple of Landauer minimum
        **kwargs: Additional arguments for calibrate_for_realistic_r_l
    Returns:
        BOLDCalibrator configured for realistic biological energy costs
    """
    from .bold_calibration import BOLDCalibrator

    # Get calibrated parameters
    calibration = calibrate_for_realistic_r_l(
        target_r_l=target_efficiency,
        **kwargs,
    )
    # Create calibrator with these parameters
    calibrator = BOLDCalibrator(
        conversion_factor=calibration["calibrated_conversion_factor"],
        tissue_volume=DEFAULT_TISSUE_VOLUME,
        ignition_spike_factor=1.075,
        T=calibration["temperature_k"],
    )
    return calibrator, calibration


def demonstrate_calibration_range() -> None:
    """Demonstrate calibration across different efficiency ranges."""
    print("BOLD Calibration for Different Biological Efficiency Levels")
    print("=" * 60)
    efficiency_levels = [
        ("Theoretical minimum", 1.0),
        ("Highly efficient synthetic", 10.0),
        ("Optimized biological", 100.0),
        ("Typical neural", 1000.0),
        ("Inefficient biological", 10000.0),
        ("Pathological", 1000000.0),
    ]
    for name, efficiency in efficiency_levels:
        calibration = calibrate_for_realistic_r_l(
            target_r_l=efficiency,
        )
        print(f"\n{name} (R_L = {efficiency:.0f}× Landauer):")
        print(f"  Energy per bit: {calibration['target_energy_j_per_bit']:.2e} J/bit")
        print(f"  Conversion factor: {calibration['calibrated_conversion_factor']:.2e} J/%/cm³")
        print(f"  For 2% BOLD, 6.6 bits: {calibration['target_energy_j_per_bit'] * 6.6:.2e} J")
    print("\n" + "=" * 60)
    print("Note: Default parameters produce R_L ~ 1000× Landauer minimum")
    print("Adjust target_r_l for different biological scenarios")


if __name__ == "__main__":
    demonstrate_calibration_range()
