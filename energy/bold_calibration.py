"""BOLD signal calibration and energy conversion for APGI thermodynamic grounding.
This module provides calibrated conversion from BOLD fMRI signals to energy
consumption in Joules, enabling direct comparison with Landauer's physical minimum.
Key concepts:
1. BOLD signal reflects hemodynamic response to neural activity
2. Energy spike during ignition events: 5-10% increase in metabolic rate
3. Conversion to Joules via calibrated factors from fMRI literature
References:
- Logothetis, N. K. (2008). "What we can do and what we cannot do with fMRI"
- Attwell, D., & Laughlin, S. B. (2001). "An energy budget for signaling in the grey matter"
- Raichle, M. E., & Mintun, M. A. (2006). "Brain work and brain imaging"
- Shulman, R. G., et al. (2004). "Energetic basis of brain activity"
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.deprecation import RenamedKeyDict, renamed_kwargs, warn_renamed

# Physical constants
K_BOLTZMANN = 1.380649e-23  # Boltzmann constant (J/K), CODATA exact value
T_BODY = 310.0  # Body temperature (K, 37°C)
LN2 = np.log(2.0)  # Natural log of 2
# Empirical calibration constants from fMRI literature
# --------------------------------------------------
# Typical BOLD signal changes: 1-5% for cognitive tasks
# Energy spike during ignition: 5-10% increase
# Conversion factors based on human neuroimaging studies
# MathSpec §11 "BOLD fMRI Calibration of κ_meta": CF is the conversion
# factor (Joules per 1% BOLD change per cm³ tissue), empirically ~1.2e-18
# J/(%·cm³) from fMRI literature. This module previously used a
# reverse-engineered value (2.59e-38, ~20 orders of magnitude smaller) whose
# own comment disclosed it was chosen post-hoc to hit a specific κ_meta
# target for one internal test scenario rather than the literature value —
# config.py and core/config_schema.py already used the correct spec value
# (1.2e-18) for end-to-end pipeline runs, but direct callers of this
# module's functions (calibration_utils.py, examples) got the mismatched
# default. Corrected to match the spec-cited empirical value.
DEFAULT_BOLD_TO_ENERGY_FACTOR = 1.2e-18  # Joules per 1% BOLD signal change per cm³ tissue
DEFAULT_TISSUE_VOLUME = 1.0  # cm³ (typical voxel volume)
DEFAULT_IGNITION_SPIKE_FACTOR = 1.075  # 7.5% energy spike during ignition (midpoint of 5-10%)
# ATP conversion
ATP_ENERGY = 5.2e-21  # Energy per ATP molecule at ~300K (~50 kJ/mol)
TYPICAL_ATP_PER_BIT = 100.0  # Typical biological cost: ~100 ATP molecules per bit processed


def compute_landauer_energy_per_bit(T: float = T_BODY) -> float:
    """Compute Landauer's minimum energy to erase one bit.
    E_min = k_B * T * ln(2)
    Args:
        T: Temperature in Kelvin (default: body temperature 310K)
    Returns:
        Minimum energy in Joules per bit
    """
    return float(K_BOLTZMANN * T * LN2)


def bold_signal_to_energy(
    bold_signal_change: float,
    baseline_energy: float = 0.0,  # Baseline energy (set to 0 for calibration)
    conversion_factor: float = DEFAULT_BOLD_TO_ENERGY_FACTOR,
    tissue_volume: float = DEFAULT_TISSUE_VOLUME,
) -> float:
    """Convert BOLD signal change to energy consumption.
    Based on fMRI literature: BOLD signal ~1% corresponds to metabolic energy
    Note: Parameters adjusted to produce κ_meta in realistic biological range
    Args:
        bold_signal_change: BOLD signal change in percent (e.g., 2.5 for 2.5%)
        baseline_energy: Baseline energy consumption in J/cm³/s
        conversion_factor: Joules per 1% BOLD change per cm³ tissue
        tissue_volume: Tissue volume in cm³
    Returns:
        Energy consumption in Joules per second
    """
    # Total energy = baseline + BOLD-induced change
    energy = (
        baseline_energy * tissue_volume
        + conversion_factor * abs(bold_signal_change) * tissue_volume
    )
    return energy


def estimate_ignition_energy_spike(
    baseline_energy: float,
    spike_factor: float = DEFAULT_IGNITION_SPIKE_FACTOR,
) -> float:
    """Estimate energy spike during ignition event.
    During ignition events, fMRI shows 5-10% energy spike.
    Args:
        baseline_energy: Baseline energy consumption
        spike_factor: Multiplication factor for spike (1.05-1.10)
    Returns:
        Peak energy during ignition spike
    """
    return baseline_energy * spike_factor


def calibrate_r_l_from_bold(
    bold_signal_change: float,
    bits_erased: float,
    T: float = T_BODY,
    conversion_factor: float = DEFAULT_BOLD_TO_ENERGY_FACTOR,
    tissue_volume: float = DEFAULT_TISSUE_VOLUME,
) -> float:
    """Estimate the Landauer erasure-consistency ratio R_L from a BOLD signal.
    R_L = E_BOLD / (N_erase · k_B · T · ln 2)   (Notation Appendix §11)
    R_L is dimensionless: joules divided by (bits × joules per bit). The former
    ``calibrate_kappa_meta_from_bold`` returned the same number but labelled it
    J/bit; κ_meta is deprecated. Because N_erase is an upper bound, R_L is a
    lower bound on the erasure overhead.
    Args:
        bold_signal_change: BOLD signal change in percent
        bits_erased: Number of bits erased during ignition
        T: Temperature for Landauer calculation
        conversion_factor: BOLD to energy conversion factor
        tissue_volume: Tissue volume
    Returns:
        R_L (dimensionless)
    """
    if bits_erased <= 0:
        return 0.0
    # Energy from BOLD signal
    e_bold = bold_signal_to_energy(
        bold_signal_change,
        conversion_factor=conversion_factor,
        tissue_volume=tissue_volume,
    )
    # Landauer minimum for these bits
    e_min_per_bit = compute_landauer_energy_per_bit(T)
    e_min_total = bits_erased * e_min_per_bit
    return e_bold / e_min_total if e_min_total > 0 else 0.0


def calibrate_kappa_meta_from_bold(*args: Any, **kwargs: Any) -> float:
    """Deprecated alias of :func:`calibrate_r_l_from_bold` (κ_meta is retired)."""
    warn_renamed("calibrate_kappa_meta_from_bold", "calibrate_r_l_from_bold")
    return calibrate_r_l_from_bold(*args, **kwargs)


@renamed_kwargs(kappa_meta="r_l")
def estimate_bits_from_bold_energy(
    bold_signal_change: float,
    r_l: float,
    T: float = T_BODY,
    conversion_factor: float = DEFAULT_BOLD_TO_ENERGY_FACTOR,
    tissue_volume: float = DEFAULT_TISSUE_VOLUME,
) -> float:
    """Estimate bits erased from a BOLD signal and R_L.
    Inverse of calibrate_r_l_from_bold.
    N_erase = E_BOLD / (R_L · E_min_per_bit)
    Args:
        bold_signal_change: BOLD signal change in percent
        r_l: Erasure-consistency ratio R_L (dimensionless; formerly ``kappa_meta``)
        T: Temperature for Landauer calculation
        conversion_factor: BOLD to energy conversion factor
        tissue_volume: Tissue volume
    Returns:
        Estimated bits erased
    """
    if r_l <= 0:
        return 0.0
    # Energy from BOLD signal
    e_bold = bold_signal_to_energy(
        bold_signal_change,
        conversion_factor=conversion_factor,
        tissue_volume=tissue_volume,
    )
    # Landauer minimum per bit
    e_min_per_bit = compute_landauer_energy_per_bit(T)
    # Estimated bits
    bits_estimated = e_bold / (r_l * e_min_per_bit)
    return max(bits_estimated, 0.0)


def compute_energy_with_ignition_spike(
    baseline_bold: float,
    ignition_bold: float,
    duration: float = 1.0,  # seconds
    conversion_factor: float = DEFAULT_BOLD_TO_ENERGY_FACTOR,
    tissue_volume: float = DEFAULT_TISSUE_VOLUME,
) -> dict:
    """Compute total energy with ignition spike.
    Models the 5-10% energy spike during ignition events.
    Args:
        baseline_bold: Baseline BOLD signal in percent
        ignition_bold: Peak BOLD during ignition in percent
        duration: Time duration in seconds
        conversion_factor: BOLD to energy conversion factor
        tissue_volume: Tissue volume
    Returns:
        Dictionary with energy breakdown
    """
    # Baseline energy
    e_baseline = (
        bold_signal_to_energy(
            baseline_bold,
            conversion_factor=conversion_factor,
            tissue_volume=tissue_volume,
        )
        * duration
    )
    # Ignition spike energy (integrated over time)
    # Assuming triangular spike shape for simplicity
    e_spike = (
        bold_signal_to_energy(
            ignition_bold,
            conversion_factor=conversion_factor,
            tissue_volume=tissue_volume,
        )
        * duration
        * 0.5
    )  # Triangular approximation
    # Total energy
    e_total = e_baseline + e_spike
    # Spike factor
    spike_factor = ignition_bold / max(baseline_bold, 1e-6)
    return {
        "baseline_energy_j": e_baseline,
        "spike_energy_j": e_spike,
        "total_energy_j": e_total,
        "spike_factor": spike_factor,
        "baseline_bold_percent": baseline_bold,
        "ignition_bold_percent": ignition_bold,
        "duration_s": duration,
    }


def validate_energy_against_landauer(
    measured_energy: float,
    bits_erased: float,
    T: float = T_BODY,
    tolerance: float = 0.01,
) -> dict:
    """Validate measured energy against Landauer's principle.
    Checks: E_measured ≥ E_min (within tolerance)
    Args:
        measured_energy: Measured energy in Joules
        bits_erased: Number of bits erased
        T: Temperature for Landauer calculation
        tolerance: Relative tolerance (default: 1%)
    Returns:
        Validation results dictionary
    """
    # Landauer minimum
    e_min_per_bit = compute_landauer_energy_per_bit(T)
    e_min_total = bits_erased * e_min_per_bit
    if e_min_total == 0:
        return {
            "satisfied": True,
            "measured_energy_j": measured_energy,
            "landauer_minimum_j": 0.0,
            "ratio": float("inf") if measured_energy > 0 else 1.0,
            "violation_j": 0.0,
            "bits_erased": bits_erased,
            "temperature_k": T,
            "message": "No information to erase",
        }
    ratio = measured_energy / e_min_total
    violation = max(0.0, e_min_total - measured_energy)
    satisfied = ratio >= (1.0 - tolerance)
    return {
        "satisfied": satisfied,
        "measured_energy_j": measured_energy,
        "landauer_minimum_j": e_min_total,
        "ratio": ratio,
        "violation_j": violation,
        "bits_erased": bits_erased,
        "temperature_k": T,
        "message": (
            f"Constraint satisfied: E={measured_energy:.2e} J ≥ E_min={e_min_total:.2e} J"
            if satisfied
            else f"Constraint violated: E={measured_energy:.2e} J < E_min={e_min_total:.2e} J "
            f"(deficit: {violation:.2e} J)"
        ),
    }


class BOLDCalibrator:
    """Calibrate APGI energy model using BOLD fMRI data."""

    def __init__(
        self,
        conversion_factor: float = DEFAULT_BOLD_TO_ENERGY_FACTOR,
        tissue_volume: float = DEFAULT_TISSUE_VOLUME,
        ignition_spike_factor: float = DEFAULT_IGNITION_SPIKE_FACTOR,
        T: float = T_BODY,
    ):
        """Initialize BOLD calibrator.
        Args:
            conversion_factor: Joules per 1% BOLD change per cm³ tissue
            tissue_volume: Tissue volume in cm³
            ignition_spike_factor: Energy spike factor during ignition (1.05-1.10)
            T: Temperature for Landauer calculations
        """
        self.conversion_factor = conversion_factor
        self.tissue_volume = tissue_volume
        self.ignition_spike_factor = ignition_spike_factor
        self.T = T
        # Calibration results
        self.calibrated_r_l: float | None = None
        self.calibration_data: list[dict[str, Any]] = []

    def calibrate_from_trial(
        self,
        baseline_bold: float,
        ignition_bold: float,
        estimated_bits: float,
        duration: float = 1.0,
    ) -> float:
        """Estimate R_L from a single trial with BOLD measurements.
        Args:
            baseline_bold: Baseline BOLD signal in percent
            ignition_bold: Peak BOLD during ignition in percent
            estimated_bits: Estimated bits erased during ignition
            duration: Trial duration in seconds
        Returns:
            R_L for this trial (dimensionless)
        """
        # Compute energy with ignition spike
        energy_result = compute_energy_with_ignition_spike(
            baseline_bold,
            ignition_bold,
            duration,
            self.conversion_factor,
            self.tissue_volume,
        )
        # R_L = E / (N_erase · E_min per bit), dimensionless
        total_energy = energy_result["total_energy_j"]
        if estimated_bits > 0:
            r_l = total_energy / (estimated_bits * compute_landauer_energy_per_bit(self.T))
        else:
            r_l = 0.0
        # Store calibration data
        calibration_record = RenamedKeyDict(
            {
                "baseline_bold": baseline_bold,
                "ignition_bold": ignition_bold,
                "estimated_bits": estimated_bits,
                "total_energy_j": total_energy,
                "r_l_calibrated": r_l,
                "spike_factor": energy_result["spike_factor"],
            },
            {"kappa_calibrated": "r_l_calibrated"},
        )
        self.calibration_data.append(calibration_record)
        # Update overall calibrated R_L (average)
        if self.calibrated_r_l is None:
            self.calibrated_r_l = r_l
        else:
            # Weighted average based on energy magnitude
            self.calibrated_r_l = (self.calibrated_r_l * len(self.calibration_data) + r_l) / (
                len(self.calibration_data) + 1
            )
        return float(r_l)

    @property
    def calibrated_kappa(self) -> float | None:
        """Deprecated alias of :attr:`calibrated_r_l` (κ_meta is retired)."""
        warn_renamed("calibrated_kappa", "calibrated_r_l")
        return self.calibrated_r_l

    def get_calibration_summary(self) -> dict:
        """Get summary of calibration results."""
        if not self.calibration_data:
            return {"calibrated": False, "message": "No calibration data"}
        ratios = [d["r_l_calibrated"] for d in self.calibration_data]
        energies = [d["total_energy_j"] for d in self.calibration_data]
        spike_factors = [d["spike_factor"] for d in self.calibration_data]
        summary = {
            "calibrated": True,
            "r_l_mean": float(np.mean(ratios)),
            "r_l_std": float(np.std(ratios)),
            "r_l_median": float(np.median(ratios)),
            "energy_mean_j": float(np.mean(energies)),
            "spike_factor_mean": float(np.mean(spike_factors)),
            "n_trials": len(self.calibration_data),
            "conversion_factor": self.conversion_factor,
            "tissue_volume": self.tissue_volume,
            "temperature_k": self.T,
            "landauer_energy_per_bit_j": compute_landauer_energy_per_bit(self.T),
        }
        return RenamedKeyDict(
            summary,
            {"kappa_mean": "r_l_mean", "kappa_std": "r_l_std", "kappa_median": "r_l_median"},
        )

    def validate_against_landauer(self, measured_energy: float, bits_erased: float) -> dict:
        """Validate measured energy against Landauer's principle using calibrated parameters."""
        return validate_energy_against_landauer(
            measured_energy,
            bits_erased,
            self.T,
        )
