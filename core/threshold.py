from __future__ import annotations

from typing import Any

import numpy as np

from .deprecation import renamed_kwargs
from .thermodynamics import K_BOLTZMANN, T_ENV_DEFAULT, compute_landauer_cost


def compute_metabolic_cost(S: float, B_prev: int = 0, c1: float = 1.0, c2: float = 0.0) -> float:
    """C(t) = c1·S(t) + c2·B(t-1) per spec.
    The c2·B(t-1) term captures the metabolic cost of ignition events.
    c2=0 recovers the minimal single-term form; pass c2>0 to include it.
    """
    return float(c1 * S + c2 * int(B_prev))


@renamed_kwargs(kappa_meta="r_l", kappa_units="r_l_units")
def compute_metabolic_cost_realistic(
    S: float,
    B_prev: int,
    c1: float = 1.0,
    c2: float = 1.0,
    eps_stab: float = 1e-6,
    enforce_landauer: bool = False,
    r_l: float = 1.0,
    r_l_units: str = "dimensionless",
    bold_calibration: dict[str, Any] | None = None,
) -> float:
    """C(t)=c1*S(t)+c2*B(t-1) with optional Landauer constraint (spec §11).
    When enforce_landauer=True, ensures C(t) ≥ E_min per Landauer's principle:
        C(t) = max(c1·S + c2·B_prev, κ_meta·N_erase·k_B·T_env·ln(2))
    With BOLD calibration: C(t) = max(c1·S + c2·B_prev, E_BOLD)
    where E_BOLD is energy estimated from BOLD signal.
    Args:
        S: Signal magnitude
        B_prev: Previous ignition state
        c1: Signal cost coefficient
        c2: Ignition cost coefficient
        eps_stab: Stability threshold for bit estimation
        enforce_landauer: Whether to enforce thermodynamic constraint
        r_l: Erasure-overhead factor on the Landauer floor (formerly κ_meta,
            deprecated keyword ``kappa_meta``)
            - If r_l_units="dimensionless": dimensionless factor (default: 1.0)
            - If r_l_units="joules_per_bit": energy per bit in Joules
        r_l_units: Units of r_l (deprecated keyword ``kappa_units``) ("dimensionless" or "joules_per_bit")
        bold_calibration: Optional BOLD calibration parameters dict with keys:
            - "bold_signal_change": BOLD signal change in percent
            - "conversion_factor": Joules per 1% BOLD change per cm³ tissue
            - "tissue_volume": Tissue volume in cm³
            - "ignition_spike_factor": Energy spike factor (1.05-1.10)
    Returns:
        Metabolic cost C(t) in same units as input (AU or Joules if calibrated)
    """
    base_cost = c1 * S + c2 * B_prev
    if enforce_landauer and eps_stab < S:
        if bold_calibration is not None:
            # Use BOLD-calibrated energy estimate
            from energy.bold_calibration import (
                bold_signal_to_energy,
                estimate_ignition_energy_spike,
            )

            # Get BOLD parameters
            bold_change = bold_calibration.get("bold_signal_change", 2.0)  # default 2%
            conversion_factor = bold_calibration.get("conversion_factor", 1.2e-18)
            tissue_volume = bold_calibration.get("tissue_volume", 1.0)
            spike_factor = bold_calibration.get("ignition_spike_factor", 1.075)
            # Estimate baseline energy from BOLD (Joules)
            baseline_energy = bold_signal_to_energy(
                bold_change,
                conversion_factor=conversion_factor,
                tissue_volume=tissue_volume,
            )
            # Apply ignition spike factor; κ_meta absorbs Joules→AU unit conversion
            e_min = estimate_ignition_energy_spike(baseline_energy, spike_factor)
            e_min_scaled = e_min * r_l
        else:
            # Compute Landauer minimum using κ_meta (already applied inside)
            e_min_scaled = compute_landauer_cost(
                S=S,
                eps=eps_stab,
                k_b=K_BOLTZMANN,
                T_env=T_ENV_DEFAULT,
                r_l=r_l,
                r_l_units=r_l_units,
            )
        base_cost = max(base_cost, e_min_scaled)
    return float(base_cost)


def compute_information_value(
    z_e: float, z_i_eff: float, v1: float = 1.0, v2: float = 1.0
) -> float:
    """V(t) = v1|z_e| + v2|z_i_eff| (spec, using raw prediction errors).
    Spec is explicit: V uses unsigned raw magnitudes |z|, not φ-transformed
    values. Both positive and negative surprises carry equal informational
    value for threshold adaptation.
    """
    return float(v1 * abs(z_e) + v2 * abs(z_i_eff))


def compute_information_value_with_bias(
    z_e: float, z_i_eff: float, v1: float = 1.0, v2: float = 1.0
) -> float:
    """V(t) using raw prediction errors (§14).
    Alias for compute_information_value; maintained for API compatibility.
    """
    return compute_information_value(z_e, z_i_eff, v1, v2)


def apply_ne_threshold_modulation(theta: float, g_ne: float, gamma_ne: float) -> float:
    """θ <- θ * (1 + γ_NE * g_NE)."""
    return float(theta * (1.0 + gamma_ne * g_ne))


@renamed_kwargs(kappa="decay_exponent")
def threshold_decay(theta: float, theta_base: float, decay_exponent: float) -> float:
    """Exact decay step θ_next = θ_base + (θ − θ_base)·exp(−γ_θ·dt) (Math Spec Eq. 32).

    ``decay_exponent`` is γ_θ·dt = dt/τ_θ (see ``config.decay_exponent``). The
    keyword ``kappa`` is a deprecated alias: bare κ is reserved for ATP/bit.
    """
    if decay_exponent < 0:
        raise ValueError("decay_exponent must be >= 0")
    return float(theta_base + (theta - theta_base) * np.exp(-decay_exponent))


def update_threshold_discrete(
    theta: float,
    metabolic_cost: float,
    information_value: float,
    eta: float = 0.1,
    delta: float = 0.5,
    B_prev: int = 0,
) -> float:
    """Core allostatic update per APGI spec Section 4.
    Formula: θ(t+1) = θ(t) + η[C(t) - V(t)] + δ_reset·B(t)
    The refractory boost δ_reset·B is part of the core allostatic update,
    applied BEFORE NE modulation and ignition decision.
    Args:
        theta: Current threshold
        metabolic_cost: C(t) - metabolic cost of signal/ignition
        information_value: V(t) - expected information value of errors
        eta: Allostatic learning rate
        delta: Refractory boost magnitude (δ_reset)
        B_prev: Previous ignition state (0 or 1)
    Returns:
        Updated threshold
    """
    return float(theta + eta * (metabolic_cost - information_value) + delta * B_prev)


def apply_serotonin_threshold_offset(theta: float, beta_5ht: float) -> float:
    """Serotonergic threshold offset: θ_eff = θ + β_5HT (spec §8.4).
    5-HT encodes patience / uncertainty tolerance by raising the ignition
    threshold additively. Positive β_5HT delays premature perceptual
    commitment; zero recovers baseline behaviour.
    Theoretical grounding: Dayan & Daw (2008); Crockett et al. (2012).
    Flagged as HIGH-UNCERTAINTY mapping in spec §8.4.
    """
    return float(theta + beta_5ht)


def apply_refractory_boost(theta_next: float, B: int, delta: float) -> float:
    """Post-ignition boost: θ <- θ + δ*B."""
    return float(theta_next + delta * int(B))


def update_threshold_ode_deprecated(
    theta: float,
    theta_0: float,
    dS_dt: float,
    B_prev: int,
    gamma: float,
    delta: float,
    lam: float,
) -> float:
    """[DEPRECATED] Continuous threshold dynamics (rate of change):
    dθ/dt = γ(θ_0 - θ) + δ·B(t-1) - λ|dS/dt|.
    Warning: Derivative coupling to dS/dt is removed from the APGI spec.
    Use core.allostatic.allostatic_threshold_ode instead.
    """
    return float(gamma * (theta_0 - theta) + delta * int(B_prev) - lam * abs(dS_dt))
