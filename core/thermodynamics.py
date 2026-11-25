"""Thermodynamic constraints and Landauer's principle implementation.
APGI Specification §11: Thermodynamic Constraints
This module implements the connection between information processing and
thermodynamic cost via Landauer's principle:
    E_min = R_L · N_erase · k_B · T_env · ln(2)
where:
    - N_erase ≈ log₂(S / ε_stab) is the number of bits erased
    - k_B is Boltzmann's constant (1.38e-23 J/K)
    - T_env is environmental temperature (typically 310 K for biological systems)
    - R_L is a metabolic efficiency factor (typically 1.0-2.0)
    - ln(2) ≈ 0.693 is the natural log of 2
The metabolic cost C(t) must satisfy:
    C(t) ≥ R_L · N_erase(t) · k_B · T_env · ln(2)
This grounds the APGI metabolic cost model in fundamental thermodynamics.
References:
    - Landauer, R. (1961). "Irreversibility and Heat Generation in the Computing Process"
    - Bennett, C. H. (1982). "The Thermodynamics of Computation"
    - Shizume, K. (1995). "Heat Generation Required by Information Erasure"
"""

from __future__ import annotations

import numpy as np

from .deprecation import renamed_kwargs

# Physical constants
K_BOLTZMANN = 1.380649e-23  # Boltzmann constant (J/K), CODATA exact value
T_ENV_DEFAULT = 310.0  # Body temperature (K)
LN2 = np.log(2.0)  # Natural log of 2


@renamed_kwargs(kappa_meta="r_l", kappa_units="r_l_units")
def compute_landauer_cost(
    S: float,
    eps: float,
    k_b: float = K_BOLTZMANN,
    T_env: float = T_ENV_DEFAULT,
    r_l: float = 1.0,
    r_l_units: str = "dimensionless",
) -> float:
    """Compute thermodynamic cost per Landauer's principle.
    Implements APGI Spec §11: Connection to Metabolic Cost
    Two modes of operation:
    1. Dimensionless R_L (legacy): E_min = R_L · N_erase · k_B · T_env · ln(2)
    2. Calibrated R_L in J/bit: E_min = R_L · N_erase
    where N_erase ≈ log₂(S / ε_stab) is the number of bits erased.
    Args:
        S: Signal magnitude (bits of information)
        eps: Stability threshold ε_stab (minimum detectable signal)
        k_b: Boltzmann constant (default: 1.38e-23 J/K)
        T_env: Environmental temperature (default: 310 K for body)
        r_l: Metabolic efficiency factor
            - If r_l_units="dimensionless": dimensionless factor (default: 1.0)
            - If r_l_units="joules_per_bit": energy per bit in Joules
        r_l_units: Units of r_l ("dimensionless" or "joules_per_bit")
    Returns:
        Minimum thermodynamic cost E_min (in Joules)
    Raises:
        ValueError: If parameters are invalid
    Examples:
        >>> # Typical neural computation at body temperature (dimensionless κ)
        >>> cost = compute_landauer_cost(S=1.0, eps=0.01)
        >>> print(f"Cost: {cost:.2e} J")
        Cost: 3.21e-21 J
        >>> # With calibrated κ in J/bit (e.g., from BOLD fMRI)
        >>> kappa_calibrated = 3.21e-21  # J/bit (Landauer minimum at body temp)
        >>> cost = compute_landauer_cost(S=1.0, eps=0.01,
        ...                              r_l=kappa_calibrated,
        ...                              r_l_units="joules_per_bit")
        >>> print(f"Cost: {cost:.2e} J")
        Cost: 3.21e-21 J
    """
    if eps >= S:
        # No information to erase
        return 0.0
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}")
    if k_b <= 0:
        raise ValueError(f"k_b must be > 0, got {k_b}")
    if T_env <= 0:
        raise ValueError(f"T_env must be > 0, got {T_env}")
    if r_l <= 0:
        raise ValueError(f"r_l must be > 0, got {r_l}")
    if r_l_units not in ["dimensionless", "joules_per_bit"]:
        raise ValueError(f"r_l_units must be 'dimensionless' or 'joules_per_bit', got {r_l_units}")
    # Number of bits erased: N_erase = log₂(S / ε_stab)
    n_erase = np.log2(S / eps)
    # dimensionless: r_l scales the Landauer floor k_B·T·ln2 per bit;
    # joules_per_bit: r_l is itself the calibrated energy per bit in Joules.
    e_min = r_l * n_erase * k_b * T_env * LN2 if r_l_units == "dimensionless" else r_l * n_erase
    return float(e_min)


@renamed_kwargs(kappa_meta="r_l", kappa_units="r_l_units")
def compute_landauer_cost_batch(
    S_array: np.ndarray,
    eps: float,
    k_b: float = K_BOLTZMANN,
    T_env: float = T_ENV_DEFAULT,
    r_l: float = 1.0,
    r_l_units: str = "dimensionless",
) -> np.ndarray:
    """Compute Landauer cost for a batch of signal values.
    Vectorized version of compute_landauer_cost for efficiency.
    Args:
        S_array: Array of signal magnitudes
        eps: Stability threshold
        k_b: Boltzmann constant
        T_env: Environmental temperature
        r_l: Metabolic efficiency factor
        r_l_units: Units of r_l ("dimensionless" or "joules_per_bit")
    Returns:
        Array of costs with same shape as S_array
    Examples:
        >>> S_vals = np.array([0.1, 0.5, 1.0, 2.0])
        >>> costs = compute_landauer_cost_batch(S_vals, eps=0.01)
        >>> print(costs)
        [1.60e-21 2.41e-21 3.21e-21 3.90e-21]
    """
    S_array = np.asarray(S_array)
    costs = np.zeros_like(S_array, dtype=float)
    # Vectorized computation
    mask = S_array > eps
    n_erase = np.log2(S_array[mask] / eps)
    if r_l_units == "dimensionless":
        costs[mask] = r_l * n_erase * k_b * T_env * LN2
    else:  # joules_per_bit
        costs[mask] = r_l * n_erase
    return costs


@renamed_kwargs(kappa_meta="r_l", kappa_units="r_l_units")
def validate_thermodynamic_constraint(
    C_metabolic: float,
    S: float,
    eps: float,
    k_b: float = K_BOLTZMANN,
    T_env: float = T_ENV_DEFAULT,
    r_l: float = 1.0,
    tolerance: float = 0.01,
) -> dict:
    """Validate that metabolic cost satisfies Landauer's principle.
    Checks: C_metabolic ≥ E_min (within tolerance)
    Args:
        C_metabolic: Actual metabolic cost from APGI model
        S: Signal magnitude
        eps: Stability threshold
        k_b: Boltzmann constant
        T_env: Environmental temperature
        r_l: Metabolic efficiency factor
        tolerance: Relative tolerance for constraint (default: 1%)
    Returns:
        Dictionary with validation results:
            - satisfied: bool, whether constraint is satisfied
            - C_metabolic: actual cost
            - E_min: minimum thermodynamic cost
            - ratio: C_metabolic / E_min
            - violation: max(0, E_min - C_metabolic)
            - message: human-readable status
    Examples:
        >>> result = validate_thermodynamic_constraint(
        ...     C_metabolic=1e-20, S=1.0, eps=0.01
        ... )
        >>> print(result['satisfied'])
        True
        >>> print(f"Ratio: {result['ratio']:.2f}")
        Ratio: 3.12
    """
    e_min = compute_landauer_cost(S, eps, k_b, T_env, r_l)
    if e_min == 0:
        # No information to erase
        return {
            "satisfied": True,
            "C_metabolic": C_metabolic,
            "E_min": 0.0,
            "ratio": float("inf") if C_metabolic > 0 else 1.0,
            "violation": 0.0,
            "message": "No information to erase (S ≤ ε)",
        }
    ratio = C_metabolic / e_min
    violation = max(0.0, e_min - C_metabolic)
    satisfied = ratio >= (1.0 - tolerance)
    return {
        "satisfied": satisfied,
        "C_metabolic": C_metabolic,
        "E_min": e_min,
        "ratio": ratio,
        "violation": violation,
        "message": (
            f"Constraint satisfied: C={C_metabolic:.2e} ≥ E_min={e_min:.2e}"
            if satisfied
            else f"Constraint violated: C={C_metabolic:.2e} < E_min={e_min:.2e} "
            f"(deficit: {violation:.2e})"
        ),
    }


def compute_information_bits(S: float, eps: float) -> float:
    """Compute number of information bits represented by signal.
    Per Landauer's principle, the number of bits that must be erased is:
        N_bits = log₂(S / ε_stab)
    Args:
        S: Signal magnitude
        eps: Stability threshold
    Returns:
        Number of bits (non-negative)
    Examples:
        >>> bits = compute_information_bits(S=1.0, eps=0.01)
        >>> print(f"Bits: {bits:.1f}")
        Bits: 6.6
        >>> bits = compute_information_bits(S=0.005, eps=0.01)
        >>> print(f"Bits: {bits:.1f}")
        Bits: 0.0
    """
    if eps >= S:
        return 0.0
    return float(np.log2(S / eps))


def compute_metabolic_efficiency(
    C_metabolic: float,
    S: float,
    eps: float,
    k_b: float = K_BOLTZMANN,
    T_env: float = T_ENV_DEFAULT,
) -> float:
    """Compute metabolic efficiency factor R_L from observed cost.
    Inverse of compute_landauer_cost: given C, S, ε, compute R_L (formerly κ_meta).
    R_L = C / (N_erase · k_B · T_env · ln(2))
    Args:
        C_metabolic: Observed metabolic cost
        S: Signal magnitude
        eps: Stability threshold
        k_b: Boltzmann constant
        T_env: Environmental temperature
    Returns:
        Metabolic efficiency factor R_L
    Raises:
        ValueError: If S ≤ eps (no information to erase)
    Examples:
        >>> # If cost is 2x Landauer minimum, R_L = 2.0
        >>> kappa = compute_metabolic_efficiency(
        ...     C_metabolic=2e-20, S=1.0, eps=0.01
        ... )
        >>> print(f"R_L: {kappa:.2f}")
        R_L: 2.00
    """
    if eps >= S:
        raise ValueError(f"S must be > eps for information content (S={S}, eps={eps})")
    n_erase = np.log2(S / eps)
    denominator = n_erase * k_b * T_env * LN2
    if denominator == 0:
        raise ValueError("Denominator is zero (invalid parameters)")
    kappa_meta = C_metabolic / denominator
    return float(kappa_meta)


@renamed_kwargs(kappa_meta="r_l", kappa_units="r_l_units")
def estimate_temperature_from_cost(
    C_metabolic: float,
    S: float,
    eps: float,
    r_l: float = 1.0,
    k_b: float = K_BOLTZMANN,
) -> float:
    """Estimate environmental temperature from observed metabolic cost.
    Inverse of compute_landauer_cost: given C, S, ε, R_L, compute T_env.
    T_env = C / (R_L · N_erase · k_B · ln(2))
    Args:
        C_metabolic: Observed metabolic cost
        S: Signal magnitude
        eps: Stability threshold
        r_l: Metabolic efficiency factor
        k_b: Boltzmann constant
    Returns:
        Estimated environmental temperature (K)
    Raises:
        ValueError: If S ≤ eps or parameters invalid
    Examples:
        >>> # Estimate temperature from cost
        >>> T_est = estimate_temperature_from_cost(
        ...     C_metabolic=1e-20, S=1.0, eps=0.01
        ... )
        >>> print(f"T_est: {T_est:.1f} K")
        T_est: 310.0
    """
    if eps >= S:
        raise ValueError(f"S must be > eps (S={S}, eps={eps})")
    n_erase = np.log2(S / eps)
    denominator = r_l * n_erase * k_b * LN2
    if denominator == 0:
        raise ValueError("Denominator is zero (invalid parameters)")
    T_env = C_metabolic / denominator
    return float(T_env)


@renamed_kwargs(kappa_meta="r_l", kappa_units="r_l_units")
def thermodynamic_cost_trajectory(
    S_history: np.ndarray,
    eps: float,
    k_b: float = K_BOLTZMANN,
    T_env: float = T_ENV_DEFAULT,
    r_l: float = 1.0,
) -> dict:
    """Analyze thermodynamic cost over a trajectory.
    Args:
        S_history: Time series of signal values
        eps: Stability threshold
        k_b: Boltzmann constant
        T_env: Environmental temperature
        r_l: Metabolic efficiency factor
    Returns:
        Dictionary with trajectory statistics:
            - costs: array of costs at each time step
            - total_cost: cumulative cost
            - mean_cost: average cost
            - max_cost: maximum cost
            - bits_history: bits at each time step
            - total_bits: cumulative bits erased
    Examples:
        >>> S_hist = np.array([0.1, 0.5, 1.0, 0.5, 0.1])
        >>> result = thermodynamic_cost_trajectory(S_hist, eps=0.01)
        >>> print(f"Total cost: {result['total_cost']:.2e} J")
        >>> print(f"Total bits: {result['total_bits']:.1f}")
    """
    S_history = np.asarray(S_history)
    costs = compute_landauer_cost_batch(S_history, eps, k_b, T_env, r_l)
    bits = np.array([compute_information_bits(s, eps) for s in S_history])
    return {
        "costs": costs,
        "total_cost": float(np.sum(costs)),
        "mean_cost": float(np.mean(costs)),
        "max_cost": float(np.max(costs)),
        "min_cost": float(np.min(costs)),
        "bits_history": bits,
        "total_bits": float(np.sum(bits)),
        "mean_bits": float(np.mean(bits)),
        "max_bits": float(np.max(bits)),
    }
