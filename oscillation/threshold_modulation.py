"""Phase-locked threshold modulation for APGI.
Implements threshold modulation via oscillatory phase coupling:
θ_{t,l} = θ_{0,l} · [1 + κ · Π_{l+1} · cos(ϕ_{l+1})]
"""

from __future__ import annotations

import numpy as np


def compute_modulation_factor(
    pi_higher: float,
    phi_higher: float,
    kappa: float,
    modulation_type: str = "cosine",
) -> float:
    """Compute phase-based threshold modulation factor.
    Formula: modulation = 1 + κ · Π_higher · f(ϕ_higher)
    Where f(ϕ) is typically cos(ϕ) for rhythmic modulation.
    Args:
        pi_higher: Precision of higher level
        phi_higher: Phase of higher level oscillation (radians)
        kappa: Coupling strength
        modulation_type: "cosine" (default), "sine", or "rectified"
    Returns:
        Modulation factor (> 1 means increased threshold, < 1 means decreased)
    """
    if modulation_type == "cosine":
        phase_term = np.cos(phi_higher)
    elif modulation_type == "sine":
        phase_term = np.sin(phi_higher)
    elif modulation_type == "rectified":
        phase_term = max(0.0, np.cos(phi_higher))  # Only positive modulation
    else:
        raise ValueError(f"Unknown modulation_type: {modulation_type}")
    modulation = 1.0 + kappa * pi_higher * phase_term
    return float(modulation)


def modulate_threshold_by_phase(
    theta_baseline: float,
    pi_higher: float,
    phi_higher: float,
    kappa: float,
    modulation_type: str = "cosine",
    theta_min: float = 0.1,
    theta_max: float = 100.0,
) -> float:
    """Apply phase-based threshold modulation.
    Formula: θ_eff = θ_baseline · [1 + κ · Π_higher · cos(ϕ_higher)]
    Creates rhythmic windows of opportunity for ignition based on
    phase of higher-level oscillation.
    Args:
        theta_baseline: Baseline threshold
        pi_higher: Precision of modulating (higher) level
        phi_higher: Phase of modulating level (radians)
        kappa: Phase coupling strength
        modulation_type: Type of phase modulation
        theta_min: Minimum allowed threshold
        theta_max: Maximum allowed threshold
    Returns:
        Phase-modulated threshold
    """
    modulation = compute_modulation_factor(pi_higher, phi_higher, kappa, modulation_type)
    theta_modulated = theta_baseline * modulation
    return float(np.clip(theta_modulated, theta_min, theta_max))


def hierarchical_threshold_modulation(
    thetas: np.ndarray,
    pis: np.ndarray,
    phases: np.ndarray,
    kappa_down: float,
    pi_max: float | None = None,
) -> np.ndarray:
    """Apply phase modulation to all hierarchy levels.
    Each level is modulated by the phase of the level above it.
    Top level is not modulated (remains at baseline).
    Args:
        thetas: Baseline thresholds for each level [L]
        pis: Precisions for each level [L]
        phases: Phases for each level [L]
        kappa_down: Downward phase coupling strength
        pi_max: When given, the canonical EQ-MS-4 form (SSOT §10.7) is used:
            θ_l = θ_{0,l}·clip(1 + κ_down·(Π_{l+1}/Π_max)·cos φ_{l+1}, 0.5, 1.5).
            None keeps the unnormalized v1.1 form (SSOT-P12), for callers
            outside the pipeline only.
    Returns:
        Modulated thresholds [L]
    """
    n = len(thetas)
    thetas_mod = np.zeros(n)
    for level in range(n):
        if level < n - 1 and pi_max is not None:
            if not pi_max > 0.0:
                raise ValueError(f"pi_max must be > 0, got {pi_max}")
            factor = 1.0 + kappa_down * (pis[level + 1] / pi_max) * np.cos(phases[level + 1])
            thetas_mod[level] = thetas[level] * min(1.5, max(0.5, factor))
        elif level < n - 1:
            # Modulated by higher level
            thetas_mod[level] = modulate_threshold_by_phase(
                thetas[level], pis[level + 1], phases[level + 1], kappa_down
            )
        else:
            # Top level: no modulation
            thetas_mod[level] = thetas[level]
    return thetas_mod


def compute_phase_window(
    phi: float,
    window_center: float = 0.0,
    window_width: float = np.pi / 2,
) -> float:
    """Compute phase window factor for ignition opportunity.
    Returns a factor ∈ [0, 1] indicating how much the current phase
    overlaps with the favorable window for ignition.
    Args:
        phi: Current phase (radians)
        window_center: Center of favorable window (radians)
        window_width: Width of favorable window (radians)
    Returns:
        Window factor (1 = center of window, 0 = outside window)
    """
    # Normalize phase to [-π, π] relative to window center
    phi_normalized = ((phi - window_center + np.pi) % (2 * np.pi)) - np.pi
    if abs(phi_normalized) <= window_width / 2:
        # Inside window: cosine falloff from center
        return float(np.cos(phi_normalized * np.pi / window_width))
    else:
        # Outside window
        return 0.0


def phase_gated_ignition_probability(
    p_base: float,
    phi: float,
    window_center: float = 0.0,
    window_width: float = np.pi / 2,
) -> float:
    """Gate ignition probability by phase window.
    Reduces ignition probability outside the favorable phase window.
    Args:
        p_base: Base ignition probability
        phi: Current phase (radians)
        window_center: Center of favorable window
        window_width: Width of favorable window
    Returns:
        Phase-gated probability
    """
    window_factor = compute_phase_window(phi, window_center, window_width)
    return float(p_base * (0.5 + 0.5 * window_factor))  # Scale but don't zero out


class HierarchicalPhaseModulator:
    """Hierarchical phase modulator with broadcast capability.
    Manages phase states across multiple hierarchical levels and supports
    phase reset with optional broadcast to neighboring levels.
    """

    def __init__(self, n_levels: int, broadcast_decay: float = 0.5) -> None:
        """Initialize the hierarchical phase modulator.
        Args:
            n_levels: Number of hierarchy levels
            broadcast_decay: Decay factor for broadcast reset (default: 0.5)
        """
        self.n_levels = n_levels
        self.broadcast_decay = broadcast_decay
        self.phases = np.zeros(n_levels)

    def reset_phase_on_ignition(
        self,
        level: int,
        reset_amount: float | None = None,
        broadcast: bool = False,
        theta_coupling: float = 0.0,
    ) -> None:
        """Reset phase at a specific level, optionally broadcasting to others.
        Spec §9 canonical reset target: Δφ_reset = θ_coupling − φ_level(t)
        (default θ_coupling=0, i.e. reset to zero-phase).
        Args:
            level: Level to reset
            reset_amount: If None (default), apply the canonical
                reset-to-theta_coupling. If given explicitly, use the legacy
                fixed-additive semantics (amount added to phase, mod 2π) —
                kept for backward compatibility.
            broadcast: If True, also reset neighboring levels with decay
            theta_coupling: Target phase for the canonical reset (default 0)
        """
        # Line 196: Check for invalid level
        if level < 0 or level >= self.n_levels:
            return
        if reset_amount is None:
            delta = (theta_coupling - self.phases[level]) % (2 * np.pi)
        else:
            delta = reset_amount
        # Reset the target level
        self.phases[level] = (self.phases[level] + delta) % (2 * np.pi)
        # Lines 203-212: Broadcast to other levels
        if broadcast:
            for other_level in range(self.n_levels):
                if other_level == level:
                    continue
                # Calculate distance from reset level
                distance = abs(other_level - level)
                effective_reset = delta * (self.broadcast_decay**distance)
                self.phases[other_level] = (self.phases[other_level] + effective_reset) % (
                    2 * np.pi
                )


def phase_amplitude_coupling(phase: float, amplitude: float) -> float:
    """Compute phase-amplitude coupling factor.
    Couples phase of one oscillation with amplitude of another.
    Args:
        phase: Phase value (radians)
        amplitude: Amplitude value
    Returns:
        Coupled value: max(0, cos(phase)) * amplitude
    """
    return float(max(0.0, np.cos(phase)) * amplitude)
