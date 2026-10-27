"""Circadian and ultradian θₜ modulation — APGI spec §27 (peripheral predictions).
The APGI framework predicts that the ignition threshold θₜ is modulated by
two biological rhythms:
  - Circadian (~24h): driven by the suprachiasmatic nucleus (SCN); encodes
    time-of-day variation in alertness and metabolic state.
  - Ultradian (~90 min, BRAC): Basic Rest-Activity Cycle; encodes within-day
    oscillations in cortical excitability and REM/NREM-like transitions.
Both are implemented as additive cosine offsets on θₜ:
    Δθ_circ(t)     = A_circ     · cos(2π·t/T_circ     + φ_circ)
    Δθ_ultradian(t) = A_ultradian · cos(2π·t/T_ultradian + φ_ultradian)
    θ_eff(t)       = θ_base + Δθ_circ(t) + Δθ_ultradian(t)
Falsification criteria (§27 Tier 3 — Peripheral):
  - Null result on circadian θₜ modulation below detection threshold in
    specific populations does NOT invalidate the core framework.
  - Ultradian ~90-min oscillations not detectable in vigilance paradigms
    leaves the core intact.
References:
  - Daan, S., & Beersma, D. G. M. (1984). Circadian gating of sleep onset.
  - Kleitman, N. (1982). Basic rest-activity cycle — 22 years later.
  - Schmidt, C., et al. (2012). Circadian sleep regulation influences
    subjective alertness and sleep propensity. Chronobiology International.
"""

from __future__ import annotations

import numpy as np

# ── Physical / biological constants ──────────────────────────────────────────
T_CIRCADIAN_DEFAULT: float = 86400.0  # 24 h in seconds
T_ULTRADIAN_DEFAULT: float = 5400.0  # 90 min in seconds

# ── Scalar offset functions ───────────────────────────────────────────────────


def circadian_theta_offset(
    t: float,
    A_circ: float = 0.1,
    T_circ: float = T_CIRCADIAN_DEFAULT,
    phi_circ: float = 0.0,
) -> float:
    """Circadian additive offset on θₜ.
    Δθ_circ(t) = A_circ · cos(2π·t/T_circ + φ_circ)
    Args:
        t: Time in seconds (absolute, e.g. seconds since midnight)
        A_circ: Amplitude of circadian modulation (default: 0.1 × θ_base units)
            Positive peak corresponds to highest arousal / lowest threshold.
        T_circ: Period of circadian rhythm in seconds (default: 86400 s = 24 h)
        phi_circ: Phase offset in radians (default: 0; peak at t=0)
            Use φ_circ = π to flip phase (lowest arousal at t=0).
    Returns:
        Scalar additive offset Δθ_circ(t)
    """
    if T_circ <= 0:
        raise ValueError(f"T_circ must be > 0, got {T_circ}")
    return float(A_circ * np.cos(2.0 * np.pi * t / T_circ + phi_circ))


def ultradian_theta_offset(
    t: float,
    A_ultradian: float = 0.05,
    T_ultradian: float = T_ULTRADIAN_DEFAULT,
    phi_ultradian: float = 0.0,
) -> float:
    """Ultradian (~90 min, BRAC) additive offset on θₜ.
    Δθ_ultradian(t) = A_ultradian · cos(2π·t/T_ultradian + φ_ultradian)
    Args:
        t: Time in seconds
        A_ultradian: Amplitude of ultradian modulation (default: 0.05)
            Typically half the circadian amplitude — smaller excitability swings.
        T_ultradian: Period of BRAC in seconds (default: 5400 s = 90 min)
        phi_ultradian: Phase offset in radians (default: 0)
    Returns:
        Scalar additive offset Δθ_ultradian(t)
    """
    if T_ultradian <= 0:
        raise ValueError(f"T_ultradian must be > 0, got {T_ultradian}")
    return float(A_ultradian * np.cos(2.0 * np.pi * t / T_ultradian + phi_ultradian))


def combined_biological_rhythm_offset(
    t: float,
    A_circ: float = 0.1,
    T_circ: float = T_CIRCADIAN_DEFAULT,
    phi_circ: float = 0.0,
    A_ultradian: float = 0.05,
    T_ultradian: float = T_ULTRADIAN_DEFAULT,
    phi_ultradian: float = 0.0,
) -> float:
    """Combined circadian + ultradian additive offset on θₜ.
    Δθ(t) = A_circ · cos(2π·t/T_circ + φ_circ)
           + A_ultradian · cos(2π·t/T_ultradian + φ_ultradian)
    The two rhythms are superimposed linearly — they are driven by
    independent biological clocks (SCN vs. brainstem BRAC generator).
    Args:
        t: Time in seconds
        A_circ: Circadian amplitude (default: 0.1)
        T_circ: Circadian period in seconds (default: 86400)
        phi_circ: Circadian phase offset in radians (default: 0)
        A_ultradian: Ultradian amplitude (default: 0.05)
        T_ultradian: Ultradian period in seconds (default: 5400)
        phi_ultradian: Ultradian phase offset in radians (default: 0)
    Returns:
        Total additive offset Δθ(t) = Δθ_circ + Δθ_ultradian
    """
    return circadian_theta_offset(t, A_circ, T_circ, phi_circ) + ultradian_theta_offset(
        t, A_ultradian, T_ultradian, phi_ultradian
    )


def apply_biological_rhythm_to_theta(
    theta_base: float,
    t: float,
    A_circ: float = 0.1,
    T_circ: float = T_CIRCADIAN_DEFAULT,
    phi_circ: float = 0.0,
    A_ultradian: float = 0.05,
    T_ultradian: float = T_ULTRADIAN_DEFAULT,
    phi_ultradian: float = 0.0,
    theta_min: float = 0.0,
) -> float:
    """Apply combined biological rhythm offset to a baseline threshold.
    θ_eff(t) = max(θ_min, θ_base + Δθ_circ(t) + Δθ_ultradian(t))
    The floor θ_min prevents the rhythm from driving θ negative, which
    would make ignition unconditionally certain (biologically implausible).
    Args:
        theta_base: Baseline ignition threshold θ_base
        t: Time in seconds
        A_circ: Circadian amplitude
        T_circ: Circadian period (seconds)
        phi_circ: Circadian phase offset (radians)
        A_ultradian: Ultradian amplitude
        T_ultradian: Ultradian period (seconds)
        phi_ultradian: Ultradian phase offset (radians)
        theta_min: Minimum allowed threshold (default: 0.0)
    Returns:
        Rhythmically modulated threshold θ_eff(t) ≥ theta_min
    """
    delta = combined_biological_rhythm_offset(
        t, A_circ, T_circ, phi_circ, A_ultradian, T_ultradian, phi_ultradian
    )
    return float(max(theta_min, theta_base + delta))


# ── Vectorized variants ───────────────────────────────────────────────────────


def circadian_theta_offset_array(
    t_array: np.ndarray,
    A_circ: float = 0.1,
    T_circ: float = T_CIRCADIAN_DEFAULT,
    phi_circ: float = 0.0,
) -> np.ndarray:
    """Vectorized circadian offset over a time array."""
    if T_circ <= 0:
        raise ValueError(f"T_circ must be > 0, got {T_circ}")
    t = np.asarray(t_array, dtype=float)
    return np.asarray(A_circ * np.cos(2.0 * np.pi * t / T_circ + phi_circ))


def ultradian_theta_offset_array(
    t_array: np.ndarray,
    A_ultradian: float = 0.05,
    T_ultradian: float = T_ULTRADIAN_DEFAULT,
    phi_ultradian: float = 0.0,
) -> np.ndarray:
    """Vectorized ultradian offset over a time array."""
    if T_ultradian <= 0:
        raise ValueError(f"T_ultradian must be > 0, got {T_ultradian}")
    t = np.asarray(t_array, dtype=float)
    return np.asarray(A_ultradian * np.cos(2.0 * np.pi * t / T_ultradian + phi_ultradian))


# ── Stateful regulator ────────────────────────────────────────────────────────


class CircadianRegulator:
    """Stateful biological rhythm regulator for incremental pipeline integration.
    Tracks elapsed time internally and exposes a single `theta_offset()`
    call per timestep, making it drop-in compatible with the allostatic ODE:
        θ(t+dt) = allostatic_ode(...) + regulator.theta_offset()
    Example::
        reg = CircadianRegulator(dt=1.0)   # 1-second timesteps
        for step in range(n_steps):
            theta_eff = theta_base + reg.theta_offset()
            reg.tick()
    """

    def __init__(
        self,
        t0: float = 0.0,
        dt: float = 1.0,
        A_circ: float = 0.1,
        T_circ: float = T_CIRCADIAN_DEFAULT,
        phi_circ: float = 0.0,
        A_ultradian: float = 0.05,
        T_ultradian: float = T_ULTRADIAN_DEFAULT,
        phi_ultradian: float = 0.0,
    ) -> None:
        """Initialize the circadian regulator.
        Args:
            t0: Initial time in seconds (default: 0 — start of day)
            dt: Timestep size in seconds (default: 1.0 s)
            A_circ: Circadian amplitude (default: 0.1)
            T_circ: Circadian period in seconds (default: 86400)
            phi_circ: Circadian phase offset in radians (default: 0)
            A_ultradian: Ultradian amplitude (default: 0.05)
            T_ultradian: Ultradian period in seconds (default: 5400)
            phi_ultradian: Ultradian phase offset in radians (default: 0)
        """
        self.t = float(t0)
        self.dt = float(dt)
        self.A_circ = A_circ
        self.T_circ = T_circ
        self.phi_circ = phi_circ
        self.A_ultradian = A_ultradian
        self.T_ultradian = T_ultradian
        self.phi_ultradian = phi_ultradian

    def theta_offset(self) -> float:
        """Return combined rhythm offset at current time t."""
        return combined_biological_rhythm_offset(
            self.t,
            self.A_circ,
            self.T_circ,
            self.phi_circ,
            self.A_ultradian,
            self.T_ultradian,
            self.phi_ultradian,
        )

    def tick(self) -> None:
        """Advance internal clock by dt."""
        self.t += self.dt

    def reset(self, t0: float = 0.0) -> None:
        """Reset clock to t0 (default: 0)."""
        self.t = float(t0)

    @property
    def current_time(self) -> float:
        """Current time in seconds."""
        return self.t
