"""Cross-Level Threshold Resonance — Russian Doll Architecture.
Implements the nested resonance system from §8 / §9 of the APGI Full Specs:
  θ_l[t] = θ_{0,l} · [1 + κ_down · Π_{l+1} · cos(φ_{l+1})]   (§8 spec formula)
  dφ_l/dt = ω_l + κ_down · sin(φ_{l+1} − φ_l)                 (top-down entrainment)
  S_l[t+1] = (1 − λ_l) · S_l[t] + λ_l · S_inst_l[t]          (per-level accumulator)
Each level l owns:
  φ_l  — oscillatory phase (advances at natural frequency ω_l = 2π / τ_l)
  θ_l  — threshold continuously re-shaped by the cosine of the level above
  S_l  — ignition signal accumulated from level-specific salience
Higher levels oscillate more slowly (large τ_l).  The top-down phase-coupling
term sin(φ_{l+1} − φ_l) pulls each level toward its superior's phase, creating
nested rhythmic windows of ignition opportunity — the Russian Doll property.

Index-0 convention (Notation Appendix, "Level Architecture"): the per-level
arrays here (θ, S, φ, Π) represent spec levels 1..L only, i.e. the APGI
*ignition* hierarchy (sensory through abstract/self). Index 0 corresponds to
spec Level 1 (first APGI ignition level, primary sensory), never to spec
Level 0 (the sub-APGI reflexive brainstem/collicular substrate). Level 0 does
not participate in the precision-gated ignition mechanism and is never
allocated a slot in this system — it is architecturally excluded, not merely
weighted low.

The primary ignition check uses θ_0[t] (index 0 = spec Level 1), which is
continuously modulated by the phase and precision of index 1 (spec Level 2).
Index 1's θ_1 is modulated by index 2 (spec Level 3), etc., nesting the
hierarchy.
When an ignition event fires at index 0, a post-ignition refractory boost
δ_refractory is added to θ_0 and its signal is partially reset (ρ_S),
matching §17.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.rng import RNGLike, resolve_rng


@dataclass
class LevelState:
    """State variables owned by one level in the resonance system."""

    S: float  # accumulated ignition signal
    theta: float  # phase-modulated threshold (updated each step)
    phi: float  # oscillatory phase [0, 2π)
    pi: float  # precision (passed in from the precision pipeline)


class NestedResonanceSystem:
    """Per-level phase-modulated threshold resonance (Russian Doll architecture).
    Parameters
    ----------
    n_levels : int
        Number of cortical hierarchy levels L.
    theta_0 : array-like, shape (L,)
        Baseline thresholds for each level.
    omega : array-like, shape (L,)
        Natural angular frequencies ω_l = 2π / τ_l.
    lambda_rates : array-like, shape (L,)
        Per-level leaky-accumulator rates λ_l = dt / τ_l (clamped to (0,1)).
    kappa_down : float
        Top-down coupling strength κ_down (both phase entrainment and
        threshold modulation use the same parameter per spec §8).
    kappa_up : float
        Bottom-up suppression strength (post-ignition cascade).
    theta_min, theta_max : float
        Hard clamps on modulated thresholds.
    phi_noise_std : float
        Per-step phase jitter amplitude (biological phase jitter). 0.0
        (default) disables noise entirely (deterministic phase advancement).
    use_ou_phase_noise : bool
        When phi_noise_std > 0, use an Ornstein-Uhlenbeck (colored) noise
        process per level (default True) rather than white noise. MathSpec
        §9: "The OU process produces coloured phase noise more consistent
        with neural oscillation variability than white noise" — white noise
        was previously the only option in this (default, non-Kuramoto)
        pathway, contrary to that guidance. Set False to use the legacy
        white-noise behavior.
    rng : numpy Generator | None
        Random number generator for reproducible phase noise.
    """

    def __init__(
        self,
        n_levels: int,
        theta_0: np.ndarray,
        omega: np.ndarray,
        lambda_rates: np.ndarray,
        kappa_down: float = 0.1,
        kappa_up: float = 0.0,
        theta_min: float = 0.1,
        theta_max: float = 20.0,
        phi_noise_std: float = 0.0,
        use_ou_phase_noise: bool = True,
        ou_tau_xi: float = 1.0,
        rng: RNGLike = None,
        pi_max: float = 10.0,
    ) -> None:
        if n_levels < 1:
            raise ValueError("n_levels must be >= 1")
        self.n_levels = n_levels
        self.theta_0 = np.asarray(theta_0, dtype=float)
        self.omega = np.asarray(omega, dtype=float)
        self.lambda_rates = np.clip(np.asarray(lambda_rates, dtype=float), 1e-6, 1.0 - 1e-6)
        self.kappa_down = float(kappa_down)
        self.kappa_up = float(kappa_up)
        self.theta_min = float(theta_min)
        self.theta_max = float(theta_max)
        if not pi_max > 0.0:
            raise ValueError(f"pi_max must be > 0, got {pi_max}")
        self.pi_max = float(pi_max)
        self.phi_noise_std = float(phi_noise_std)
        self.use_ou_phase_noise = bool(use_ou_phase_noise)
        self.rng = resolve_rng(rng)
        self._ou_noise = None
        if self.use_ou_phase_noise and self.phi_noise_std > 0.0:
            from oscillation.kuramoto import OrnsteinUhlenbeckNoise

            self._ou_noise = [
                OrnsteinUhlenbeckNoise(tau_xi=ou_tau_xi, sigma_xi=self.phi_noise_std)
                for _ in range(n_levels)
            ]
        # Per-level state
        self.S = np.zeros(n_levels)
        self.phi = np.zeros(n_levels)
        self.pi = np.ones(n_levels)
        # Initialise thresholds at their baselines
        self.theta = self.theta_0.copy()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def step(
        self,
        S_inst_levels: np.ndarray,
        pi_levels: np.ndarray,
        dt: float,
        noise_std: float = 0.0,
    ) -> None:
        """Advance the full resonance system by one timestep.
        Execution order (matches §20 pipeline steps 9-14):
          1. Update per-level precisions from the precision pipeline.
          2. Advance oscillatory phases with top-down Kuramoto entrainment.
          3. Recompute per-level thresholds via the spec PAC formula.
          4. Accumulate per-level ignition signals (leaky integrators + SDE noise).
        Parameters
        ----------
        S_inst_levels : array-like (L,)
            Per-level instantaneous salience:
            S_inst_l = Π_l · φ(ε_l) · Γ_l
        pi_levels : array-like (L,)
            Per-level precisions from the precision pipeline.
        dt : float
            Integration timestep.
        noise_std : float
            Standard deviation of per-level SDE diffusion noise (spec §12).
            Matches the precision-coupled noise of the single-scale ODE path:
            σ_S = 1 / sqrt(Π_e^eff + Π_i^eff).  Default 0.0 (deterministic).
        """
        S_inst = np.asarray(S_inst_levels, dtype=float)
        pi = np.asarray(pi_levels, dtype=float)
        self.pi = pi.copy()
        # 1 — Phase advancement with top-down Kuramoto entrainment
        self.phi = self._advance_phases(dt)
        # 2 — Threshold update using EQ-MS-4:
        #     θ_l = θ_{0,l} · clip(1 + κ_down · (Π_{l+1}/Π_max) · cos(φ_{l+1}), 0.5, 1.5)
        self.theta = self._compute_thresholds()
        # 3 — Per-level leaky accumulation with optional SDE noise (spec §12):
        #     S_l(t+1) = max(0, (1 − λ_l)·S_l + λ_l·S_inst_l + σ·√dt·ξ_l)
        # The non-negativity floor keeps S as an evidence counter — negative accumulated
        # evidence has no interpretation in the ignition model and would be clipped to 0
        # by the log stabiliser anyway, so we absorb it here to keep the ODE stable.
        self.S = (1.0 - self.lambda_rates) * self.S + self.lambda_rates * S_inst
        if noise_std > 0.0:
            self.S += self.rng.normal(0.0, noise_std * np.sqrt(dt), size=self.n_levels)
        self.S = np.maximum(0.0, self.S)

    def set_baseline(self, level: int, value: float) -> None:
        """Update the baseline threshold θ_{0,level} in place and immediately
        recompute all phase-modulated thresholds.
        This is the composition point required by MathSpec §4 / §8: the PAC
        formula θ_l = θ_{0,l}·(1 + κ_down·Π_{l+1}·cos(φ_{l+1})) modulates a
        baseline θ_{0,l} that is *itself* governed by the allostatic ODE
        θ(t+1) = θ(t) + η_θ·[C(t) − V(t)] + δ_reset·B(t) for the primary
        (level-0) ignition level. Callers must feed the freshly-computed
        allostatic/NE/5-HT-corrected threshold in here every step *before*
        reading `primary_threshold`, rather than leaving θ_{0,level} pinned
        to the static config baseline — otherwise the allostatic update is
        silently discarded whenever hierarchical mode is active.
        """
        if not (0 <= level < self.n_levels):
            raise ValueError(f"level {level} out of range [0, {self.n_levels})")
        self.theta_0[level] = float(value)
        self.theta = self._compute_thresholds()

    def apply_level_ignition(
        self,
        level: int,
        rho_S: float = 0.1,
        delta_refractory: float = 0.5,
        reset_phase: bool = True,
        theta_coupling: float = 0.0,
        phase_broadcast_decay: float | None = None,
    ) -> None:
        """Apply post-ignition refractory reset for the given level (§17).
        Parameters
        ----------
        level : int
            The level that fired (usually 0 for the primary ignition).
        rho_S : float
            Signal retention fraction after reset (ρ_S ∈ [0.1, 0.5]).
        delta_refractory : float
            Temporary threshold elevation δ_refractory.
        reset_phase : bool
            If True (default), also reset the igniting level's phase to
            theta_coupling (spec §9 canonical Δφ_reset = θ_coupling − φ(t),
            default θ_coupling=0 -> zero-phase reset), with an optional
            decayed broadcast to neighboring levels. This was previously
            missing from this (default, non-Kuramoto) resonance pathway —
            S and θ were reset on ignition but φ was left to drift
            independently, so the PAC mechanism in §8 never produced the
            rhythmic ignition windows the phase reset is "necessary" for.
        theta_coupling : float
            Target phase for the canonical reset (default 0).
        phase_broadcast_decay : float | None
            Decay coefficient d ∈ (0,1) for propagating a fraction of the
            phase correction to neighboring levels (Δφ_j = Δφ_reset·d^|j-level|).
            Defaults to self.kappa_up if not given (reusing the existing
            bottom-up cascade strength as a reasonable default coupling), or
            0.5 if kappa_up is 0.
        """
        if not (0 <= level < self.n_levels):
            raise ValueError(f"level {level} out of range [0, {self.n_levels})")
        self.S[level] *= rho_S
        self.theta[level] = min(self.theta[level] + delta_refractory, self.theta_max)
        if reset_phase:
            # Signed shortest-path correction, wrapped to (-pi, pi]. Using the
            # unsigned [0, 2*pi) wrap here would still land the igniting level
            # on theta_coupling (mod 2*pi), but would turn small negative
            # corrections into large positive ones before the broadcast below
            # scales them out to neighbors, propagating phase in the wrong
            # direction across the hierarchy.
            delta = (theta_coupling - self.phi[level] + np.pi) % (2.0 * np.pi) - np.pi
            self.phi[level] = (self.phi[level] + delta) % (2.0 * np.pi)
            decay = (
                phase_broadcast_decay
                if phase_broadcast_decay is not None
                else (self.kappa_up if self.kappa_up > 0.0 else 0.5)
            )
            for j in range(self.n_levels):
                if j == level:
                    continue
                effective = delta * (decay ** abs(j - level))
                self.phi[j] = (self.phi[j] + effective) % (2.0 * np.pi)
        # Bottom-up cascade: lower-level ignition suppresses the level above
        if level + 1 < self.n_levels and self.kappa_up > 0.0:
            self.theta[level + 1] *= 1.0 - self.kappa_up

    @property
    def primary_signal(self) -> float:
        """Accumulated ignition signal at the bottom level (primary sensory)."""
        return float(self.S[0])

    @property
    def primary_threshold(self) -> float:
        """Phase-modulated threshold at the bottom level."""
        return float(self.theta[0])

    @property
    def ignition_windows(self) -> np.ndarray:
        """Boolean array: True where S_l exceeds θ_l at the current timestep."""
        return self.theta < self.S

    @property
    def modulation_depth(self) -> np.ndarray:
        """Per-level modulation depth: Δθ_l / θ_{0,l} = κ · Π_{l+1} · cos(φ_{l+1}).
        Zero at the top level; positive when in an inhibitory half-cycle,
        negative when in an excitatory half-cycle.
        """
        depth = np.zeros(self.n_levels)
        for level in range(self.n_levels - 1):
            depth[level] = self.kappa_down * self.pi[level + 1] * np.cos(self.phi[level + 1])
        return depth

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _advance_phases(self, dt: float) -> np.ndarray:
        """Advance φ_l using natural frequency + top-down Kuramoto coupling.
        dφ_l/dt = ω_l + κ_down · sin(φ_{l+1} − φ_l)
        The coupling term pulls each level toward the phase of the level above
        it, creating nested synchrony windows (the Russian Doll property).
        """
        phi_new = np.zeros(self.n_levels)
        for level in range(self.n_levels):
            dphi = self.omega[level] * dt
            # Top-down entrainment from level level+1
            if level < self.n_levels - 1:
                dphi += self.kappa_down * np.sin(self.phi[level + 1] - self.phi[level]) * dt
            # Optional biological phase jitter — OU (colored) by default
            # when enabled, per MathSpec §9's preference over white noise.
            if self._ou_noise is not None:
                dphi += self._ou_noise[level].step(dt) * np.sqrt(dt)
            elif self.phi_noise_std > 0.0:
                dphi += self.phi_noise_std * self.rng.standard_normal() * np.sqrt(dt)
            phi_new[level] = (self.phi[level] + dphi) % (2.0 * np.pi)
        return phi_new

    def _compute_thresholds(self) -> np.ndarray:
        """Apply EQ-MS-4 (SSOT §10.7) to all levels.
        θ_l = θ_{0,l} · clip(1 + κ_down · (Π_{l+1}/Π_max) · cos(φ_{l+1}), 0.5, 1.5)
        The precision is normalized by Π_max and the factor bounded in [0.5, 1.5]
        (SSOT-P12; the former form without /Π_max spanned [0, 2] at Π_max = 10,
        κ_down = 0.1). The top level has no level above it, so it stays at its
        baseline θ_{0,L-1}.
        """
        thetas = np.empty(self.n_levels)
        for level in range(self.n_levels):
            if level < self.n_levels - 1:
                mod = 1.0 + self.kappa_down * (self.pi[level + 1] / self.pi_max) * np.cos(
                    self.phi[level + 1]
                )
                thetas[level] = self.theta_0[level] * min(1.5, max(0.5, mod))
            else:
                thetas[level] = self.theta_0[level]
        return np.clip(thetas, self.theta_min, self.theta_max)


def build_resonance_system(
    n_levels: int,
    taus: np.ndarray,
    theta_base: float,
    dt: float,
    kappa_down: float = 0.1,
    kappa_up: float = 0.0,
    theta_min: float = 0.1,
    theta_max: float = 20.0,
    phi_noise_std: float = 0.0,
    use_ou_phase_noise: bool = True,
    rng: RNGLike = None,
    pi_max: float = 10.0,
) -> NestedResonanceSystem:
    """Convenience factory: build a NestedResonanceSystem from pipeline timescales.
    Parameters
    ----------
    n_levels : int
    taus : array (L,)  — timescale per level (increasing with level index)
    theta_base : float — common baseline threshold for all levels
    dt : float         — integration step (used to derive λ_l = dt/τ_l)
    use_ou_phase_noise : bool — colored (OU) vs white phase noise when
        phi_noise_std > 0 (default True, per MathSpec §9)
    """
    theta_0 = np.full(n_levels, theta_base)
    omega = 2.0 * np.pi / taus
    # λ_l = dt / τ_l  — faster levels respond more quickly to salience
    lambda_rates = np.clip(dt / taus, 1e-4, 0.999)
    return NestedResonanceSystem(
        n_levels=n_levels,
        theta_0=theta_0,
        omega=omega,
        lambda_rates=lambda_rates,
        kappa_down=kappa_down,
        kappa_up=kappa_up,
        theta_min=theta_min,
        theta_max=theta_max,
        phi_noise_std=phi_noise_std,
        use_ou_phase_noise=use_ou_phase_noise,
        rng=rng,
        pi_max=pi_max,
    )
