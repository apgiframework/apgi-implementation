"""Allostatic threshold dynamics.
Implements continuous-time ODE for dynamic threshold adaptation matching
Paper §4.1.3 (Equation 3):

    dθₜ/dt = −λ_θ(θₜ − θ₀) + κ_meta·Cₜ − Δ_info·Iₜ + η_NE·NE(t)

where:
  λ_θ     — restoring rate toward homeostatic baseline θ₀
  κ_meta  — metabolic cost gain (raises threshold after costly ignitions)
  Δ_info  — information-value gain (lowers threshold after informative events)
  η_NE    — noradrenergic coupling (LC-NE phasic output)

The refractory boost δ·B(t) is applied as a *separate* post-ignition step
(spec §6 / pipeline step 9), not inside this ODE, consistent with the paper.
Backward-compatible single-η form (η·[C−V]) is preserved when c_theta
and delta_info are left at their None defaults.
"""

from __future__ import annotations

import numpy as np

from .deprecation import renamed_kwargs
from .rng import RNGLike, resolve_rng


@renamed_kwargs(kappa_meta="c_theta")
def allostatic_threshold_ode(
    theta: float,
    theta_0: float,
    gamma: float,
    B_prev: int,
    delta: float,
    C: float = 0.0,
    V: float = 0.0,
    eta: float = 0.0,
    dt: float = 1.0,
    noise_std: float = 0.01,
    c_theta: float | None = None,
    delta_info: float | None = None,
    ne: float = 0.0,
    eta_ne: float = 0.0,
    rng: RNGLike = None,
) -> float:
    """Compute one Euler-Maruyama step of the allostatic threshold ODE.

    Paper §4.1.3 (Eq. 3):
        dθ/dt = −λ_θ(θ − θ₀) + κ_meta·C − Δ_info·V + η_NE·NE(t)

    Backward-compatible form (when c_theta=None, delta_info=None):
        dθ/dt = −γ_θ(θ − θ_base) + δ_reset·B(t) + η·(C − V)

    The δ_reset·B(t) refractory term is retained for legacy code paths
    that set use_ode_refractory_drift=True; new code should leave B_prev=0
    and delta=0.0 here, applying the boost in pipeline step 9 instead.

    Args:
        theta: Current threshold value.
        theta_0: Homeostatic baseline θ₀.
        gamma: Restoring rate λ_θ = 1/τ_θ.
        B_prev: Previous ignition indicator (legacy refractory path only).
        delta: Refractory boost magnitude (legacy; use 0.0 for paper-spec path).
        C: Metabolic cost Cₜ.
        V: Information value Iₜ.
        eta: Shared allostatic gain (legacy; overridden by c_theta/delta_info).
        dt: Integration time step (seconds).
        noise_std: Euler-Maruyama noise amplitude (set to 0.0 for deterministic).
        c_theta: Threshold-update metabolic-cost coefficient c_θ (formerly κ_meta,
            deprecated keyword ``kappa_meta``). Metabolic cost gain κ_meta (paper Eq. 3). Defaults to eta.
        delta_info: Information-value gain Δ_info (paper Eq. 3). Defaults to eta.
        ne: Noradrenergic signal NE(t). Zero by default.
        eta_ne: Noradrenergic coupling η_NE (paper Eq. 3).
        rng: Generator/seed for the Euler-Maruyama noise draw; ``None`` uses
            the process-global APGI generator (see :mod:`core.rng`).
    Returns:
        Updated threshold θ(t+dt).
    """
    _c_theta = c_theta if c_theta is not None else eta
    _delta_i = delta_info if delta_info is not None else eta
    mean_reversion = -gamma * (theta - theta_0)
    refractory = delta * B_prev
    allostatic = _c_theta * C - _delta_i * V
    ne_term = eta_ne * ne
    drift = mean_reversion + refractory + allostatic + ne_term
    noise = float(resolve_rng(rng).normal(0.0, noise_std * np.sqrt(dt)))
    return float(theta + drift * dt + noise)


def update_threshold_euler(
    theta: float,
    theta_dot: float,
    dt: float = 1.0,
    theta_min: float = 0.0,
    theta_max: float = 1e6,
) -> float:
    """Euler integration for threshold update.
    θ(t+dt) = θ(t) + dt · dθ/dt
    Args:
        theta: Current threshold
        theta_dot: Threshold derivative (dθ/dt)
        dt: Time step
        theta_min: Minimum threshold (hard floor); default θ_floor = 0 (Math Spec §4.5)
        theta_max: Maximum threshold (hard ceiling)
    Returns:
        Updated threshold
    """
    theta_new = theta + dt * theta_dot
    return float(np.clip(theta_new, theta_min, theta_max))


class AllostaticThresholdController:
    """Continuous-time threshold controller with ODE dynamics per APGI spec."""

    def __init__(
        self,
        theta_0: float = 1.0,
        gamma: float = 0.01,
        delta: float = 0.5,
        dt: float = 1.0,
        rng: RNGLike = None,
    ):
        """Initialize allostatic controller.
        Args:
            theta_0: Baseline threshold (θ_base)
            gamma: Mean-reversion rate (γ_θ = 1/τ_θ)
            delta: Refractory boost magnitude (δ_reset)
            dt: Integration time step
            rng: Generator/seed for the ODE noise draws; ``None`` uses the
                process-global APGI generator.
        """
        self.theta = theta_0
        self.theta_0 = theta_0
        self.gamma = gamma
        self.delta = delta
        self.dt = dt
        self.B_prev = 0
        self.rng = resolve_rng(rng)

    def step(self, C: float, V: float, eta: float, B: int, noise_std: float = 0.01) -> float:
        """Single ODE step for threshold update with stochastic scaling.
        Args:
            C: Metabolic cost
            V: Information value
            eta: Allostatic learning rate
            B: Current ignition state
            noise_std: Threshold noise amplitude
        Returns:
            Updated threshold
        """
        # Compute threshold update per APGI spec (§7.2) with correct scaling
        self.theta = allostatic_threshold_ode(
            theta=self.theta,
            theta_0=self.theta_0,
            gamma=self.gamma,
            B_prev=self.B_prev,
            delta=self.delta,
            C=C,
            V=V,
            eta=eta,
            dt=self.dt,
            noise_std=noise_std,
            rng=self.rng,
        )
        # Store state for next step
        self.B_prev = B
        return self.theta

    def reset(self, theta: float | None = None) -> None:
        """Reset controller state."""
        self.theta = theta if theta is not None else self.theta_0
        self.B_prev = 0
