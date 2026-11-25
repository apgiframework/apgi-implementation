from __future__ import annotations

import numpy as np

from .deprecation import renamed_kwargs
from .rng import RNGLike, resolve_rng


def signal_drift(
    S: float,
    phi_e: float,
    phi_i: float,
    pi_e: float,
    pi_i: float,
    tau_s: float,
) -> float:
    """Deterministic ODE drift using signed valence-specific phi transforms (§6/§12).
    S_inst = Π_e · φ(ε_e) + Π_i · φ(ε_i).
    The sign is preserved: positive errors drive ignition, negative errors suppress it.
    """
    if tau_s <= 0:
        raise ValueError("tau_s must be > 0")
    return float(-S / tau_s + pi_e * phi_e + pi_i * phi_i)


def update_signal_ode(
    S: float,
    phi_e: float,
    phi_i: float,
    pi_e: float,
    pi_i: float,
    tau_s: float,
    dt: float = 1.0,
    noise_std: float = 0.01,
    rng: RNGLike = None,
) -> float:
    """dS/dt = -S/τ_S + Π^e·φ(ε^e) + Π^i·φ(ε^i) + η_S(t) (§12).

    Implements Euler-Maruyama integration with signed valence-specific
    transforms. ``rng=None`` uses the process-global APGI generator.
    """
    if tau_s <= 0:
        raise ValueError("tau_s must be > 0")
    drift = -S / tau_s + pi_e * phi_e + pi_i * phi_i
    noise = float(resolve_rng(rng).normal(0.0, noise_std * np.sqrt(dt)))
    return float(S + drift * dt + noise)


def compute_precision_coupled_noise_std(pi_e_eff: float, pi_i_eff: float) -> float:
    """σ_S = 1 / sqrt(Π_e^eff + Π_i^eff) (§7.3)."""
    total_pi = pi_e_eff + pi_i_eff
    if total_pi <= 0:
        return 1.0  # Default to high noise if precision is zero
    return float(1.0 / np.sqrt(total_pi))


@renamed_kwargs(kappa="eta_pred")
def update_prediction(
    x_hat: float,
    epsilon: float,
    pi: float,
    eta_pred: float,
    pi_max: float = 100.0,
) -> float:
    """Generative Model Dynamics (Prediction Update) per §1.4.
    x̂(t+1) = x̂(t) + η_pred · Π(t) · ε(t)
    Stability condition: η_pred < 2 / Π_max ensures gradient descent convergence.
    Args:
        x_hat: Current prediction
        epsilon: Raw prediction error (x - x_hat)
        pi: Precision (Bayesian confidence)
        eta_pred: Prediction learning rate η_pred (Notation Appendix; formerly ``kappa``)
        pi_max: Maximum precision clamp (for stability check)
    Returns:
        Updated prediction x_hat(t+1)
    """
    # Optional stability check: η_pred < 2 / Π_max
    # We don't enforce it strictly here but document it.
    return float(x_hat + eta_pred * pi * epsilon)


def update_threshold_ode(
    theta: float,
    theta_base: float,
    C: float,
    V: float,
    tau_theta: float,
    eta: float,
    delta: float = 0.0,
    B: int = 0,
    dt: float = 1.0,
    noise_std: float = 0.01,
    rng: RNGLike = None,
) -> float:
    """Continuous threshold dynamics ODE per APGI spec (§7.2/7.4).
    dθ/dt = -(θ - θ_base)/τ_θ + η·(C - V) + δ_reset·B(t) + η_θ(t)
    Implements Euler-Maruyama integration step:
    θ(t+dt) = θ(t) + dθ/dt * dt + noise_std * sqrt(dt) * N(0,1)
    Components:
    - Decay term: -(θ - θ_base)/τ_θ (mean-reversion to baseline)
    - Allostatic term: η·(C - V) (cost-value mismatch)
    - Refractory term: δ_reset·B(t) (post-ignition jump)
    - Noise term: η_θ(t) (stochastic fluctuations)
    Args:
        theta: Current threshold
        theta_base: Baseline threshold value
        C: Metabolic cost
        V: Information value
        tau_theta: Threshold decay time constant (τ_θ)
        eta: Allostatic learning rate
        delta: Refractory boost magnitude (δ_reset)
        B: Ignition state (0 or 1)
        dt: Integration time step
        noise_std: Standard deviation of threshold noise
        rng: Generator/seed for the noise draw; ``None`` uses the
            process-global APGI generator.
    Returns:
        Updated threshold θ(t+dt)
    """
    if tau_theta <= 0:
        raise ValueError("tau_theta must be > 0")
    # Mean-reversion toward baseline: -(θ - θ_base)/τ_θ
    decay_term = -(theta - theta_base) / tau_theta
    # Allostatic update (cost - value)
    allostatic_term = eta * (C - V)
    # Refractory boost (impulse-like term in ODE)
    # Note: B(t) is treated as a density-like term per spec §7.4
    refractory_term = delta * int(B)
    drift = decay_term + allostatic_term + refractory_term
    noise = float(resolve_rng(rng).normal(0.0, noise_std * np.sqrt(dt)))
    return float(theta + drift * dt + noise)
