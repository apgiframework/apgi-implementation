"""Somatic-marker gain and threshold adaptation (S1-IMP-03).

Two coupled mechanisms, both composed from ``core`` primitives:

* **Somatic-marker gain** — interoceptive precision is scaled exponentially by
  the marker, ``Π_i^eff = clamp(Π_i · exp(β_SM · M))`` (Notation Appendix:
  β_SM canonical range [0.1, 1.2]). The clamp is applied *after* the exponential
  so it bounds the blow-up.
* **Threshold adaptation** — the allostatic update: relaxation toward the
  homeostatic baseline, ``θ ← θ₀ + (θ−θ₀)e^{−κ}``, the optional multiplicative
  norepinephrine modulation ``θ ← θ(1+γ_NE g_NE)`` subject to the mandatory
  ``γ_NE·g_NE ≤ 0.5`` clamp (MathSpec §2), the serotonergic offset ``+β_5HT``,
  and the post-ignition refractory boost ``+δB``.

The marker ``M`` follows ``dM/dt = −(M − M*)/τ_M`` toward an arousal-derived
target and is bounded to [−2, 2].
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from core.precision import compute_interoceptive_precision_exponential
from core.somatic import compute_somatic_gain, update_somatic_marker_euler
from core.threshold import (
    apply_ne_threshold_modulation,
    apply_refractory_boost,
    apply_serotonin_threshold_offset,
    threshold_decay,
)

__all__ = [
    "NE_CLAMP_MAX",
    "BETA_SM_RANGE",
    "somatic_gain",
    "interoceptive_precision",
    "clamp_ne_gain",
    "SomaticThresholdAdapter",
]

#: Spec-mandated ceiling on γ_NE · g_NE (MathSpec §2).
NE_CLAMP_MAX: float = 0.5
#: Canonical range of the exponential somatic gain β_SM.
BETA_SM_RANGE: tuple[float, float] = (0.1, 1.2)


def somatic_gain(M: float, beta_sm: float = 0.3) -> float:
    """Multiplicative gain ``exp(β_SM·M)``."""
    return compute_somatic_gain(M, beta_sm)


def interoceptive_precision(
    pi_baseline: float, M: float, beta_sm: float = 0.3, pi_min: float = 0.1, pi_max: float = 10.0
) -> float:
    """``clamp(Π_i · exp(β_SM·M), Π_min, Π_max)``."""
    return compute_interoceptive_precision_exponential(pi_baseline, beta_sm, M, pi_min, pi_max)


def clamp_ne_gain(g_ne: float, gamma_ne: float) -> float:
    """Rescale ``g_NE`` so ``γ_NE·g_NE ≤ 0.5`` (MathSpec §2): ``g_NE ← 0.5/γ_NE``."""
    if gamma_ne > 0 and gamma_ne * g_ne > NE_CLAMP_MAX:
        return NE_CLAMP_MAX / gamma_ne
    return g_ne


@dataclass(frozen=True)
class AdapterOutput:
    """Result of one adapter step."""

    M: float
    gain: float
    theta: float
    theta_eff: float


class SomaticThresholdAdapter:
    """Joint somatic-marker and threshold state.

    Args:
        theta0: Homeostatic baseline θ₀ (also the initial threshold).
        kappa: Relaxation rate κ ≥ 0.
        delta: Refractory boost δ ≥ 0.
        beta_sm: Somatic gain β_SM.
        tau_M: Marker time constant (same units as ``dt``).
        gamma_ne: NE threshold-modulation strength γ_NE (0 disables the pathway).
        g_ne: NE gain; rescaled by :func:`clamp_ne_gain` when the product exceeds 0.5.
        beta_5ht: Serotonergic additive offset applied to the *effective* threshold.
        M0: Initial marker.
    """

    def __init__(
        self,
        theta0: float = 0.55,
        kappa: float = 0.15,
        delta: float = 0.5,
        beta_sm: float = 0.3,
        tau_M: float = 500.0,
        gamma_ne: float = 0.0,
        g_ne: float = 1.0,
        beta_5ht: float = 0.0,
        M0: float = 0.0,
    ):
        if kappa < 0 or delta < 0:
            raise ValueError("kappa and delta must be >= 0")
        if tau_M <= 0:
            raise ValueError("tau_M must be > 0")
        if not -2.0 <= M0 <= 2.0:
            raise ValueError("M0 must lie in [-2, 2]")
        self.theta0 = theta0
        self.kappa = kappa
        self.delta = delta
        self.beta_sm = beta_sm
        self.tau_M = tau_M
        self.gamma_ne = gamma_ne
        self.g_ne = clamp_ne_gain(g_ne, gamma_ne)
        self.beta_5ht = beta_5ht
        self.M = M0
        self.theta = theta0

    def step(self, arousal_target: float, B: int, dt: float = 1.0) -> AdapterOutput:
        """Advance marker and threshold by one step.

        Args:
            arousal_target: Arousal in [0, 1] that the marker relaxes toward.
            B: Ignition indicator for this step (0/1).
            dt: Step length.
        """
        if B not in (0, 1):
            raise ValueError(f"B must be 0 or 1, got {B!r}")
        self.M = update_somatic_marker_euler(self.M, arousal_target, self.tau_M, dt)
        theta = threshold_decay(self.theta, self.theta0, self.kappa)
        if self.gamma_ne > 0:
            theta = apply_ne_threshold_modulation(theta, self.g_ne, self.gamma_ne)
        theta = apply_refractory_boost(theta, B, self.delta)
        if not math.isfinite(theta):
            raise FloatingPointError("threshold became non-finite")
        self.theta = theta
        return AdapterOutput(
            M=self.M,
            gain=somatic_gain(self.M, self.beta_sm),
            theta=self.theta,
            theta_eff=apply_serotonin_threshold_offset(self.theta, self.beta_5ht),
        )
