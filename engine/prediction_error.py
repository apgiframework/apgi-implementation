"""Precision-weighted prediction-error calculations (S1-IMP-02).

Implements pipeline steps 1–5 for one channel in isolation, built from the same
``core`` primitives the pipeline uses so the two cannot drift:

1. raw error ``ε = x − x̂``;
2. z-score against the *previous* running mean/variance (standard filtering);
3. EMA mean and centred variance update (MathSpec §1 Method A, variance
   centred on the pre-update mean);
4. precision ``Π = 1/(σ²+ε)`` clamped to ``[Π_min, Π_max]`` (§7.2);
5. optional signed φ-transform and precision weighting.

``tests/test_engine.py`` checks parity against :class:`pipeline.APGIPipeline`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.phi_transform import phi_transform
from core.precision import compute_precision, update_mean_ema
from core.preprocessing import compute_prediction_error

__all__ = [
    "PredictionErrorResult",
    "PrecisionWeightedErrorChannel",
    "precision_weighted_errors",
    "precision_weighted_error",
]


def precision_weighted_error(eps: float, pi: float) -> float:
    """Return ``Π·ε``, the quantity that drives the signal.

    Raises:
        ValueError: If ``pi`` is negative.
    """
    if pi < 0:
        raise ValueError(f"precision must be >= 0, got {pi}")
    return float(pi * eps)


@dataclass(frozen=True)
class PredictionErrorResult:
    """Output of one :meth:`PrecisionWeightedErrorChannel.update`."""

    eps: float
    z_norm: float
    phi: float
    mu: float
    sigma2: float
    pi: float
    weighted: float


class PrecisionWeightedErrorChannel:
    """One sensory channel's running error statistics and precision.

    Args:
        alpha: EMA rate ``α ∈ (0, 1]`` (``alpha_e``/``alpha_i`` in the pipeline).
        pi_min: Precision floor.
        pi_max: Precision ceiling.
        eps: Numerical floor in ``1/(σ²+ε)`` and in the z-score denominator.
        mu0: Initial running mean.
        sigma2_0: Initial running variance.
        alpha_plus, alpha_minus, gamma_plus, gamma_minus: φ-transform parameters.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        pi_min: float = 0.1,
        pi_max: float = 10.0,
        eps: float = 1e-8,
        mu0: float = 0.0,
        sigma2_0: float = 1.0,
        alpha_plus: float = 1.0,
        alpha_minus: float = 1.0,
        gamma_plus: float = 2.0,
        gamma_minus: float = 2.0,
        reference_window: int | None = None,
    ):
        """``reference_window`` (samples) selects the APGI v1.2 minimal-core order (SSOT §8):
        z is standardized against a sliding reference window that excludes the current error
        (EQ-STD-1), and the returned Π is computed from the variance BEFORE the current error
        enters it (step 4). ``None`` keeps the v1.1 decomposition (one shared estimator)."""
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if not 0.0 < pi_min <= pi_max:
            raise ValueError(f"need 0 < pi_min <= pi_max, got {pi_min}, {pi_max}")
        self.alpha = alpha
        self.pi_min = pi_min
        self.pi_max = pi_max
        self.eps = eps
        self.mu = mu0
        self.sigma2 = sigma2_0
        self._phi = (alpha_plus, alpha_minus, gamma_plus, gamma_minus)
        self.pi = compute_precision(sigma2_0, eps, pi_min, pi_max)
        self.reference_window = reference_window
        self._ref: list[float] = []

    def update(self, x: float, x_hat: float) -> PredictionErrorResult:
        """Process one observation against its prediction."""
        z = compute_prediction_error(x, x_hat)
        if self.reference_window is None:
            z_norm = (z - self.mu) / (self.sigma2**0.5 + self.eps)
        else:
            n = len(self._ref)
            if n < 2:
                m_ref, s_ref = 0.0, 1.0
            else:
                m_ref = sum(self._ref) / n
                s_ref = (sum((w - m_ref) ** 2 for w in self._ref) / n) ** 0.5
            z_norm = (z - m_ref) / (s_ref + self.eps)
            self.pi = compute_precision(self.sigma2, self.eps, self.pi_min, self.pi_max)
        phi = phi_transform(z_norm, *self._phi)
        mu_prev = self.mu
        self.mu = update_mean_ema(self.mu, z, self.alpha)
        self.sigma2 = (1.0 - self.alpha) * self.sigma2 + self.alpha * (z - mu_prev) ** 2
        if self.reference_window is None:
            self.pi = compute_precision(self.sigma2, self.eps, self.pi_min, self.pi_max)
        else:
            self._ref.append(z)
            del self._ref[: -max(2, self.reference_window)]
        return PredictionErrorResult(
            eps=z,
            z_norm=z_norm,
            phi=phi,
            mu=self.mu,
            sigma2=self.sigma2,
            pi=self.pi,
            weighted=precision_weighted_error(phi, self.pi),
        )


def precision_weighted_errors(
    x: Sequence[float] | np.ndarray,
    x_hat: Sequence[float] | np.ndarray,
    **channel_kwargs: Any,
) -> dict[str, np.ndarray]:
    """Vectorised driver: run a fresh channel over paired sequences.

    Returns:
        Arrays ``eps``, ``z_norm``, ``phi``, ``mu``, ``sigma2``, ``pi``, ``weighted``.
    """
    xs = np.asarray(x, dtype=float)
    hs = np.asarray(x_hat, dtype=float)
    if xs.shape != hs.shape or xs.ndim != 1:
        raise ValueError("x and x_hat must be 1-D arrays of equal length")
    ch = PrecisionWeightedErrorChannel(**channel_kwargs)
    keys = ("eps", "z_norm", "phi", "mu", "sigma2", "pi", "weighted")
    out = {k: np.empty(xs.shape[0]) for k in keys}
    for t in range(xs.shape[0]):
        r = ch.update(float(xs[t]), float(hs[t]))
        for k in keys:
            out[k][t] = getattr(r, k)
    return out
