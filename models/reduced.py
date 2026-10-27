"""``apgi_reduced`` — the single qualified numerical reference (decision D14).

A deliberately small, fully specified model whose every equation can be checked
against the mathematical contract. Per trial ``t`` with stimulus ``x_t``:

    ε_t      = x_t − x̂_t                              prediction error
    drive_t  = tanh(γ_φ · Π · |ε_t|) + β              φ-transformed, precision-weighted
    S_t      = (1−λ)·ρ^{B_{t−1}}·S_{t−1} + λ·drive_t   leaky accumulation, post-ignition reset
    p_t      = σ((S_t − θ_t)/τ_σ)                     ignition probability
    B_t      = Bernoulli(p_t)  |  1[S_t > θ_t]        stochastic | hard-threshold ignition
    θ_{t+1}  = θ₀ + (θ_t − θ₀)·e^{−κ} + δ·B_t         allostatic decay + refractory boost
    x̂_{t+1}  = x̂_t + k·ε_t,   k = αΠ/(1+αΠ)           precision-weighted learning

The learning gain depends on α and Π only through the product ``αΠ`` — the
structural fact that makes the pair non-separable from behaviour alone and that
the K7a analysis tests explicitly.

Fixed constants follow the canonical pipeline defaults (``config.CONFIG``):
λ = 1−exp(−dt/τ_S) with dt=0.5, τ_S=5; ρ = RHO_RETAIN; κ = ``kappa``;
δ = ``delta``; τ_σ = ``ignite_tau``.

Three observation channels are available and are what the identifiability
analysis varies:

* ``behavior``  — report r_t ~ N(x̂_t, σ_r²) of the prediction before outcome;
* ``ignition``  — B_t, a Bernoulli draw with probability p_t;
* ``amplitude`` — y_t ~ N(S_t, σ_y²), an ERP-like amplitude.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any, ClassVar

import numpy as np

from core.rng import resolve_rng

from .base import APGIModel
from .parameters import REDUCED_PARAMETER_NAMES, reduced_parameter_spec

__all__ = [
    "LAM",
    "RHO",
    "KAPPA",
    "DELTA",
    "TAU_SIGMA",
    "GAMMA_PHI",
    "CHANNELS",
    "ReducedParams",
    "ReducedState",
    "ReducedTrajectory",
    "learning_gain",
    "reduced_step",
    "simulate_reduced",
    "ApgiReduced",
]

#: λ = 1 − exp(−dt/τ_S), dt=0.5, τ_S=5.0 (MathSpec §3).
LAM: float = 1.0 - math.exp(-0.5 / 5.0)
#: Fraction of S retained after ignition (RHO_RETAIN, MathSpec §6.1).
RHO: float = 0.1
#: Threshold relaxation rate κ.
KAPPA: float = 0.15
#: Refractory threshold boost δ.
DELTA: float = 0.5
#: Ignition sigmoid temperature τ_σ (``ignite_tau``).
TAU_SIGMA: float = 0.5
#: Steepness of the φ transform inside the drive.
GAMMA_PHI: float = 2.0
#: Observation channels in canonical order.
CHANNELS: tuple[str, ...] = ("behavior", "ignition", "amplitude")

_EXP_NEG_KAPPA = math.exp(-KAPPA)


@dataclass(frozen=True)
class ReducedParams:
    """Parameters of ``apgi_reduced``.

    ``alpha``, ``pi``, ``theta_base`` and ``beta`` are the estimable parameters;
    ``sigma_r`` and ``sigma_y`` are the observation noise levels, treated as
    known nuisance constants.
    """

    alpha: float = 0.3
    pi: float = 1.5
    theta_base: float = 0.55
    beta: float = 0.05
    sigma_r: float = 0.15
    sigma_y: float = 0.10

    def __post_init__(self) -> None:
        for name in REDUCED_PARAMETER_NAMES:
            value = getattr(self, name)
            if not math.isfinite(value) or not reduced_parameter_spec(name).contains(value):
                spec = reduced_parameter_spec(name)
                raise ValueError(
                    f"{name}={value!r} outside support [{spec.lower}, {spec.upper}] ({spec.symbol})"
                )
        if self.sigma_r <= 0 or self.sigma_y <= 0:
            raise ValueError("sigma_r and sigma_y must be > 0")

    @classmethod
    def from_mapping(cls, params: Mapping[str, Any] | None) -> ReducedParams:
        """Build from a (possibly partial) mapping; unknown keys raise."""
        known = set(cls.__dataclass_fields__)
        extra = set(params or {}) - known
        if extra:
            raise KeyError(f"unknown apgi_reduced parameters: {sorted(extra)}")
        return cls(**{k: float(v) for k, v in (params or {}).items()})

    def estimable(self) -> np.ndarray:
        """Return the estimable parameters as a vector in canonical order."""
        return np.array([getattr(self, n) for n in REDUCED_PARAMETER_NAMES], dtype=float)

    def with_estimable(self, vector: Sequence[float] | np.ndarray) -> ReducedParams:
        """Return a copy with the estimable parameters replaced by ``vector``."""
        if len(vector) != len(REDUCED_PARAMETER_NAMES):
            raise ValueError(f"expected {len(REDUCED_PARAMETER_NAMES)} values, got {len(vector)}")
        return replace(
            self, **{n: float(v) for n, v in zip(REDUCED_PARAMETER_NAMES, vector, strict=True)}
        )

    def as_dict(self) -> dict[str, float]:
        """Plain-dict form (JSON-serialisable)."""
        return asdict(self)


@dataclass
class ReducedState:
    """Latent state carried between trials."""

    xhat: float = 0.0
    S: float = 0.0
    theta: float = 0.55
    B_prev: int = 0

    def as_dict(self) -> dict[str, float | int]:
        return {"xhat": self.xhat, "S": self.S, "theta": self.theta, "B_prev": self.B_prev}


@dataclass
class ReducedTrajectory:
    """Per-trial arrays from :func:`simulate_reduced` (all length ``T``)."""

    xhat: np.ndarray
    eps: np.ndarray
    S: np.ndarray
    theta: np.ndarray
    p: np.ndarray
    B: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {k: getattr(self, k) for k in ("xhat", "eps", "S", "theta", "p", "B")}


def learning_gain(alpha: float, pi: float) -> float:
    """Prediction gain ``k = αΠ/(1+αΠ)`` — a function of the product only."""
    ap = alpha * pi
    return ap / (1.0 + ap)


def _sigmoid(z: float) -> float:
    z = max(-500.0, min(500.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def reduced_step(
    params: ReducedParams,
    state: ReducedState,
    x: float,
    *,
    B_forced: int | None = None,
    u: float | None = None,
    hard: bool = False,
) -> tuple[ReducedState, dict[str, float]]:
    """Advance ``apgi_reduced`` by one trial.

    Args:
        params: Model parameters.
        state: State entering the trial. Not mutated.
        x: Stimulus ``x_t``.
        B_forced: If given, use this ignition outcome (teacher forcing, as in
            likelihood evaluation) instead of sampling.
        u: Uniform(0,1) draw for stochastic ignition; ignored when ``hard`` or
            ``B_forced`` is set.
        hard: Use the hard threshold ``B = 1[S > θ]``.

    Returns:
        ``(new_state, record)`` where ``record`` holds the trial's ``eps``,
        ``S``, ``theta`` (the threshold in force), ``p_ignite``, ``B`` and the
        pre-update prediction ``xhat``.
    """
    eps = x - state.xhat
    drive = math.tanh(GAMMA_PHI * params.pi * abs(eps)) + params.beta
    carry = RHO if state.B_prev else 1.0
    S = (1.0 - LAM) * carry * state.S + LAM * drive
    p = _sigmoid((S - state.theta) / TAU_SIGMA)
    if B_forced is not None:
        B = int(B_forced)
    elif hard:
        B = int(state.theta < S)
    else:
        if u is None:
            raise ValueError("stochastic ignition requires a uniform draw u")
        B = int(u < p)
    theta_next = params.theta_base + (state.theta - params.theta_base) * _EXP_NEG_KAPPA + DELTA * B
    xhat_next = state.xhat + learning_gain(params.alpha, params.pi) * eps
    record = {
        "xhat": state.xhat,
        "eps": eps,
        "S": S,
        "theta": state.theta,
        "p_ignite": p,
        "B": float(B),
    }
    return ReducedState(xhat_next, S, theta_next, B), record


def simulate_reduced(
    params: ReducedParams,
    stimulus: Sequence[float] | np.ndarray,
    *,
    rng: np.random.Generator | int | None = None,
    B_forced: Sequence[int] | np.ndarray | None = None,
    hard: bool = False,
) -> ReducedTrajectory:
    """Simulate ``apgi_reduced`` over a stimulus sequence.

    Args:
        params: Model parameters.
        stimulus: Stimulus per trial.
        rng: Generator or seed for stochastic ignition (unused when ``hard`` or
            ``B_forced`` is set).
        B_forced: Observed ignition sequence for teacher-forced evaluation.
        hard: Use the hard threshold instead of Bernoulli sampling.

    Returns:
        A :class:`ReducedTrajectory`.
    """
    x = np.asarray(stimulus, dtype=float)
    T = x.shape[0]
    forced = None if B_forced is None else np.asarray(B_forced, dtype=int)
    if forced is not None and forced.shape[0] != T:
        raise ValueError("B_forced must have the same length as stimulus")
    draws = None
    if forced is None and not hard:
        draws = resolve_rng(rng).random(T)
    state = ReducedState(theta=params.theta_base)
    cols = {k: np.empty(T) for k in ("xhat", "eps", "S", "theta", "p_ignite", "B")}
    for t in range(T):
        state, rec = reduced_step(
            params,
            state,
            float(x[t]),
            B_forced=None if forced is None else int(forced[t]),
            u=None if draws is None else float(draws[t]),
            hard=hard,
        )
        for k, v in rec.items():
            cols[k][t] = v
    return ReducedTrajectory(
        xhat=cols["xhat"],
        eps=cols["eps"],
        S=cols["S"],
        theta=cols["theta"],
        p=cols["p_ignite"],
        B=cols["B"].astype(int),
    )


class ApgiReduced(APGIModel):
    """Step-wise wrapper over :func:`reduced_step` (the qualified reference)."""

    model_id: ClassVar[str] = "apgi_reduced"
    version: ClassVar[str] = "1.0.0"
    status: ClassVar[str] = "registered"
    authority: ClassVar[bool] = True
    parameter_names: ClassVar[tuple[str, ...]] = REDUCED_PARAMETER_NAMES

    def __init__(self, params: Mapping[str, Any] | None = None, seed: int | None = None):
        extra = dict(params or {})
        self.hard = bool(extra.pop("hard", False))
        self.reduced_params = ReducedParams.from_mapping(extra)
        super().__init__(self.reduced_params.as_dict() | {"hard": self.hard}, seed)

    def reset(self, seed: int | None = None) -> None:
        self.seed = seed
        self._rng = resolve_rng(seed)
        self._state = ReducedState(theta=self.reduced_params.theta_base)
        self._n = 0

    def step(self, x_e: float, x_i: float | None = None) -> dict[str, float]:
        u = None if self.hard else float(self._rng.random())
        self._state, rec = reduced_step(self.reduced_params, self._state, x_e, u=u, hard=self.hard)
        self._n += 1
        rec["x_hat_e"] = self._state.xhat
        return rec

    def get_state(self) -> dict[str, Any]:
        return {
            "state": self._state.as_dict(),
            "n": self._n,
            "rng": _jsonable_bitgen_state(self._rng.bit_generator.state),
        }

    def set_state(self, state: Mapping[str, Any]) -> None:
        s = state["state"]
        self._state = ReducedState(
            xhat=float(s["xhat"]),
            S=float(s["S"]),
            theta=float(s["theta"]),
            B_prev=int(s["B_prev"]),
        )
        self._n = int(state["n"])
        self._rng.bit_generator.state = state["rng"]


def _jsonable_bitgen_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """PCG64 state is already plain ints/strs; return an independent copy."""
    return {
        "bit_generator": state["bit_generator"],
        "state": {"state": int(state["state"]["state"]), "inc": int(state["state"]["inc"])},
        "has_uint32": int(state["has_uint32"]),
        "uinteger": int(state["uinteger"]),
    }
