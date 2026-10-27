"""Counterfactual and simulated experimental conditions (S1-IMP-06).

A :class:`Condition` is a named, reproducible intervention on a model run: it
overrides parameters and/or transforms the stimulus. :func:`run_conditions`
runs every condition on the *same* spawned random streams (common random
numbers), so a contrast between conditions reflects the intervention rather
than sampling noise; :func:`contrast` summarises paired differences.

Built-in interventions model the manipulations the framework makes predictions
about: dopaminergic blockade (``β → 0``), a precision lesion, a clamped
threshold (no refractory boost), and a stimulus-gain change.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.rng import spawn_rng
from model_registry.variants import create_model

__all__ = [
    "Condition",
    "ConditionResult",
    "baseline",
    "dopamine_blockade",
    "precision_lesion",
    "stimulus_gain",
    "run_conditions",
    "contrast",
]


@dataclass(frozen=True)
class Condition:
    """A named intervention.

    Attributes:
        name: Unique label.
        param_overrides: Parameters replacing the base parameters.
        gain: Multiplier applied to the stimulus.
        offset: Constant added to the stimulus after scaling.
    """

    name: str
    param_overrides: Mapping[str, Any] = field(default_factory=dict)
    gain: float = 1.0
    offset: float = 0.0

    def apply(
        self, params: Mapping[str, Any], stimulus: np.ndarray
    ) -> tuple[dict[str, Any], np.ndarray]:
        """Return the intervened ``(params, stimulus)`` without mutating inputs."""
        return {**params, **self.param_overrides}, self.gain * stimulus + self.offset


def baseline() -> Condition:
    """The unmodified condition."""
    return Condition("baseline")


def dopamine_blockade() -> Condition:
    """Remove the additive dopaminergic bias (``β = 0``)."""
    return Condition("dopamine_blockade", {"beta": 0.0})


def precision_lesion(pi: float = 0.3) -> Condition:
    """Reduce precision to ``pi`` (default 0.3, the plausible-range floor)."""
    return Condition("precision_lesion", {"pi": pi})


def stimulus_gain(gain: float) -> Condition:
    """Scale the stimulus by ``gain``."""
    return Condition(f"stimulus_gain_{gain:g}", gain=gain)


@dataclass
class ConditionResult:
    """Per-condition output: arrays have shape ``(n_seeds, T)``."""

    name: str
    S: np.ndarray
    theta: np.ndarray
    B: np.ndarray
    p_ignite: np.ndarray

    @property
    def ignition_rate_per_seed(self) -> np.ndarray:
        return np.asarray(self.B.mean(axis=1))

    def summary(self) -> dict[str, float]:
        r = self.ignition_rate_per_seed
        return {
            "ignition_rate": float(r.mean()),
            "ignition_rate_sd": float(r.std(ddof=1)) if r.size > 1 else 0.0,
            "mean_S": float(self.S.mean()),
            "mean_theta": float(self.theta.mean()),
        }


def run_conditions(
    model_id: str,
    base_params: Mapping[str, Any],
    stimulus: Sequence[float] | np.ndarray,
    conditions: Sequence[Condition],
    *,
    n_seeds: int = 10,
    seed: int = 0,
) -> dict[str, ConditionResult]:
    """Run each condition on identical random streams.

    Args:
        model_id: Registered model id.
        base_params: Parameters before intervention.
        stimulus: Stimulus before intervention.
        conditions: Conditions to run; names must be unique.
        n_seeds: Number of independent sessions per condition.
        seed: Master seed from which per-session streams are spawned.

    Returns:
        Mapping from condition name to :class:`ConditionResult`.
    """
    names = [c.name for c in conditions]
    if len(set(names)) != len(names):
        raise ValueError(f"condition names must be unique, got {names}")
    if n_seeds < 1:
        raise ValueError("n_seeds must be >= 1")
    x = np.asarray(stimulus, dtype=float)
    # Integer seeds per session, fixed across conditions: common random numbers.
    session_seeds = [
        int(g.integers(0, 2**31 - 1, dtype=np.int64)) for g in spawn_rng(seed, n_seeds)
    ]
    out: dict[str, ConditionResult] = {}
    for cond in conditions:
        params, xs = cond.apply(base_params, x)
        runs = [create_model(model_id, params, s).run(xs) for s in session_seeds]
        out[cond.name] = ConditionResult(
            name=cond.name,
            S=np.stack([r["S"] for r in runs]),
            theta=np.stack([r["theta"] for r in runs]),
            B=np.stack([r["B"] for r in runs]),
            p_ignite=np.stack([r["p_ignite"] for r in runs]),
        )
    return out


def contrast(results: Mapping[str, ConditionResult], a: str, b: str) -> dict[str, float]:
    """Paired difference in ignition rate, ``a − b``, across shared sessions.

    Returns:
        ``mean_diff``, ``se`` (standard error of the paired difference) and ``n``.
    """
    diff = results[a].ignition_rate_per_seed - results[b].ignition_rate_per_seed
    n = int(diff.size)
    se = float(diff.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {"mean_diff": float(diff.mean()), "se": se, "n": float(n)}
