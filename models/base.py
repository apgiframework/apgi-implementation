"""Common interface for APGI model variants.

Every variant — the qualified reference ``apgi_reduced``, the pipeline-backed
extensions, and the ``apgi_null`` benchmark — exposes the same small surface so
the session engine, batch runner and counterfactual machinery never need to
know which one they are driving.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import numpy as np

__all__ = ["APGIModel", "RUN_KEYS"]

#: Keys every model's :meth:`APGIModel.run` result must contain.
RUN_KEYS: tuple[str, ...] = ("S", "theta", "B", "p_ignite")


class APGIModel(ABC):
    """Abstract APGI model variant.

    Class attributes describe the variant's registry identity. ``authority`` is
    True only for the single qualified numerical reference (decision D14):
    results from any other variant never carry reference authority.
    """

    model_id: ClassVar[str]
    version: ClassVar[str] = "1.0.0"
    status: ClassVar[str] = "exploratory"
    authority: ClassVar[bool] = False
    parameter_names: ClassVar[tuple[str, ...]] = ()

    def __init__(self, params: Mapping[str, Any] | None = None, seed: int | None = None):
        self._params: dict[str, Any] = dict(params or {})
        self.seed = seed
        self.reset(seed)

    @property
    def params(self) -> dict[str, Any]:
        """A copy of the parameters this instance was built with."""
        return dict(self._params)

    @abstractmethod
    def reset(self, seed: int | None = None) -> None:
        """Return the model to its initial state and re-seed its generator."""

    @abstractmethod
    def step(self, x_e: float, x_i: float | None = None) -> dict[str, float]:
        """Advance one step and return at least ``S``, ``theta``, ``B``, ``p_ignite``."""

    def run(
        self, x_e: Sequence[float] | np.ndarray, x_i: Sequence[float] | np.ndarray | None = None
    ) -> dict[str, np.ndarray]:
        """Drive the model over an input sequence.

        Args:
            x_e: Exteroceptive input per step.
            x_i: Optional interoceptive input per step (same length).

        Returns:
            Mapping of :data:`RUN_KEYS` to float arrays of length ``len(x_e)``.
        """
        xs = np.asarray(x_e, dtype=float)
        xi = None if x_i is None else np.asarray(x_i, dtype=float)
        if xi is not None and xi.shape != xs.shape:
            raise ValueError(f"x_i shape {xi.shape} does not match x_e shape {xs.shape}")
        out: dict[str, list[float]] = {k: [] for k in RUN_KEYS}
        for t in range(xs.shape[0]):
            rec = self.step(float(xs[t]), None if xi is None else float(xi[t]))
            for k in RUN_KEYS:
                out[k].append(float(rec[k]))
        return {k: np.asarray(v, dtype=float) for k, v in out.items()}

    def get_state(self) -> dict[str, Any] | None:
        """Return a JSON-serialisable state snapshot, or ``None`` if unsupported.

        Models returning ``None`` are checkpointed by deterministic replay.
        """
        return None

    def set_state(self, state: Mapping[str, Any]) -> None:
        """Restore a snapshot produced by :meth:`get_state`."""
        raise NotImplementedError(f"{self.model_id} does not support state snapshots")
