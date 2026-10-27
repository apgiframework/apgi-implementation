"""Session engine (S1-IMP-11.1).

Drives one model through an input sequence step by step, keeps the history, and
can checkpoint and resume. A resumed session produces output identical to an
uninterrupted one — the property :mod:`tests.test_reproducibility` pins.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from adapters.canonical_inputs import CanonicalInputs, to_canonical_inputs
from adapters.canonical_outputs import CanonicalOutputs, to_canonical_outputs
from engine.provenance import array_digest
from model_registry.variants import create_model, get_variant
from models.base import RUN_KEYS, APGIModel

from .checkpoint import Checkpoint, load_checkpoint, save_checkpoint

__all__ = ["SessionEngine", "ReplayMismatchError", "InputMismatchError"]


class ReplayMismatchError(RuntimeError):
    """Deterministic replay did not reproduce the stored outputs."""


class InputMismatchError(ValueError):
    """The inputs supplied on resume differ from those the checkpoint consumed."""


class SessionEngine:
    """One model, one seed, one session.

    Args:
        model_id: Registered model id.
        params: Model parameters.
        seed: Seed (required for checkpoint/resume to be meaningful).
        checkpoint_path: If set together with ``checkpoint_every``, a checkpoint
            is written every that-many steps during :meth:`run`.
        checkpoint_every: Steps between automatic checkpoints.
    """

    def __init__(
        self,
        model_id: str,
        params: Mapping[str, Any] | None = None,
        seed: int | None = None,
        *,
        checkpoint_path: str | Path | None = None,
        checkpoint_every: int | None = None,
    ):
        if checkpoint_every is not None and checkpoint_every < 1:
            raise ValueError("checkpoint_every must be >= 1")
        if checkpoint_every is not None and checkpoint_path is None:
            raise ValueError("checkpoint_every requires checkpoint_path")
        self.model_id = model_id
        self.seed = seed
        self._params = dict(params or {})
        self.model: APGIModel = create_model(model_id, self._params, seed)
        self.checkpoint_path = None if checkpoint_path is None else Path(checkpoint_path)
        self.checkpoint_every = checkpoint_every
        self._x_e: list[float] = []
        self._x_i: list[float] | None = None
        self._hist: dict[str, list[float]] = {k: [] for k in RUN_KEYS}

    @property
    def n_done(self) -> int:
        """Steps completed."""
        return len(self._x_e)

    def advance(self, x_e: float, x_i: float | None = None) -> dict[str, float]:
        """Process one step and record it."""
        if x_i is not None:
            if self._x_i is None:
                if self._x_e:
                    raise ValueError("x_i supplied mid-session after steps without it")
                self._x_i = []
        elif self._x_i is not None:
            raise ValueError("x_i omitted mid-session after steps with it")
        rec = self.model.step(float(x_e), None if x_i is None else float(x_i))
        self._x_e.append(float(x_e))
        if self._x_i is not None and x_i is not None:
            self._x_i.append(float(x_i))
        for k in RUN_KEYS:
            self._hist[k].append(float(rec[k]))
        return rec

    def run(self, inputs: Any, *, max_steps: int | None = None) -> CanonicalOutputs:
        """Process the not-yet-consumed part of ``inputs``.

        Args:
            inputs: Anything :func:`adapters.to_canonical_inputs` accepts. The
                first :attr:`n_done` entries must equal what was already consumed.
            max_steps: Stop after this many *additional* steps (to model an
                interruption); ``None`` runs to the end.

        Returns:
            Outputs for all steps done so far (including earlier ones).

        Raises:
            InputMismatchError: If the already-consumed prefix differs.
        """
        data: CanonicalInputs = to_canonical_inputs(inputs)
        n = self.n_done
        if n > data.n_steps:
            raise InputMismatchError(
                f"session already consumed {n} steps; inputs have {data.n_steps}"
            )
        if not np.array_equal(data.x_e[:n], np.asarray(self._x_e)):
            raise InputMismatchError("inputs differ from those already consumed (x_e)")
        if self._x_i is not None and (
            data.x_i is None or not np.array_equal(data.x_i[:n], np.asarray(self._x_i))
        ):
            raise InputMismatchError("inputs differ from those already consumed (x_i)")
        stop = data.n_steps if max_steps is None else min(data.n_steps, n + max_steps)
        for t in range(n, stop):
            self.advance(float(data.x_e[t]), None if data.x_i is None else float(data.x_i[t]))
            if (
                self.checkpoint_every
                and self.checkpoint_path is not None
                and (t + 1) % self.checkpoint_every == 0
            ):
                self.save(self.checkpoint_path)
        return self.outputs()

    def outputs(self) -> CanonicalOutputs:
        """Canonical outputs for the steps done so far."""
        if not self._x_e:
            raise ValueError("no steps have been run")
        return to_canonical_outputs(
            {k: np.asarray(v) for k, v in self._hist.items()},
            model_id=self.model_id,
            model_version=get_variant(self.model_id).version,
        )

    def output_digest(self) -> str:
        """Digest of the outputs so far (for comparing runs)."""
        return array_digest({k: np.asarray(v) for k, v in self._hist.items()})

    def checkpoint(self) -> Checkpoint:
        """Snapshot the session."""
        return Checkpoint(
            model_id=self.model_id,
            model_version=get_variant(self.model_id).version,
            params=dict(self._params),
            seed=self.seed,
            n_done=self.n_done,
            x_e=list(self._x_e),
            x_i=None if self._x_i is None else list(self._x_i),
            history={k: list(v) for k, v in self._hist.items()},
            state=self.model.get_state(),
            extra={"output_digest": self.output_digest() if self._x_e else None},
        )

    def save(self, path: str | Path) -> Path:
        """Write a checkpoint file."""
        return save_checkpoint(self.checkpoint(), path)

    @classmethod
    def resume(cls, source: str | Path | Checkpoint, **kwargs: Any) -> SessionEngine:
        """Rebuild a session from a checkpoint (path or object).

        Raises:
            ReplayMismatchError: If a replay-restored model does not regenerate the
                stored outputs (the model is not deterministic under its seed).
        """
        ck = load_checkpoint(source) if isinstance(source, (str, os.PathLike)) else source
        engine = cls(ck.model_id, ck.params, ck.seed, **kwargs)
        if ck.state is not None:
            engine.model.set_state(ck.state)
            engine._x_e = list(ck.x_e)
            engine._x_i = None if ck.x_i is None else list(ck.x_i)
            engine._hist = {k: list(v) for k, v in ck.history.items()}
        else:
            for t in range(ck.n_done):
                engine.advance(ck.x_e[t], None if ck.x_i is None else ck.x_i[t])
            expected = ck.extra.get("output_digest")
            if expected is not None and engine.output_digest() != expected:
                raise ReplayMismatchError(
                    f"replay of {ck.model_id} did not reproduce the checkpointed outputs"
                )
        return engine
