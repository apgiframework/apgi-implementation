"""Session checkpoints (S1-IMP-11.2).

A checkpoint captures a session mid-run — model identity and parameters, the
seed, how many steps are done, the inputs consumed and outputs produced so far,
and (for models that support it) a state snapshot. Resuming reproduces the
uninterrupted run exactly.

* Models with :meth:`~models.base.APGIModel.get_state` are restored from the
  snapshot.
* Others are restored by **deterministic replay** of the consumed inputs, after
  which the regenerated outputs are compared with the stored digest; a mismatch
  means the run was not deterministic and raises rather than resuming silently
  from a different trajectory.

Files are written atomically (temp file + rename) and carry a SHA-256 of their
payload, verified on load.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "CHECKPOINT_VERSION",
    "Checkpoint",
    "CheckpointCorruptError",
    "payload_digest",
    "save_checkpoint",
    "load_checkpoint",
]

CHECKPOINT_VERSION = "1"


class CheckpointCorruptError(ValueError):
    """The checkpoint file is unreadable, tampered with, or of an unsupported version."""


@dataclass
class Checkpoint:
    """Serialisable session state."""

    model_id: str
    model_version: str
    params: dict[str, Any]
    seed: int | None
    n_done: int
    x_e: list[float]
    x_i: list[float] | None
    history: dict[str, list[float]]
    state: dict[str, Any] | None = None
    version: str = CHECKPOINT_VERSION
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def payload_digest(body: Mapping[str, Any]) -> str:
    """SHA-256 over the canonical JSON of ``body``."""
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def save_checkpoint(checkpoint: Checkpoint, path: str | Path) -> Path:
    """Atomically write ``checkpoint`` to ``path``.

    Raises:
        ValueError: If the state contains NaN/inf (not representable in strict JSON).
    """
    body = checkpoint.as_dict()
    doc = {"sha256": payload_digest(body), "checkpoint": body}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=target.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, allow_nan=False)
        os.replace(tmp, target)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return target


def load_checkpoint(path: str | Path) -> Checkpoint:
    """Read and verify a checkpoint.

    Raises:
        CheckpointCorruptError: On unreadable JSON, a digest mismatch or an
            unsupported ``version``.
    """
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        body = doc["checkpoint"]
        expected = doc["sha256"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CheckpointCorruptError(f"{path}: unreadable checkpoint ({exc})") from exc
    if payload_digest(body) != expected:
        raise CheckpointCorruptError(f"{path}: digest mismatch (file modified or truncated)")
    if body.get("version") != CHECKPOINT_VERSION:
        raise CheckpointCorruptError(
            f"{path}: unsupported checkpoint version {body.get('version')!r}"
        )
    return Checkpoint(**body)
