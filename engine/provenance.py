"""Run-level provenance and deterministic replay (S1-IMP-08).

A :class:`RunRecord` pins everything needed to regenerate a run bit-for-bit —
model id and version, parameters, seed, and digests of the inputs and outputs —
plus the environment manifest from :mod:`core.manifest`. :func:`replay_run`
re-executes the run from the record and reports whether the output digest
matches, which is how "reproducible" is verified rather than asserted.

Digests round to a fixed number of decimals before hashing (as in the golden
master tests) so last-bit libm differences across platforms do not read as
irreproducibility, while any real change in the dynamics does.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.manifest import build_manifest
from model_registry.variants import create_model, get_variant

__all__ = [
    "RECORD_VERSION",
    "DIGEST_DECIMALS",
    "array_digest",
    "RunRecord",
    "ReplayReport",
    "record_run",
    "replay_run",
    "save_record",
    "load_record",
]

RECORD_VERSION = "1"
DIGEST_DECIMALS = 9


def array_digest(
    arrays: Mapping[str, Sequence[float] | np.ndarray], decimals: int = DIGEST_DECIMALS
) -> str:
    """SHA-256 over named arrays, order-independent in the mapping.

    Values are rounded to ``decimals`` and normalised so ``-0.0`` and ``0.0``
    hash alike. Non-finite values hash as the strings ``nan``/``inf``/``-inf``.
    """
    payload: dict[str, list[Any]] = {}
    for key in sorted(arrays):
        a = np.round(np.asarray(arrays[key], dtype=float), decimals) + 0.0
        payload[key] = [
            v if np.isfinite(v) else ("nan" if np.isnan(v) else ("inf" if v > 0 else "-inf"))
            for v in a.ravel().tolist()
        ]
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class RunRecord:
    """Everything needed to replay a run and verify its output."""

    model_id: str
    model_version: str
    params: dict[str, Any]
    seed: int | None
    n_steps: int
    input_digest: str
    output_digest: str
    manifest: dict[str, Any] = field(default_factory=dict)
    record_version: str = RECORD_VERSION

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReplayReport:
    """Outcome of :func:`replay_run`."""

    match: bool
    expected_digest: str
    actual_digest: str
    inputs_match: bool


def _input_arrays(
    x_e: Sequence[float] | np.ndarray, x_i: Sequence[float] | np.ndarray | None
) -> dict[str, np.ndarray]:
    arrays = {"x_e": np.asarray(x_e, dtype=float)}
    if x_i is not None:
        arrays["x_i"] = np.asarray(x_i, dtype=float)
    return arrays


def record_run(
    model_id: str,
    params: Mapping[str, Any] | None,
    seed: int | None,
    x_e: Sequence[float] | np.ndarray,
    x_i: Sequence[float] | np.ndarray | None = None,
    *,
    extra: Mapping[str, Any] | None = None,
) -> tuple[RunRecord, dict[str, np.ndarray]]:
    """Execute a run and capture its provenance.

    Args:
        model_id: Registered model id.
        params: Model parameters (must be JSON-serialisable).
        seed: Seed (``None`` records a non-reproducible run; replay then cannot match).
        x_e: Exteroceptive inputs.
        x_i: Interoceptive inputs.
        extra: Additional fields merged into the manifest.

    Returns:
        ``(record, outputs)`` where ``outputs`` is the model's :meth:`run` result.
    """
    variant = get_variant(model_id)
    p = dict(params or {})
    outputs = create_model(model_id, p, seed).run(x_e, x_i)
    config = {"model_id": model_id, "model_version": variant.version, "params": p, "seed": seed}
    record = RunRecord(
        model_id=model_id,
        model_version=variant.version,
        params=json.loads(json.dumps(p, default=repr)),
        seed=seed,
        n_steps=int(np.asarray(x_e).shape[0]),
        input_digest=array_digest(_input_arrays(x_e, x_i)),
        output_digest=array_digest(outputs),
        manifest=build_manifest(config, extra=dict(extra) if extra else None),
    )
    return record, outputs


def replay_run(
    record: RunRecord,
    x_e: Sequence[float] | np.ndarray,
    x_i: Sequence[float] | np.ndarray | None = None,
) -> ReplayReport:
    """Re-execute ``record`` on ``x_e``/``x_i`` and compare output digests.

    A mismatch in the *input* digest is reported separately so a failed replay
    can be attributed to the data rather than the model.
    """
    inputs_match = array_digest(_input_arrays(x_e, x_i)) == record.input_digest
    outputs = create_model(record.model_id, record.params, record.seed).run(x_e, x_i)
    actual = array_digest(outputs)
    return ReplayReport(
        match=inputs_match and actual == record.output_digest,
        expected_digest=record.output_digest,
        actual_digest=actual,
        inputs_match=inputs_match,
    )


def save_record(record: RunRecord, path: str | Path) -> Path:
    """Write ``record`` as JSON, creating parent directories."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(record.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return target


def load_record(path: str | Path) -> RunRecord:
    """Read a record written by :func:`save_record`.

    Raises:
        ValueError: If the file's ``record_version`` is not supported.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("record_version") != RECORD_VERSION:
        raise ValueError(f"unsupported record_version {data.get('record_version')!r}")
    return RunRecord(**data)
