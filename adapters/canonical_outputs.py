"""Canonical outputs (S1-IMP-10.2).

Models return dictionaries whose keys have varied across versions. This module
maps them onto one schema with units and invariants — ``B`` binary, ``p_ignite``
in [0, 1], all values finite, all series the same length — so downstream
analysis, manifests and the gate machinery depend on one shape.

Exchanged payloads (APGI-INT-001, AUD-INT-001) must carry their schema
version, protocol version and producing-software version.
:meth:`CanonicalOutputs.exchange_payload` refuses to serialise a payload that
lacks any of them (or the model identity), so an unattributable result cannot
leave the process. ``as_dict`` remains available for in-process use.

Energy fields (APGI-REP-005) declare their unit explicitly: ``C_landauer`` is
in joules (J) and ``bits_erased`` in bits. Pipeline ``C`` values in arbitrary
units are not mapped onto an energy field.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "OUTPUT_SCHEMA",
    "OUTPUT_ALIASES",
    "OUTPUT_SCHEMA_VERSION",
    "PRODUCER_NAME",
    "IncompletePayloadError",
    "producer_version",
    "CanonicalOutputs",
    "to_canonical_outputs",
]

OUTPUT_SCHEMA_VERSION = "2"
#: Name of the producing software recorded on every exchanged payload.
PRODUCER_NAME = "apgi-implementation"


class IncompletePayloadError(ValueError):
    """An exchanged payload lacks a required version or identity field."""


def producer_version() -> str | None:
    """Installed version of the producing package, or ``None`` from a bare checkout."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("apgi")
    except PackageNotFoundError:
        return None


#: Canonical fields: name → (unit, description). The first four are required.
OUTPUT_SCHEMA: dict[str, tuple[str, str]] = {
    "S": ("AU", "accumulated precision-weighted signal"),
    "theta": ("AU", "ignition threshold in force"),
    "B": ("{0,1}", "ignition indicator"),
    "p_ignite": ("probability", "ignition probability"),
    "pi_e": ("AU", "exteroceptive precision"),
    "C_landauer": ("J", "Landauer minimum energy cost per step (joules)"),
    "bits_erased": ("bit", "information erased per step (bits)"),
}
_REQUIRED = ("S", "theta", "B", "p_ignite")

#: Legacy / alternative names → canonical name.
OUTPUT_ALIASES: dict[str, str] = {
    "signal": "S",
    "threshold": "theta",
    "ignition": "B",
    "p_ign": "p_ignite",
    "P_ignite": "p_ignite",
    "precision_e": "pi_e",
}


@dataclass
class CanonicalOutputs:
    """Validated model outputs, keyed by canonical field name."""

    series: dict[str, np.ndarray]
    model_id: str | None = None
    model_version: str | None = None
    schema_version: str = OUTPUT_SCHEMA_VERSION
    meta: dict[str, Any] = field(default_factory=dict)
    protocol_version: str | None = None
    producer: dict[str, str | None] = field(
        default_factory=lambda: {"name": PRODUCER_NAME, "version": producer_version()}
    )

    @property
    def n_steps(self) -> int:
        return int(next(iter(self.series.values())).shape[0])

    def to_records(self) -> list[dict[str, float]]:
        """One dict per step (row-oriented)."""
        return [{k: float(v[t]) for k, v in self.series.items()} for t in range(self.n_steps)]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "protocol_version": self.protocol_version,
            "producer": dict(self.producer),
            "units": {k: OUTPUT_SCHEMA.get(k, ("", ""))[0] for k in self.series},
            "series": {k: v.tolist() for k, v in self.series.items()},
            "meta": self.meta,
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True)

    def exchange_payload(self) -> dict[str, Any]:
        """Payload for exchange with another component (APGI-INT-001).

        Raises:
            IncompletePayloadError: If the schema, protocol or producer version,
                or the model id/version, is missing.
        """
        missing = [
            name
            for name, value in (
                ("schema_version", self.schema_version),
                ("protocol_version", self.protocol_version),
                ("producer.name", self.producer.get("name")),
                ("producer.version", self.producer.get("version")),
                ("model_id", self.model_id),
                ("model_version", self.model_version),
            )
            if not value
        ]
        if missing:
            raise IncompletePayloadError(f"exchanged payload lacks {missing}")
        return self.as_dict()


def to_canonical_outputs(
    run: Mapping[str, Any],
    *,
    model_id: str | None = None,
    model_version: str | None = None,
    keep_extra: bool = False,
    protocol_version: str | None = None,
) -> CanonicalOutputs:
    """Map a model's :meth:`run` result (or a list of step dicts) to the schema.

    Args:
        run: Mapping name → array, or ignored if a list of per-step dicts is
            passed via :func:`list_of_steps_to_run`'s output.
        model_id: Recorded verbatim.
        model_version: Recorded verbatim.
        keep_extra: Also keep unknown 1-D numeric series under their own names.
        protocol_version: Protocol version the run was executed under.

    Raises:
        ValueError: On missing required fields, ragged lengths, non-finite
            values, non-binary ``B`` or ``p_ignite`` outside [0, 1].
    """
    series: dict[str, np.ndarray] = {}
    extras: dict[str, np.ndarray] = {}
    for key, value in run.items():
        canon = OUTPUT_ALIASES.get(key, key)
        try:
            arr = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            continue
        if arr.ndim != 1:
            continue
        if canon in OUTPUT_SCHEMA:
            if canon in series and not np.array_equal(series[canon], arr):
                raise ValueError(f"conflicting values for {canon!r} (via {key!r})")
            series[canon] = arr
        elif keep_extra:
            extras[canon] = arr
    missing = [k for k in _REQUIRED if k not in series]
    if missing:
        raise ValueError(f"missing required output fields: {missing}")
    series.update(extras)
    lengths = {k: v.shape[0] for k, v in series.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"series have different lengths: {lengths}")
    for k, v in series.items():
        if not np.all(np.isfinite(v)):
            raise ValueError(f"{k}: contains non-finite values")
    if not np.all((series["B"] == 0) | (series["B"] == 1)):
        raise ValueError("B must be binary (0/1)")
    if np.any((series["p_ignite"] < 0) | (series["p_ignite"] > 1)):
        raise ValueError("p_ignite must lie in [0, 1]")
    return CanonicalOutputs(series, model_id, model_version, protocol_version=protocol_version)


def list_of_steps_to_run(steps: list[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    """Transpose per-step dicts into the column form :func:`to_canonical_outputs` expects."""
    if not steps:
        raise ValueError("no steps")
    keys = [k for k, v in steps[0].items() if isinstance(v, (int, float, np.floating, np.integer))]
    return {k: np.asarray([s[k] for s in steps], dtype=float) for k in keys}
