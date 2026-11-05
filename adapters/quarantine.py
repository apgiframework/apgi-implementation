"""Record-level screening, quarantine and reconciliation (APGI-DAT-002, AUD-DAT-002).

:func:`adapters.to_canonical_inputs` rejects a whole series that contains a bad
value. When inputs arrive as individual *records* (trials, events) the
application must instead segregate the bad ones, give each a named outcome and
account for every record. :func:`screen_records` does that:

=================  ==========================================================
outcome            meaning
=================  ==========================================================
``accepted``       valid, first occurrence, in order, expected (if a list is given)
``malformed``      missing required field, non-numeric, NaN/inf, or no id
``duplicate``      same id and identical content as an accepted record
``conflict``       same id but *different* content from an accepted record
``delayed``        sequence number lower than one already accepted (out of order)
``unmatched``      id not in ``expected_ids``
=================  ==========================================================

Expected ids that never arrive are reported as ``missing``. The report's
:meth:`ScreeningReport.reconcile` checks the accounting identities
(``received = accepted + quarantined`` and, with an expected list,
``expected = accepted + missing``) and raises if any record is unaccounted
for. Quarantined records are retained verbatim with their reason so they can
be reviewed; none of them reach a model.

Which outcomes are acceptable for a given study is a profile decision (see
``docs/APPLICATION-VALIDATION-PROFILE.md``); this module only classifies and
counts.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

__all__ = [
    "OUTCOMES",
    "QuarantinedRecord",
    "ScreeningReport",
    "ReconciliationError",
    "screen_records",
]

OUTCOMES: tuple[str, ...] = (
    "accepted",
    "malformed",
    "duplicate",
    "conflict",
    "delayed",
    "unmatched",
)


class ReconciliationError(RuntimeError):
    """Record counts do not reconcile; some input is unaccounted for."""


@dataclass(frozen=True)
class QuarantinedRecord:
    """A segregated record with its outcome and reason."""

    index: int
    record: Mapping[str, Any]
    outcome: str
    reason: str


@dataclass
class ScreeningReport:
    """Result of :func:`screen_records`."""

    accepted: list[Mapping[str, Any]]
    quarantined: list[QuarantinedRecord]
    missing: list[Any]
    n_received: int
    n_expected: int | None
    counts: dict[str, int] = field(default_factory=dict)

    def reconcile(self) -> dict[str, int]:
        """Check the accounting identities; return the counts.

        Raises:
            ReconciliationError: If any record is unaccounted for.
        """
        if len(self.accepted) + len(self.quarantined) != self.n_received:
            raise ReconciliationError(
                f"received {self.n_received} but accepted {len(self.accepted)} + "
                f"quarantined {len(self.quarantined)}"
            )
        if self.n_expected is not None and len(self.accepted) + len(self.missing) != (
            self.n_expected
        ):
            raise ReconciliationError(
                f"expected {self.n_expected} but accepted {len(self.accepted)} + "
                f"missing {len(self.missing)}"
            )
        return dict(self.counts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_received": self.n_received,
            "n_expected": self.n_expected,
            "counts": dict(self.counts),
            "missing": list(self.missing),
            "quarantined": [
                {"index": q.index, "outcome": q.outcome, "reason": q.reason}
                for q in self.quarantined
            ],
        }


def _canonical(record: Mapping[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, default=repr)


def _numeric_problem(record: Mapping[str, Any], fields: Sequence[str]) -> str | None:
    for name in fields:
        if name not in record or record[name] is None:
            return f"missing required field {name!r}"
        value = record[name]
        if isinstance(value, bool) or not isinstance(value, Real):
            return f"field {name!r} is not a real number ({type(value).__name__})"
        if not math.isfinite(float(value)):
            return f"field {name!r} is non-finite ({value!r})"
    return None


def screen_records(
    records: Iterable[Mapping[str, Any]],
    *,
    id_field: str = "trial_id",
    numeric_fields: Sequence[str] = ("x_e",),
    seq_field: str | None = None,
    expected_ids: Iterable[Any] | None = None,
) -> ScreeningReport:
    """Classify every record; accept only valid, unique, in-order, expected ones.

    Args:
        records: Input records (mappings).
        id_field: Field holding the record identity.
        numeric_fields: Fields that must be present, real and finite.
        seq_field: Optional ordering field; a record whose value is below the
            highest accepted one is ``delayed``.
        expected_ids: Optional list of ids the batch should contain.

    Returns:
        A :class:`ScreeningReport` whose counts already reconcile.
    """
    expected = None if expected_ids is None else list(dict.fromkeys(expected_ids))
    expected_set = None if expected is None else set(expected)
    accepted: list[Mapping[str, Any]] = []
    quarantined: list[QuarantinedRecord] = []
    seen: dict[Any, str] = {}
    max_seq: float | None = None
    n = 0
    for n, rec in enumerate(records, start=1):
        idx = n - 1
        if not isinstance(rec, Mapping):
            quarantined.append(QuarantinedRecord(idx, {"value": rec}, "malformed", "not a mapping"))
            continue
        rid = rec.get(id_field)
        problem = "missing record id" if rid is None else _numeric_problem(rec, numeric_fields)
        if problem is None and seq_field is not None:
            problem = _numeric_problem(rec, (seq_field,))
        if problem is not None:
            quarantined.append(QuarantinedRecord(idx, dict(rec), "malformed", problem))
            continue
        if rid in seen:
            same = seen[rid] == _canonical(rec)
            outcome = "duplicate" if same else "conflict"
            reason = f"id {rid!r} already accepted" + ("" if same else " with different content")
            quarantined.append(QuarantinedRecord(idx, dict(rec), outcome, reason))
            continue
        if expected_set is not None and rid not in expected_set:
            quarantined.append(
                QuarantinedRecord(idx, dict(rec), "unmatched", f"id {rid!r} not expected")
            )
            continue
        if seq_field is not None:
            seq = float(rec[seq_field])
            if max_seq is not None and seq < max_seq:
                quarantined.append(
                    QuarantinedRecord(
                        idx, dict(rec), "delayed", f"{seq_field}={seq} after {max_seq}"
                    )
                )
                continue
            max_seq = seq
        seen[rid] = _canonical(rec)
        accepted.append(rec)
    missing = [] if expected is None else [i for i in expected if i not in seen]
    counts = dict.fromkeys(OUTCOMES, 0)
    counts["accepted"] = len(accepted)
    for q in quarantined:
        counts[q.outcome] += 1
    counts["missing"] = len(missing)
    report = ScreeningReport(
        accepted=accepted,
        quarantined=quarantined,
        missing=missing,
        n_received=n,
        n_expected=None if expected is None else len(expected),
        counts=counts,
    )
    report.reconcile()
    return report
