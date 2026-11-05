"""Canonical five-level temporal hierarchy L0–L4 (APGI-SCI-003, AUD-SCI-003).

APGI-SCI-003 requires every component to implement, label and document the
hierarchy as

==== ======================= ==========================================
L0   10 ms – 100 ms          reflexive substrate (sub-APGI, no ignition)
L1   100 ms – 3 s            first APGI ignition level
L2   1 min – 30 min
L3   hours – days
L4   months – years
==== ======================= ==========================================

This module is the single machine-readable source for those labels and
ranges. It does not change any dynamics; it lets code, outputs and docs
name levels consistently and check timescales against the canonical bands.

Index conventions — do not conflate
-----------------------------------
The per-level arrays in :mod:`pipeline`, :mod:`hierarchy.coupling` and
``config.CONFIG`` hold **APGI ignition levels only**: array index 0 is **L1**,
index 1 is L2, and so on. L0 has no slot because it never participates in
ignition. Use :func:`label_for_index` / :func:`index_for_label` rather than
re-deriving the offset. The lightweight demo configuration (``n_levels=3``,
``tau_0=10`` ms geometric spacing) is a numerical device, not a claim to
occupy these bands; :func:`audit_tau_ms` reports which bands it actually hits.

Quantification of the verbal ranges — interpretation pending confirmation
-------------------------------------------------------------------------
"hours – days" and "months – years" are verbal. They are quantified here as
L3 = [1 h, 7 d] and L4 = [1 month (30.4375 d), 100 years]. The specification
leaves gaps between L1 and L2 (3 s – 1 min), L2 and L3 (30 min – 1 h) and L3
and L4 (7 d – 1 month); :func:`level_of` returns ``None`` inside a gap rather
than guessing. These choices are recorded in :data:`SOURCE_AMBIGUITIES` and
need confirmation by the scientific authority (AUD-SCI-004 / AUD-GOV-01).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

__all__ = [
    "LevelBand",
    "LEVELS",
    "LEVEL_LABELS",
    "SOURCE_AMBIGUITIES",
    "level_of",
    "check_in_level",
    "label_for_index",
    "index_for_label",
    "audit_tau_ms",
]

_S = 1.0
_MIN = 60.0
_H = 3600.0
_D = 86_400.0
_MONTH = 30.4375 * _D
_YEAR = 365.25 * _D


@dataclass(frozen=True)
class LevelBand:
    """One canonical level: label, inclusive bounds in seconds, description."""

    label: str
    lower_s: float
    upper_s: float
    span: str
    role: str
    quantified_by_interpretation: bool = False

    def contains(self, seconds: float) -> bool:
        """Inclusive containment (both bounds belong to the band)."""
        return self.lower_s <= seconds <= self.upper_s


LEVELS: tuple[LevelBand, ...] = (
    LevelBand("L0", 0.010 * _S, 0.100 * _S, "10-100 ms", "reflexive substrate (no ignition)"),
    LevelBand("L1", 0.100 * _S, 3.0 * _S, "100 ms-3 s", "first APGI ignition level"),
    LevelBand("L2", 1.0 * _MIN, 30.0 * _MIN, "1-30 min", "APGI ignition level"),
    LevelBand("L3", 1.0 * _H, 7.0 * _D, "hours-days", "APGI ignition level", True),
    LevelBand("L4", 1.0 * _MONTH, 100.0 * _YEAR, "months-years", "APGI ignition level", True),
)
LEVEL_LABELS: tuple[str, ...] = tuple(b.label for b in LEVELS)
_BY_LABEL: dict[str, LevelBand] = {b.label: b for b in LEVELS}

#: Open source questions this module had to resolve provisionally.
SOURCE_AMBIGUITIES: tuple[str, ...] = (
    "L0/L1 share the 100 ms boundary; level_of(0.1) returns L0 (first match).",
    "Gap 3 s - 1 min between L1 and L2 is unassigned in the specification.",
    "Gap 30 min - 1 h between L2 and L3 is unassigned in the specification.",
    "'hours-days' quantified as [1 h, 7 d] by interpretation.",
    "Gap 7 d - 1 month between L3 and L4 is unassigned in the specification.",
    "'months-years' quantified as [1 month, 100 y] by interpretation.",
    "Notation Appendix tau table forces tau_theta_4 > tau_ign_4 (see hierarchy.timescales).",
)


def level_of(seconds: float) -> str | None:
    """Return the canonical label whose band contains ``seconds``, or ``None``.

    Raises:
        ValueError: If ``seconds`` is not a positive finite number.
    """
    if not (seconds > 0.0 and seconds != float("inf")):
        raise ValueError(f"timescale must be positive and finite, got {seconds!r}")
    for band in LEVELS:
        if band.contains(seconds):
            return band.label
    return None


def check_in_level(label: str, seconds: float) -> None:
    """Raise ``ValueError`` unless ``seconds`` lies in the band of ``label``."""
    try:
        band = _BY_LABEL[label]
    except KeyError:
        raise ValueError(f"unknown level {label!r}; expected one of {LEVEL_LABELS}") from None
    if not band.contains(seconds):
        raise ValueError(
            f"{seconds!r} s is outside {label} ({band.span}: [{band.lower_s}, {band.upper_s}] s)"
        )


def label_for_index(index: int) -> str:
    """Canonical label of per-level array ``index`` (index 0 is L1, not L0)."""
    if not 0 <= index < len(LEVELS) - 1:
        raise ValueError(f"ignition-level index must be in [0, {len(LEVELS) - 2}], got {index}")
    return LEVELS[index + 1].label


def index_for_label(label: str) -> int:
    """Per-level array index of ``label``; L0 has none (it never ignites)."""
    if label == "L0":
        raise ValueError("L0 is the reflexive substrate and has no ignition-level array slot")
    if label not in _BY_LABEL:
        raise ValueError(f"unknown level {label!r}; expected one of {LEVEL_LABELS}")
    return LEVEL_LABELS.index(label) - 1


def audit_tau_ms(
    taus_ms: Iterable[float] | Mapping[int, float], *, index_base: int = 0
) -> list[dict[str, object]]:
    """Report, per array slot, the intended level and the band actually hit.

    Args:
        taus_ms: Per-level timescales in milliseconds — a sequence (array index
            0 = L1) or a mapping keyed by spec level number (1..4), as used in
            :mod:`hierarchy.timescales`.
        index_base: Offset added to sequence positions before mapping to a
            label (ignored for mappings, whose keys are spec level numbers).

    Returns:
        One row per slot: ``intended``, ``tau_s``, ``actual`` (label or None)
        and ``conforms``. Nothing is raised: this is a reporting tool so demo
        configurations can state their departure explicitly.
    """
    if isinstance(taus_ms, Mapping):
        items = [(int(k) - 1, float(v)) for k, v in sorted(taus_ms.items())]
    else:
        items = [(i + index_base, float(v)) for i, v in enumerate(taus_ms)]
    rows: list[dict[str, object]] = []
    for idx, tau_ms in items:
        intended = label_for_index(idx)
        tau_s = tau_ms / 1000.0
        actual = level_of(tau_s)
        rows.append(
            {"intended": intended, "tau_s": tau_s, "actual": actual, "conforms": actual == intended}
        )
    return rows
