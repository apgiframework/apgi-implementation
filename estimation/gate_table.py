"""Per-parameter gate table (S1-K7-15).

Combines K7a (structural identifiability) and K7c (recovery) classifications
into ``gates/gate-table.yaml``. The table drives the battery allow-list: only
parameters flagged ``exposable`` may be reported from a task.

Statuses are ``CONFIRMED`` / ``FALSIFIED`` / ``INDETERMINATE``. K7b is the
empirical (data-dependent) gate and cannot be resolved by desk science: every
row carries ``k7b: PENDING_DATA``. ``exposable`` is therefore the *desk-science*
verdict — K7a and K7c both CONFIRMED — and the allow-list stays provisional
until K7b reports.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from models.parameters import REDUCED_PARAMETERS

from .composite import COMPOSITE_NAME
from .recovery_simulation import COVERAGE_RANGE, R_MIN, RecoveryResult, evaluate_criteria

__all__ = [
    "STATUSES",
    "K7B_PENDING",
    "classify_k7c",
    "build_gate_table",
    "dump_yaml",
    "write_gate_table",
    "allow_list",
]

STATUSES = ("CONFIRMED", "FALSIFIED", "INDETERMINATE")
K7B_PENDING = "PENDING_DATA"
SCHEMA_VERSION = "1"


def _wilson(coverage: float, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson 95% interval for an observed proportion."""
    if n <= 0:
        return 0.0, 1.0
    denom = 1.0 + z * z / n
    centre = (coverage + z * z / (2 * n)) / denom
    half = z * math.sqrt(coverage * (1.0 - coverage) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def _coverage_marginal(coverage: float, n: int) -> bool:
    """Is coverage outside the window but statistically compatible with it?

    With ``n`` subjects the observed coverage carries binomial error; a value
    outside ``[0.90, 0.98]`` whose Wilson 95% interval overlaps the window
    cannot be called a miscalibration, so the gate is INDETERMINATE rather than
    FALSIFIED.
    """
    lo, hi = COVERAGE_RANGE
    if lo <= coverage <= hi:
        return False
    w_lo, w_hi = _wilson(coverage, n)
    return w_hi >= lo and w_lo <= hi


def classify_k7c(result: RecoveryResult) -> dict[str, dict[str, Any]]:
    """K7c classification per parameter, plus the ``αΠ`` composite.

    * ``INDETERMINATE`` — sampler did not converge (R̂/ESS), or the only failure
      is a coverage value too close to the window edge to resolve with
      ``n_subjects`` subjects;
    * ``CONFIRMED`` — all criteria met;
    * ``FALSIFIED`` — otherwise (recovery below threshold, interval miscalibration,
      or a pairwise posterior correlation at or above the limit).
    """
    crit = evaluate_criteria(result)
    out: dict[str, dict[str, Any]] = {}
    for name, c in crit.items():
        if c["converged"] is False:
            status = "INDETERMINATE"
        elif c["passes"]:
            status = "CONFIRMED"
        elif (
            c["r_ok"]
            and c["pairs_ok"]
            and _coverage_marginal(float(str(c["coverage95"])), result.n_subjects)
        ):
            status = "INDETERMINATE"
        else:
            status = "FALSIFIED"
        out[name] = {**c, "status": status}
    cm = result.composite_metrics
    comp_ok = bool(cm["r"] >= R_MIN and COVERAGE_RANGE[0] <= cm["coverage95"] <= COVERAGE_RANGE[1])
    comp_status = (
        "CONFIRMED"
        if comp_ok
        else "INDETERMINATE"
        if cm["r"] >= R_MIN and _coverage_marginal(cm["coverage95"], result.n_subjects)
        else "FALSIFIED"
    )
    out[COMPOSITE_NAME] = {**cm, "status": comp_status}
    return out


def _reason(name: str, k7a: str, k7c: str, detail: Mapping[str, Any]) -> str:
    if k7a == "CONFIRMED" and k7c == "CONFIRMED":
        return "structurally identifiable and recoverable"
    parts = []
    if k7a != "CONFIRMED":
        parts.append(f"K7a {k7a}")
    if k7c != "CONFIRMED":
        why = []
        if detail.get("r_ok") is False:
            why.append(f"r={detail.get('r'):.2f}<{R_MIN}")
        if detail.get("coverage_ok") is False:
            why.append(f"coverage={detail.get('coverage95'):.2f}")
        if detail.get("pairs_ok") is False:
            why.append(f"pairwise |r|={detail.get('worst_pairwise_abs_corr'):.2f}")
        if detail.get("converged") is False:
            why.append("not converged")
        parts.append(f"K7c {k7c}" + (f" ({', '.join(why)})" if why else ""))
    return "; ".join(parts)


def build_gate_table(
    k7a: Mapping[str, Any],
    k7c: RecoveryResult,
    *,
    meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the gate table.

    Args:
        k7a: Output of :func:`estimation.structural_identifiability.run_k7a`
            for the production design.
        k7c: Recovery result for the same design.
        meta: Extra provenance (manifest digest, seeds, ...), stored under ``meta``.
    """
    k7c_cls = classify_k7c(k7c)
    rows: list[dict[str, Any]] = []
    symbols = {p.name: p.symbol for p in REDUCED_PARAMETERS}
    for name, k7a_status in dict(k7a["classification"]).items():
        c = k7c_cls[name]
        rows.append(
            {
                "parameter": name,
                "symbol": symbols[name],
                "k7a": k7a_status,
                "k7b": K7B_PENDING,
                "k7c": c["status"],
                "exposable": bool(k7a_status == "CONFIRMED" and c["status"] == "CONFIRMED"),
                "reason": _reason(name, k7a_status, c["status"], c),
                "recovery_r": round(float(c["r"]), 4),
                "coverage95": round(float(c["coverage95"]), 4),
                "nrmse": round(float(c["nrmse"]), 4),
                "bias": round(float(c["bias"]), 5),
            }
        )
    comp_a = dict(k7a["composite"])["status"]
    comp_c = k7c_cls[COMPOSITE_NAME]
    rows.append(
        {
            "parameter": COMPOSITE_NAME,
            "symbol": "αΠ",
            "k7a": comp_a,
            "k7b": K7B_PENDING,
            "k7c": comp_c["status"],
            "exposable": bool(comp_a == "CONFIRMED" and comp_c["status"] == "CONFIRMED"),
            "reason": _reason(COMPOSITE_NAME, comp_a, comp_c["status"], comp_c),
            "recovery_r": round(float(comp_c["r"]), 4),
            "coverage95": round(float(comp_c["coverage95"]), 4),
            "nrmse": round(float(comp_c["nrmse"]), 4),
            "bias": round(float(comp_c["bias"]), 5),
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "design": {
            "channels": list(k7a["channels"]),
            "n_subjects": k7c.n_subjects,
            "estimator": k7c.estimator,
        },
        "k7b_note": "K7b is empirical and cannot be resolved by desk science; "
        "exposable reflects K7a and K7c only and is provisional until K7b reports.",
        "criteria": {
            "recovery_r_min": R_MIN,
            "pairwise_abs_posterior_corr_max": 0.60,
            "coverage95_range": list(COVERAGE_RANGE),
        },
        "meta": dict(meta or {}),
        "parameters": rows,
        "allow_list": [r["parameter"] for r in rows if r["exposable"]],
    }


def allow_list(table: Mapping[str, Any]) -> list[str]:
    """Parameters that may be exposed according to ``table``."""
    return [r["parameter"] for r in table["parameters"] if r["exposable"]]


def _scalar(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(float(v)) if isinstance(v, float) else str(v)
    return json.dumps(str(v), ensure_ascii=False)


def dump_yaml(obj: Any, indent: int = 0) -> str:
    """Minimal block-style YAML emitter for dict/list/scalar trees.

    The repository avoids a YAML runtime dependency; the output is valid YAML 1.2
    (strings are JSON-quoted) and round-trips through any YAML parser. Lists of
    scalars are written in flow style, lists of mappings in block style.
    """
    pad = "  " * indent
    if isinstance(obj, Mapping):
        if not obj:
            return pad + "{}\n"
        lines = []
        for k, v in obj.items():
            key = json.dumps(str(k), ensure_ascii=False) if not str(k).isidentifier() else str(k)
            if (
                isinstance(v, Mapping)
                and v
                or isinstance(v, (list, tuple))
                and v
                and not all(not isinstance(i, (Mapping, list, tuple)) for i in v)
            ):
                lines.append(f"{pad}{key}:\n{dump_yaml(v, indent + 1)}")
            elif isinstance(v, (list, tuple)):
                lines.append(f"{pad}{key}: [{', '.join(_scalar(i) for i in v)}]\n")
            elif isinstance(v, Mapping):
                lines.append(f"{pad}{key}: {{}}\n")
            else:
                lines.append(f"{pad}{key}: {_scalar(v)}\n")
        return "".join(lines)
    if isinstance(obj, (list, tuple)):
        lines = []
        for item in obj:
            if isinstance(item, Mapping) and item:
                body = dump_yaml(item, indent + 1).splitlines()
                first = body[0].lstrip()
                lines.append(f"{pad}- {first}\n" + "".join(f"{b}\n" for b in body[1:]))
            else:
                lines.append(f"{pad}- {_scalar(item)}\n")
        return "".join(lines)
    return pad + _scalar(obj) + "\n"


def write_gate_table(table: Mapping[str, Any], path: str | Path) -> Path:
    """Write ``table`` as YAML at ``path`` (and a ``.json`` twin beside it)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Per-parameter gate table (S1-K7-15). Generated by tools/run_k7.py - do not edit by hand.\n"
        "# Statuses: CONFIRMED / FALSIFIED / INDETERMINATE. k7b is pending real data.\n"
    )
    target.write_text(header + dump_yaml(table), encoding="utf-8")
    target.with_suffix(".json").write_text(
        json.dumps(table, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    return target


def channels_key(channels: Sequence[str]) -> str:
    """Stable label for a channel set."""
    return "+".join(channels)
