"""Parameter mapping across representations (S1-IMP-12.2).

One spec symbol can appear as a reduced-model parameter, a pipeline config key
and a legacy alias. This module records those correspondences so that
adapters and the manifest can translate between them, and so that *non*-
correspondences are explicit: the reduced ``alpha`` is not the pipeline's
``alpha_e`` (an EMA variance rate).

The machine-checkable registry in ``specifications/model/parameter-registry.yaml``
(S1-REG-01) lives in the spec repository. :func:`load_parameter_registry` reads
it from a JSON rendering when one is supplied, and :func:`check_against_registry`
reports drift against :data:`MAPPINGS`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from models.parameters import REDUCED_PARAMETER_NAMES, REDUCED_PARAMETERS

__all__ = [
    "ParameterMapping",
    "MAPPINGS",
    "reduced_to_pipeline",
    "pipeline_to_reduced",
    "canonical_name",
    "bounds",
    "load_parameter_registry",
    "check_against_registry",
]


@dataclass(frozen=True)
class ParameterMapping:
    """Correspondence of one spec symbol across representations.

    ``reduced`` / ``pipeline`` are ``None`` where the symbol has no counterpart.
    ``note`` records any caveat on the correspondence.
    """

    symbol: str
    reduced: str | None
    pipeline: str | None
    aliases: tuple[str, ...] = ()
    note: str = ""


MAPPINGS: tuple[ParameterMapping, ...] = (
    ParameterMapping(
        "α",
        "alpha",
        "kappa_e",
        note="Pipeline prediction update is x̂ += κ_e·Π·ε; the reduced model uses the "
        "bounded gain k=αΠ/(1+αΠ). Same product structure, different link function.",
    ),
    ParameterMapping(
        "Π",
        "pi",
        None,
        note="Pipeline precision is estimated online (1/σ², clamped to [pi_min, pi_max]); "
        "the reduced model treats it as a fixed parameter.",
    ),
    ParameterMapping("θ₀", "theta_base", "theta_base", aliases=("theta_0",)),
    ParameterMapping("β_DA", "beta", "beta", aliases=("beta_da",)),
    ParameterMapping("τ_σ", None, "ignite_tau", aliases=("tau_sigma",)),
    ParameterMapping("κ", None, "kappa"),
    ParameterMapping("δ", None, "delta"),
    ParameterMapping("λ", None, "lam"),
    ParameterMapping("β_SM", None, "beta_sm", aliases=("beta_somatic",)),
)

_ALIAS: dict[str, str] = {}
for _m in MAPPINGS:
    for _name in (_m.reduced, _m.pipeline, *_m.aliases):
        if _name is not None:
            _ALIAS[_name] = _m.reduced or _m.pipeline or _name


def canonical_name(name: str) -> str:
    """Resolve ``name`` (alias, pipeline key or reduced name) to its canonical form."""
    return _ALIAS.get(name, name)


def reduced_to_pipeline(params: Mapping[str, float]) -> dict[str, float]:
    """Translate reduced-model parameters to the pipeline keys that have a counterpart.

    Parameters with no pipeline counterpart (``pi``) are omitted rather than
    guessed; callers must handle them explicitly.
    """
    by_reduced = {m.reduced: m.pipeline for m in MAPPINGS if m.reduced and m.pipeline}
    return {by_reduced[k]: v for k, v in params.items() if k in by_reduced}


def pipeline_to_reduced(config: Mapping[str, Any]) -> dict[str, Any]:
    """Inverse of :func:`reduced_to_pipeline` for keys present in ``config``."""
    by_pipeline = {m.pipeline: m.reduced for m in MAPPINGS if m.reduced and m.pipeline}
    return {by_pipeline[k]: v for k, v in config.items() if k in by_pipeline}


def bounds(name: str) -> tuple[float, float]:
    """Hard support ``(lower, upper)`` of a reduced-model parameter."""
    canonical = canonical_name(name)
    for spec in REDUCED_PARAMETERS:
        if spec.name == canonical:
            return spec.lower, spec.upper
    raise KeyError(f"no bounds registered for {name!r}")


def load_parameter_registry(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load a JSON rendering of the spec repository's parameter registry.

    Expected shape: ``{"parameters": {"<name>": {"lower": .., "upper": .., "symbol": ..}}}``.

    Raises:
        ValueError: If the file lacks a ``parameters`` mapping.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    params = data.get("parameters") if isinstance(data, dict) else None
    if not isinstance(params, dict):
        raise ValueError(f"{path}: expected an object with a 'parameters' mapping")
    return dict(params)


def check_against_registry(registry: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Return human-readable drift between ``registry`` and the reduced-model bounds."""
    problems: list[str] = []
    for spec in REDUCED_PARAMETERS:
        entry = registry.get(spec.name)
        if entry is None:
            problems.append(f"{spec.name}: missing from registry")
            continue
        for key, mine in (("lower", spec.lower), ("upper", spec.upper)):
            theirs = entry.get(key)
            if theirs is None or float(theirs) != mine:
                problems.append(f"{spec.name}.{key}: registry={theirs!r} implementation={mine!r}")
        if entry.get("symbol") not in (None, spec.symbol):
            problems.append(
                f"{spec.name}.symbol: registry={entry.get('symbol')!r} != {spec.symbol!r}"
            )
    extra = set(registry) - set(REDUCED_PARAMETER_NAMES)
    for name in sorted(extra):
        problems.append(f"{name}: in registry but not implemented by apgi_reduced")
    return problems
