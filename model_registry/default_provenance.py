"""Provenance of every default parameter value (APGI-SCI-005, AUD-SCI-005).

APGI-SCI-005 requires each default to be labelled *analytically derived*,
*post-hoc calibrated* or *assumed*, with a source reference. This registry
covers

* the qualified reference model ``apgi_reduced`` — its nominal parameters,
  observation-noise constants and fixed model constants; and
* every numeric default in ``config.CONFIG`` (the pipeline defaults).

Entries that have not yet been reviewed carry :data:`UNCLASSIFIED`. That label
is deliberately not one of the three permitted bases: it marks an open gap,
keeps APGI-SCI-005 visibly open, and must be replaced by the approved
parameter registry (S1-REG-01) before certification. The basis given for
reviewed entries is this repository's own reading of where the value came
from; it is not an approval.

``tests/test_default_provenance.py`` fails when a default is added, removed or
changed in code without updating this registry (drift check).
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Any

__all__ = [
    "BASES",
    "UNCLASSIFIED",
    "DefaultProvenance",
    "REDUCED_DEFAULTS",
    "config_defaults",
    "all_defaults",
    "provenance_for",
    "provenance_summary",
]

#: The three bases APGI-SCI-005 permits.
BASES: tuple[str, ...] = ("analytically_derived", "post_hoc_calibrated", "assumed")
#: Marker for defaults not yet reviewed (an open gap, not a permitted basis).
UNCLASSIFIED = "unclassified"

_UNREVIEWED_SOURCE = "config.py default; not yet reviewed against S1-REG-01 parameter-registry.yaml"


@dataclass(frozen=True)
class DefaultProvenance:
    """Where one default value comes from."""

    name: str
    value: float
    basis: str
    source: str
    scope: str

    def __post_init__(self) -> None:
        if self.basis not in (*BASES, UNCLASSIFIED):
            raise ValueError(f"{self.name}: basis {self.basis!r} not in {BASES + (UNCLASSIFIED,)}")
        if not self.source:
            raise ValueError(f"{self.name}: source reference is required")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "basis": self.basis,
            "source": self.source,
            "scope": self.scope,
        }


def _r(name: str, value: float, basis: str, source: str) -> DefaultProvenance:
    return DefaultProvenance(name, value, basis, source, "apgi_reduced")


#: ``apgi_reduced`` defaults (``ReducedParams`` and module constants).
REDUCED_DEFAULTS: tuple[DefaultProvenance, ...] = (
    _r("alpha", 0.3, "assumed", "models/parameters.py nominal; no calibration source"),
    _r("pi", 1.5, "assumed", "models/parameters.py nominal inside Π_min..Π_max [0.1, 10]"),
    _r(
        "theta_base",
        0.55,
        "assumed",
        "midpoint of the Notation Appendix canonical θ₀ range [0.25, 0.85]",
    ),
    _r("beta", 0.05, "assumed", "models/parameters.py nominal; no calibration source"),
    _r("sigma_r", 0.15, "assumed", "report-noise nuisance constant (models/reduced.py)"),
    _r("sigma_y", 0.10, "assumed", "amplitude-noise nuisance constant (models/reduced.py)"),
    _r(
        "LAM",
        1.0 - math.exp(-0.5 / 5.0),
        "analytically_derived",
        "λ = 1 − exp(−dt/τ_S) (exact discretisation of the leaky integrator, MathSpec §3) "
        "with dt=0.5 and τ_S=5 — dt and τ_S themselves are assumed",
    ),
    _r("RHO", 0.1, "assumed", "RHO_RETAIN post-ignition retention, MathSpec §6.1 default"),
    _r("KAPPA", 0.15, "assumed", "threshold relaxation κ, config.CONFIG['kappa']"),
    _r("DELTA", 0.5, "assumed", "refractory boost δ, config.CONFIG['delta']"),
    _r("TAU_SIGMA", 0.5, "assumed", "ignition sigmoid temperature, config.CONFIG['ignite_tau']"),
    _r("GAMMA_PHI", 2.0, "assumed", "φ-transform steepness, config.CONFIG['gamma_plus']"),
)

#: Reviewed pipeline defaults: name → (basis, source). Everything else numeric
#: in ``config.CONFIG`` is listed as :data:`UNCLASSIFIED`.
_CONFIG_REVIEWED: dict[str, tuple[str, str]] = {
    "lam": (
        "analytically_derived",
        "λ = 1 − exp(−dt/τ_S) for dt=0.5, τ_S=5.0 (config.py comment, MathSpec §3)",
    ),
    "k_boltzmann": ("assumed", "CODATA exact SI value of k_B (J/K)"),
    "T_env": ("assumed", "nominal human body temperature 310 K"),
    "dt": ("assumed", "integration step; MathSpec §7.4 requires dt ≤ min(τ)/10"),
    "tau_s": ("assumed", "signal time constant τ_S default (MathSpec §3)"),
    "theta_0": ("assumed", "midpoint of the Notation Appendix canonical θ₀ range [0.25, 0.85]"),
    "theta_base": ("assumed", "alias of theta_0"),
    "pi_min": ("assumed", "Notation Appendix canonical Π_min"),
    "pi_max": ("assumed", "Notation Appendix canonical Π_max"),
    "eps": ("assumed", "numerical guard ε for divisions, not a scientific quantity"),
    "bold_conversion_factor": (
        "assumed",
        "J per 1% BOLD change per cm³ (config.py); literature source not recorded",
    ),
    "bold_ignition_spike_factor": ("assumed", "7.5% spike, midpoint of a quoted 5-10% range"),
    "kappa_meta": ("assumed", "dimensionless Joules→AU scaling; no calibration record"),
    "history_maxlen": ("assumed", "memory bound for diagnostic history, not scientific"),
}


def config_defaults() -> tuple[DefaultProvenance, ...]:
    """Provenance rows for every numeric default in ``config.CONFIG``."""
    from config import CONFIG

    rows: list[DefaultProvenance] = []
    for name, value in CONFIG.items():
        if isinstance(value, bool) or not isinstance(value, numbers.Number):
            continue
        basis, source = _CONFIG_REVIEWED.get(name, (UNCLASSIFIED, _UNREVIEWED_SOURCE))
        rows.append(DefaultProvenance(name, float(value), basis, source, "config.CONFIG"))
    return tuple(rows)


def all_defaults() -> tuple[DefaultProvenance, ...]:
    """Reduced-model and pipeline defaults together."""
    return REDUCED_DEFAULTS + config_defaults()


def provenance_for(name: str, scope: str = "apgi_reduced") -> DefaultProvenance:
    """Provenance row for ``name`` in ``scope`` (``apgi_reduced`` or ``config.CONFIG``).

    Raises:
        KeyError: If the default is not registered.
    """
    for row in all_defaults():
        if row.name == name and row.scope == scope:
            return row
    raise KeyError(f"no provenance registered for {scope}:{name}")


def provenance_summary() -> dict[str, int]:
    """Count of defaults per basis — the open gap is ``unclassified``."""
    counts: dict[str, int] = dict.fromkeys((*BASES, UNCLASSIFIED), 0)
    for row in all_defaults():
        counts[row.basis] += 1
    return counts
