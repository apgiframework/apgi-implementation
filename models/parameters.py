"""Parameter definitions shared by every APGI model variant.

This module is a dependency-free leaf: it states, once, which parameters the
qualified reference model (``apgi_reduced``) exposes, their support, their
plausible ranges for desk-science analyses (K7a/K7c), and the spec symbols they
correspond to. :mod:`model_registry.parameter_mapping` builds the cross-model
mapping on top of it.

Symbol identity is normative in the Notation Appendix, so each entry records
the spec symbol next to the code name.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "ParameterSpec",
    "REDUCED_PARAMETERS",
    "REDUCED_PARAMETER_NAMES",
    "reduced_parameter_spec",
]


@dataclass(frozen=True)
class ParameterSpec:
    """Definition of one model parameter.

    Attributes:
        name: Code name (key in ``ReducedParams`` and in parameter dicts).
        symbol: Spec symbol from the Notation Appendix.
        description: One-line meaning.
        lower: Hard lower bound of the mathematical support.
        upper: Hard upper bound of the mathematical support.
        plausible_lower: Lower end of the plausible range swept by K7a/K7c.
        plausible_upper: Upper end of the plausible range swept by K7a/K7c.
        nominal: Nominal value used for reference fixtures.
        log_scale: Whether the plausible range is sampled log-uniformly.
        units: Unit label (``"AU"`` for arbitrary units).
    """

    name: str
    symbol: str
    description: str
    lower: float
    upper: float
    plausible_lower: float
    plausible_upper: float
    nominal: float
    log_scale: bool = False
    units: str = "AU"

    def contains(self, value: float) -> bool:
        """Return True if ``value`` lies inside the hard support."""
        return self.lower <= value <= self.upper


#: Parameters of the qualified reference model, in canonical order.
REDUCED_PARAMETERS: tuple[ParameterSpec, ...] = (
    ParameterSpec(
        name="alpha",
        symbol="α",
        description="Learning-rate factor; the prediction gain is k = αΠ/(1+αΠ)",
        lower=1e-3,
        upper=1.0,
        plausible_lower=0.05,
        plausible_upper=0.9,
        nominal=0.3,
        units="dimensionless",
    ),
    ParameterSpec(
        name="pi",
        symbol="Π",
        description="Precision weight on the prediction error (Π_min..Π_max clamp)",
        lower=0.1,
        upper=10.0,
        plausible_lower=0.3,
        plausible_upper=5.0,
        nominal=1.5,
        log_scale=True,
    ),
    ParameterSpec(
        name="theta_base",
        symbol="θ₀",
        description="Homeostatic baseline of the ignition threshold",
        lower=0.25,
        upper=0.85,
        plausible_lower=0.25,
        plausible_upper=0.85,
        nominal=0.55,
    ),
    ParameterSpec(
        name="beta",
        symbol="β_DA",
        description="Additive dopaminergic bias on the accumulated drive",
        lower=-1.0,
        upper=1.0,
        plausible_lower=-0.3,
        plausible_upper=0.3,
        nominal=0.05,
    ),
)

REDUCED_PARAMETER_NAMES: tuple[str, ...] = tuple(p.name for p in REDUCED_PARAMETERS)

_BY_NAME = {p.name: p for p in REDUCED_PARAMETERS}


def reduced_parameter_spec(name: str) -> ParameterSpec:
    """Return the :class:`ParameterSpec` for ``name``.

    Raises:
        KeyError: If ``name`` is not a reduced-model parameter.
    """
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError(
            f"unknown reduced-model parameter {name!r}; expected one of {REDUCED_PARAMETER_NAMES}"
        ) from None
