"""Registry of APGI model variants (S1-IMP-12.1).

Seven variants are registered. Exactly one — ``apgi_reduced`` — carries
numerical-reference authority (decision D14); the registry refuses to register
a second authoritative variant so that cannot drift.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from models import (
    ApgiAgent,
    ApgiExtended,
    ApgiHierarchical,
    ApgiLiquid,
    APGIModel,
    ApgiNull,
    ApgiReduced,
    ApgiReference,
)

__all__ = [
    "ModelVariant",
    "REFERENCE_MODEL_ID",
    "get_variant",
    "list_variants",
    "register_variant",
    "unregister_variant",
    "create_model",
    "reference_variant",
]

#: The single qualified numerical reference (decision D14).
REFERENCE_MODEL_ID = "apgi_reduced"


@dataclass(frozen=True)
class ModelVariant:
    """Registry entry for one model class."""

    model_id: str
    cls: type[APGIModel]
    description: str

    @property
    def version(self) -> str:
        return self.cls.version

    @property
    def status(self) -> str:
        return self.cls.status

    @property
    def authority(self) -> bool:
        return self.cls.authority

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return self.cls.parameter_names

    def describe(self) -> dict[str, Any]:
        """JSON-serialisable description (used in manifests)."""
        return {
            "model_id": self.model_id,
            "version": self.version,
            "status": self.status,
            "authority": self.authority,
            "parameter_names": list(self.parameter_names),
            "description": self.description,
        }


_REGISTRY: dict[str, ModelVariant] = {}


def register_variant(variant: ModelVariant, *, replace: bool = False) -> None:
    """Add ``variant`` to the registry.

    Raises:
        ValueError: On a duplicate id (unless ``replace``), when the class's
            ``model_id`` disagrees with the entry, or when a second authoritative
            variant would be registered.
    """
    if variant.cls.model_id != variant.model_id:
        raise ValueError(f"class model_id {variant.cls.model_id!r} != entry {variant.model_id!r}")
    if variant.model_id in _REGISTRY and not replace:
        raise ValueError(f"variant {variant.model_id!r} already registered")
    if variant.authority:
        for other in _REGISTRY.values():
            if other.authority and other.model_id != variant.model_id:
                raise ValueError(
                    f"{other.model_id!r} already holds reference authority (D14); "
                    f"cannot grant it to {variant.model_id!r}"
                )
    _REGISTRY[variant.model_id] = variant


def unregister_variant(model_id: str) -> None:
    """Remove a variant (primarily for tests of third-party registration)."""
    _REGISTRY.pop(model_id, None)


def get_variant(model_id: str) -> ModelVariant:
    """Return the entry for ``model_id``.

    Raises:
        KeyError: With the list of known ids if it is not registered.
    """
    try:
        return _REGISTRY[model_id]
    except KeyError:
        raise KeyError(f"unknown model {model_id!r}; registered: {sorted(_REGISTRY)}") from None


def list_variants() -> list[ModelVariant]:
    """All registered variants, sorted by id."""
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]


def reference_variant() -> ModelVariant:
    """The single authoritative variant."""
    return get_variant(REFERENCE_MODEL_ID)


def create_model(
    model_id: str, params: Mapping[str, Any] | None = None, seed: int | None = None
) -> APGIModel:
    """Instantiate a registered variant."""
    return get_variant(model_id).cls(params, seed)


for _cls, _desc in (
    (ApgiReference, "Fixture-conformant reading of the spec (pipeline at canonical defaults)."),
    (ApgiReduced, "Registered reduced model; the single qualified numerical reference."),
    (ApgiExtended, "Exploratory: somatic precision, NE threshold pathway, circadian drift."),
    (ApgiHierarchical, "Extension-specific: multi-timescale hierarchy with precision ODE."),
    (ApgiLiquid, "Extension-specific: liquid-state-machine reservoir layer."),
    (ApgiAgent, "Computational comparison: perception-action loop."),
    (ApgiNull, "Benchmark: constant-rate ignition independent of input."),
):
    register_variant(ModelVariant(_cls.model_id, _cls, _desc))
