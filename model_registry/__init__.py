"""Model registry: variants and parameter mapping (S1-IMP-12)."""

from .parameter_mapping import (
    MAPPINGS,
    ParameterMapping,
    bounds,
    canonical_name,
    check_against_registry,
    load_parameter_registry,
    pipeline_to_reduced,
    reduced_to_pipeline,
)
from .variants import (
    REFERENCE_MODEL_ID,
    ModelVariant,
    create_model,
    get_variant,
    list_variants,
    reference_variant,
    register_variant,
    unregister_variant,
)

__all__ = [
    "MAPPINGS",
    "ParameterMapping",
    "bounds",
    "canonical_name",
    "check_against_registry",
    "load_parameter_registry",
    "pipeline_to_reduced",
    "reduced_to_pipeline",
    "REFERENCE_MODEL_ID",
    "ModelVariant",
    "create_model",
    "get_variant",
    "list_variants",
    "reference_variant",
    "register_variant",
    "unregister_variant",
]
