"""Adapters: canonical inputs, canonical outputs, version adapter (S1-IMP-10)."""

from .canonical_inputs import CanonicalInputs, to_canonical_inputs
from .canonical_outputs import (
    OUTPUT_SCHEMA,
    CanonicalOutputs,
    list_of_steps_to_run,
    to_canonical_outputs,
)
from .version_adapter import (
    DEPRECATIONS,
    SPEC_VERSION,
    ConflictingAliasError,
    Deprecation,
    adapt_config,
    adapt_output_keys,
    migrate,
    register_migration,
)

__all__ = [
    "CanonicalInputs",
    "to_canonical_inputs",
    "OUTPUT_SCHEMA",
    "CanonicalOutputs",
    "list_of_steps_to_run",
    "to_canonical_outputs",
    "DEPRECATIONS",
    "SPEC_VERSION",
    "ConflictingAliasError",
    "Deprecation",
    "adapt_config",
    "adapt_output_keys",
    "migrate",
    "register_migration",
]
