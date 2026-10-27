"""APGI model variants (S1-IMP-14).

``apgi_reduced`` is the single qualified numerical reference; see
:mod:`model_registry.variants` for the registry that exposes all seven.
"""

from .base import RUN_KEYS, APGIModel
from .parameters import (
    REDUCED_PARAMETER_NAMES,
    REDUCED_PARAMETERS,
    ParameterSpec,
    reduced_parameter_spec,
)
from .pipeline_models import (
    ApgiAgent,
    ApgiExtended,
    ApgiHierarchical,
    ApgiLiquid,
    ApgiNull,
    ApgiReference,
    PipelineModel,
)
from .reduced import ApgiReduced, ReducedParams, ReducedState, simulate_reduced

__all__ = [
    "APGIModel",
    "RUN_KEYS",
    "ParameterSpec",
    "REDUCED_PARAMETERS",
    "REDUCED_PARAMETER_NAMES",
    "reduced_parameter_spec",
    "ReducedParams",
    "ReducedState",
    "simulate_reduced",
    "ApgiReduced",
    "PipelineModel",
    "ApgiReference",
    "ApgiExtended",
    "ApgiHierarchical",
    "ApgiLiquid",
    "ApgiAgent",
    "ApgiNull",
]
