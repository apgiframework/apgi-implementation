"""Computation engine primitives (S1-IMP-01..08)."""

from .counterfactual import (
    Condition,
    ConditionResult,
    baseline,
    contrast,
    dopamine_blockade,
    precision_lesion,
    run_conditions,
    stimulus_gain,
)
from .ignition_sim import (
    IgnitionSummary,
    hard_ignition,
    hard_limit_gap,
    ignition_probability,
    sample_ignition,
    simulate_session,
    summarise_ignitions,
    threshold_sweep,
)
from .latent_state import FilterResult, filter_latent_signal
from .prediction_error import (
    PrecisionWeightedErrorChannel,
    PredictionErrorResult,
    precision_weighted_error,
    precision_weighted_errors,
)
from .provenance import RunRecord, array_digest, load_record, record_run, replay_run, save_record
from .somatic import SomaticThresholdAdapter, clamp_ne_gain, interoceptive_precision, somatic_gain

__all__ = [
    "Condition",
    "ConditionResult",
    "baseline",
    "contrast",
    "dopamine_blockade",
    "precision_lesion",
    "run_conditions",
    "stimulus_gain",
    "IgnitionSummary",
    "hard_ignition",
    "hard_limit_gap",
    "ignition_probability",
    "sample_ignition",
    "simulate_session",
    "summarise_ignitions",
    "threshold_sweep",
    "FilterResult",
    "filter_latent_signal",
    "PrecisionWeightedErrorChannel",
    "PredictionErrorResult",
    "precision_weighted_error",
    "precision_weighted_errors",
    "RunRecord",
    "array_digest",
    "load_record",
    "record_run",
    "replay_run",
    "save_record",
    "SomaticThresholdAdapter",
    "clamp_ne_gain",
    "interoceptive_precision",
    "somatic_gain",
]
