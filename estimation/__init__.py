"""Estimation package (S1-IMP-09): PyMC model, recovery, identifiability, profile likelihood.

Home of the estimation model per decision D2; ``apgi-framework`` keeps the
reference equations and fixtures. PyMC is optional (``apgi[estimation]``); every
numpy estimator here runs without it.
"""

from .composite import (
    COMPOSITE_NAME,
    CompositeEstimate,
    FactorsNotSeparableError,
    design_separates_factors,
    expose_factors,
    fit_composite,
)
from .fitting import PosteriorSummary, fit, fit_laplace, fit_map, fit_mcmc
from .gate_table import allow_list, build_gate_table, write_gate_table
from .likelihood import Observed, log_likelihood
from .profile_likelihood import ProfileResult, profile_all, profile_parameter
from .pymc_models import build_pymc_model, pymc_available, sample_pymc
from .recovery_simulation import RecoveryResult, evaluate_criteria, run_recovery
from .structural_identifiability import AnalysisSetup, run_k7a
from .synthetic_generator import (
    Design,
    SyntheticDataset,
    check_scoring_regression,
    generate_dataset,
    reversal_stimulus,
)

__all__ = [
    "COMPOSITE_NAME",
    "CompositeEstimate",
    "FactorsNotSeparableError",
    "design_separates_factors",
    "expose_factors",
    "fit_composite",
    "PosteriorSummary",
    "fit",
    "fit_laplace",
    "fit_map",
    "fit_mcmc",
    "allow_list",
    "build_gate_table",
    "write_gate_table",
    "Observed",
    "log_likelihood",
    "ProfileResult",
    "profile_all",
    "profile_parameter",
    "build_pymc_model",
    "pymc_available",
    "sample_pymc",
    "RecoveryResult",
    "evaluate_criteria",
    "run_recovery",
    "AnalysisSetup",
    "run_k7a",
    "Design",
    "SyntheticDataset",
    "check_scoring_regression",
    "generate_dataset",
    "reversal_stimulus",
]
