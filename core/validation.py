"""Comprehensive parameter validation for APGI configuration.
APGI Specification §15: Design Constraints
This module enforces all parameter constraints specified in the APGI specification
to prevent invalid configurations and ensure numerical stability.
Constraints enforced:
    - Neuromodulator separation (§2.3-2.4)
    - Parameter ranges (§15)
    - Numerical stability bounds
    - Integration step size constraints (§7.4)
    - Timescale hierarchy constraints (§8.1)
    - Precision bounds (§2.2)
All constraints include spec section references for traceability.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from core.deprecation import config_value

if TYPE_CHECKING:
    from core.logging_config import get_logger

    logger = get_logger("apgi.validation")


class ValidationError(ValueError):
    """Raised when configuration validation fails."""

    pass


#: Fallback defaults used when a config omits a key. These mirror
#: config.CONFIG and are mutually consistent — in particular
#: DEFAULT_DT <= min(DEFAULT_TAU_*) / 10, so a config that specifies none of
#: them validates cleanly (MathSpec §7.4).
DEFAULT_DT = 0.5
DEFAULT_TAU_S = 5.0
DEFAULT_TAU_THETA = 1000.0
DEFAULT_TAU_PI = 1000.0
#: Default base timescale for the hierarchical integrators. MathSpec §15
#: requires τ_ℓ > 1 (discrete integrator stability); τ_0 = 1.0 sits exactly on
#: the excluded boundary, so the default is lifted above it.
DEFAULT_TAU_0 = 10.0


class CanonicalRangeWarning(UserWarning):
    """Emitted when a parameter is inside its mathematical support but outside
    the Notation Appendix's canonical range.

    This is deliberately a warning rather than an error: values outside the
    canonical band are legitimate for sensitivity analysis and for modelling
    non-waking regimes (e.g. γ_sig → 0 under anaesthesia, which the spec
    describes as "below the waking band [2, 7.5], which is the operational
    band, not the mathematical support (0, ∞)"). Runs that trip these are
    still valid — they are simply not the canonical parameterisation, and the
    run manifest records that fact.
    """


def validate_config(config: dict) -> None:
    """Validate APGI configuration against all spec constraints.
    Performs comprehensive validation of all parameters. Raises ValidationError
    if any constraint is violated.
    Args:
        config: Configuration dictionary
    Raises:
        ValidationError: If any constraint is violated
    Examples:
        >>> config = {"lam": 0.2, "kappa": 0.15, "dt": 1.0}
        >>> validate_config(config)  # Passes
        >>> config = {"lam": 1.5}  # Invalid
        >>> validate_config(config)  # Raises ValidationError
    """
    # Validate each constraint category
    _validate_neuromodulator_separation(config)
    _validate_signal_accumulation(config)
    _validate_threshold_dynamics(config)
    _validate_ignition_dynamics(config)
    _validate_continuous_time_sde(config)
    _validate_hierarchical_parameters(config)
    _validate_precision_parameters(config)
    _validate_learning_rates(config)  # Spec §1.4
    _validate_ema_parameters(config)  # Spec §1.2
    _validate_sliding_window_params(config)  # Spec §1.2
    _validate_numerical_stability(config)
    # Validate post-ignition reset factor (§6)
    if "reset_factor" in config:
        validate_reset_factor(config["reset_factor"])
    # Validate neuromodulatory gains
    g_ach = config.get("g_ach", 1.0)
    g_ne = config.get("g_ne", 1.0)
    if g_ach < 0:
        raise ValidationError(f"g_ach={g_ach} must be ≥ 0. Spec §2.3: ACh gain")
    if g_ne < 0:
        raise ValidationError(f"g_ne={g_ne} must be ≥ 0. Spec §2.3: NE gain")
    # β_DA is an UNCONSTRAINED REAL (MathSpec glossary: "β_DA(t) | Dopaminergic
    # additive bias on interoceptive error | ℝ"). Negative values are not an
    # error — they model dopaminergic depletion, which the spec names as a
    # prediction ("a DA agonist must shift detection function gain"; the
    # depletion arm is its control). The previous `beta < 0` rejection made a
    # spec-sanctioned regime unreachable, so no non-negativity check applies.
    _ = config.get("beta", 1.15)


#: Spec-mandated ceiling on the product γ_NE · g_NE (MathSpec §2).
NE_SUBDOMINANCE_MAX = 0.5


def _validate_neuromodulator_separation(config: dict) -> None:
    """Validate neuromodulator coupling constraints (MathSpec §2, §15).

    The spec permits NE to act on precision AND threshold simultaneously,
    subject to a quantitative sub-dominance constraint:

        "when both Π_i^eff = g_NE · Π_i and θ ← θ(1 + γ_NE · g_NE) are active,
         we adopt the working constraint γ_NE · g_NE_max ≤ 0.5 to ensure the
         multiplicative threshold modulation remains sub-dominant"

    and §15 says NE "should affect precision OR threshold — not both **at
    maximum gain** simultaneously". The constraint is therefore on the
    *product*, not on the pair of booleans. A previous blanket ban on enabling
    both was stricter than the spec and directly contradicted config.py's own
    instruction to "Set ne_on_threshold=True AND lower gamma_ne (≤0.01) to
    enable both pathways" — that documented configuration was unusable.

    Violating the product is recoverable by the remedy the spec itself
    prescribes (rescale g_NE ← 0.5/γ_NE), which :meth:`APGIPipeline._enforce_ne_clamp`
    applies. Here it is reported as a warning so direct callers of
    ``validate_config`` still learn about it.
    """
    ne_on_precision = config.get("ne_on_precision", False)
    ne_on_threshold = config.get("ne_on_threshold", False)
    if not (ne_on_precision and ne_on_threshold):
        return
    gamma_ne = float(config.get("gamma_ne", 0.1))
    g_ne = float(config.get("g_ne", 1.0))
    product = gamma_ne * g_ne
    if product > NE_SUBDOMINANCE_MAX:
        warnings.warn(
            f"Both NE pathways are active with γ_NE·g_NE = {product:.3f} > "
            f"{NE_SUBDOMINANCE_MAX}. The spec mandates rescaling "
            f"g_NE ← {NE_SUBDOMINANCE_MAX}/γ_NE = "
            f"{NE_SUBDOMINANCE_MAX / gamma_ne:.3f} to keep multiplicative "
            "threshold modulation sub-dominant (MathSpec §2). APGIPipeline "
            "applies this clamp automatically.",
            RuntimeWarning,
            stacklevel=3,
        )


def _validate_signal_accumulation(config: dict) -> None:
    """Validate signal accumulation constraints (§3.2).
    Spec §3.2: Integration rate λ must be in (0, 1) for leaky integrator.
    """
    lam = config.get("lam", 0.2)
    if not (0 < lam < 1):
        raise ValidationError(
            f"lam must be in (0, 1), got {lam}. Spec §3.2: S(t+1) = (1-λ)S(t) + λ·S_inst(t)"
        )


def validate_reset_factor(reset_factor: float) -> None:
    """Validate reset factor ρ per APGI spec §6.
    Spec requirement: ρ ∈ (0, 1)
    - ρ close to 0: Full reset (strong refractory period)
    - ρ close to 1: Partial reset (mild inhibition)
    Args:
        reset_factor: Signal reset factor
    Raises:
        ValidationError: If ρ not in (0, 1)
    """
    if not (0.0 < reset_factor < 1.0):
        raise ValidationError(
            f"reset_factor must be in (0, 1), got {reset_factor}. "
            "Spec §6: Signal reset S ← ρ·S requires ρ ∈ (0, 1). "
            "ρ ≤ 0 causes signal to grow; ρ ≥ 1 prevents reset."
        )


def _validate_threshold_dynamics(config: dict) -> None:
    """Validate threshold dynamics constraints (§4, §6).
    Spec §4.5: Exponential decay rate κ must be > 0
    Spec §6.1: Reset factor ρ must be in (0, 1) - validated at step() time
    """
    from config import decay_exponent

    exponent = decay_exponent({"tau_theta": 20.0, "dt": 0.01, **config})
    if exponent <= 0:
        raise ValidationError(
            f"decay exponent dt/tau_theta must be > 0, got {exponent}. "
            "Spec Eq. 32: θ(t+1) = θ_base + (θ(t) − θ_base)·exp(−γ_θ·dt)"
        )


def _validate_ignition_dynamics(config: dict) -> None:
    """Validate ignition mechanism constraints (§5.2).
    Spec §5.2: Sigmoid temperature τ_σ must be > 0
    """
    tau_sigma = config.get("ignite_tau", 0.5)
    if tau_sigma <= 0:
        raise ValidationError(
            f"ignite_tau (τ_σ) must be > 0, got {tau_sigma}. Spec §5.2: P_ign = σ([S-θ]/τ_σ)"
        )


def _validate_continuous_time_sde(config: dict) -> None:
    """Validate continuous-time SDE constraints (§7.4).
    Spec §7.4: Euler-Maruyama integration requires:
        dt ≤ min(τ_S, τ_θ, τ_Π) / 10
    This ensures numerical stability of the SDE integration.
    """
    # Fallback defaults MUST mirror config.CONFIG and MUST satisfy this
    # function's own constraint. Previously dt defaulted to 1.0 while tau_s
    # defaulted to 5.0, so a config specifying neither was judged invalid
    # against defaults it never chose (max_dt = 5.0/10 = 0.5 < 1.0). Under the
    # old fail-open behaviour that produced a spurious warning; under
    # fail-closed it would reject valid configurations outright.
    dt = config.get("dt", DEFAULT_DT)
    tau_s = config.get("tau_s", DEFAULT_TAU_S)
    tau_theta = config.get("tau_theta", DEFAULT_TAU_THETA)
    tau_pi = config.get("tau_pi", DEFAULT_TAU_PI)
    if dt <= 0:
        raise ValidationError(
            f"dt must be > 0, got {dt}. Spec §7.4: Euler-Maruyama integration step"
        )
    min_tau = min(tau_s, tau_theta, tau_pi)
    max_dt = min_tau / 10.0
    if dt > max_dt:
        raise ValidationError(
            f"dt={dt} exceeds max {max_dt:.4f}. "
            f"Spec §7.4 requires dt ≤ min(τ_S, τ_θ, τ_Π) / 10. "
            f"(τ_S={tau_s}, τ_θ={tau_theta}, τ_Π={tau_pi})"
        )


def _validate_hierarchical_parameters(config: dict) -> None:
    """Validate hierarchical architecture constraints (§8.1).
    Spec §8.1: Timescale ratio k must be > 1
    Spec §8.2: All timescales τ_ℓ must be > 1
    """
    if config.get("use_hierarchical", False):
        k = config.get("timescale_k", 1.6)
        if k <= 1:
            raise ValidationError(
                f"timescale_k must be > 1, got {k}. Spec §8.1: τ_ℓ = τ_0·k^ℓ requires k > 1"
            )
        # Validate that all timescales are > 1. The default mirrors
        # config.CONFIG's tau_0=10.0; the previous fallback of 1.0 sat exactly
        # on the excluded boundary and so failed this very check.
        tau_0 = config.get("tau_0", DEFAULT_TAU_0)
        n_levels = config.get("n_levels", 3)
        for level in range(n_levels):
            tau_ell = tau_0 * (k**level)
            if tau_ell <= 1:
                raise ValidationError(
                    f"τ_{level} = {tau_ell:.2f} ≤ 1. Spec §8.2: All timescales must be > 1"
                )


def _validate_precision_parameters(config: dict) -> None:
    """Validate precision system constraints (§2.2).
    Spec §2.2: Precision bounds Π_min < Π_max
    Spec §2.2: Clamping prevents numerical issues
    """
    pi_min = config.get("pi_min", 0.01)
    pi_max = config.get("pi_max", 10.0)
    if pi_min <= 0:
        raise ValidationError(f"pi_min must be > 0, got {pi_min}. Spec §2.2: Precision lower bound")
    if pi_max <= pi_min:
        raise ValidationError(
            f"pi_max must be > pi_min, got pi_max={pi_max}, pi_min={pi_min}. "
            "Spec §2.2: Precision bounds"
        )
    # Reasonable range check - warn in production
    if pi_max / pi_min > 1e8:
        warnings.warn(
            f"Precision range very large: pi_max/pi_min = {pi_max / pi_min:.1e}. "
            "May cause numerical issues. Spec §2.2 recommends (0.01, 10.0). "
            "Use strict_mode=True to enforce strict validation.",
            RuntimeWarning,
            stacklevel=2,
        )


def _validate_learning_rates(config: dict) -> None:
    """Validate generative model learning rates (Spec §1.4).
    Spec §1.4: Learning rates η_pred,e, η_pred,i must satisfy η_pred < 2/Π_max
    for gradient descent convergence in the generative model.
    """
    if not config.get("use_internal_predictions", False):
        return  # Skip if generative model disabled
    eta_pred_e = config_value(config, "eta_pred_e", "kappa_e", 0.01)
    eta_pred_i = config_value(config, "eta_pred_i", "kappa_i", 0.01)
    pi_max = config.get("pi_max", 10.0)
    max_eta = 2.0 / pi_max
    if eta_pred_e >= max_eta:
        raise ValidationError(
            f"eta_pred_e={eta_pred_e} >= {max_eta:.4f}. "
            f"Spec §1.4 requires η_pred,e < 2/Π_max for generative model stability. "
            f"Current Π_max={pi_max}, max allowed η_pred,e={max_eta:.4f}"
        )
    if eta_pred_i >= max_eta:
        raise ValidationError(
            f"eta_pred_i={eta_pred_i} >= {max_eta:.4f}. "
            f"Spec §1.4 requires η_pred,i < 2/Π_max for generative model stability. "
            f"Current Π_max={pi_max}, max allowed η_pred,i={max_eta:.4f}"
        )


def _validate_ema_parameters(config: dict) -> None:
    """Validate EMA variance estimation parameters (Spec §1.2).
    Spec §1.2: EMA smoothing parameters α must be in (0, 1].
    """
    variance_method = config.get("variance_method", "ema")
    if variance_method == "ema":
        alpha_e = config.get("alpha_e", 0.05)
        alpha_i = config.get("alpha_i", 0.05)
        if not (0 < alpha_e <= 1):
            raise ValidationError(f"alpha_e={alpha_e} must be in (0, 1] for EMA. Spec §1.2")
        if not (0 < alpha_i <= 1):
            raise ValidationError(f"alpha_i={alpha_i} must be in (0, 1] for EMA. Spec §1.2")


def _validate_sliding_window_params(config: dict) -> None:
    """Validate sliding window variance estimation parameters.
    T_win must be positive integer for sliding window method.
    """
    variance_method = config.get("variance_method", "ema")
    if variance_method == "sliding_window":
        T_win = config.get("T_win", 50)
        if not isinstance(T_win, int) or T_win <= 0:
            raise ValidationError(
                f"T_win={T_win} must be positive integer for sliding window. Spec §1.2 (Method B)"
            )
        # Warn if window is too small for meaningful Bessel correction
        if T_win < 5:
            warnings.warn(
                f"T_win={T_win} is very small. Bessel correction may be unstable. "
                f"Recommend T_win >= 10. Spec §1.2",
                RuntimeWarning,
                stacklevel=2,
            )


def _validate_numerical_stability(config: dict) -> None:
    """Validate general numerical stability constraints.
    Checks for reasonable parameter ranges and potential numerical issues.
    """
    eps = config.get("eps", 1e-8)
    if eps <= 0 or eps >= 1:
        raise ValidationError(f"eps must be in (0, 1), got {eps}. Numerical stability threshold")
    # Check for reasonable learning rates
    eta = config.get("eta", 0.1)
    if eta <= 0 or eta > 1:
        raise ValidationError(
            f"eta must be in (0, 1], got {eta}. Spec §4.1: Threshold learning rate"
        )
    # Check for reasonable noise levels
    noise_std = config.get("noise_std", 0.01)
    if noise_std < 0:
        raise ValidationError(
            f"noise_std must be ≥ 0, got {noise_std}. Spec §7.2: SDE diffusion coefficient"
        )
    # Validate cost-value coefficients
    c0 = config.get("c0", 0.0)
    c1 = config.get("c1", 0.2)
    c2 = config.get("c2", 0.5)
    v1 = config.get("v1", 0.5)
    v2 = config.get("v2", 0.5)
    for name, value in [("c0", c0), ("c1", c1), ("c2", c2), ("v1", v1), ("v2", v2)]:
        if value < 0:
            raise ValidationError(f"{name}={value} must be ≥ 0. Spec §4.2-4.3: Cost-value model")
    # Validate refractory boost
    delta = config.get("delta", 0.5)
    if delta < 0:
        raise ValidationError(f"delta={delta} must be ≥ 0. Spec §6: Refractory boost")
    # Validate log nonlinearity toggle
    signal_log_nonlinearity = config.get("signal_log_nonlinearity", True)
    if not isinstance(signal_log_nonlinearity, bool):
        raise ValidationError(
            f"signal_log_nonlinearity must be boolean, got {type(signal_log_nonlinearity)}. "
            f"Spec §3.3: Optional log compression"
        )


def validate_parameter(name: str, value: Any, constraint: str, spec_section: str = "") -> None:
    """Validate a single parameter against a constraint.
    Args:
        name: Parameter name
        value: Parameter value
        constraint: Constraint description (e.g., "> 0", "in (0, 1)")
        spec_section: Spec section reference (e.g., "§3.2")
    Raises:
        ValidationError: If constraint violated
    Examples:
        >>> validate_parameter("lam", 0.2, "in (0, 1)", "§3.2")
        >>> validate_parameter("lam", 1.5, "in (0, 1)", "§3.2")  # Raises
    """
    # Parse constraint
    if constraint.startswith("in ("):
        # Parse "in (a, b)" format
        parts = constraint[4:-1].split(",")
        a, b = float(parts[0].strip()), float(parts[1].strip())
        if not (a < value < b):
            raise ValidationError(f"{name}={value} not {constraint}. Spec {spec_section}")
    elif constraint.startswith(">="):
        # Parse ">= a" format (must check before ">")
        threshold = float(constraint[2:].strip())
        if value < threshold:
            raise ValidationError(f"{name}={value} not {constraint}. Spec {spec_section}")
    elif constraint.startswith(">"):
        # Parse "> a" format
        threshold = float(constraint[1:].strip())
        if value <= threshold:
            raise ValidationError(f"{name}={value} not {constraint}. Spec {spec_section}")
    elif constraint.startswith("<="):
        # Parse "<= a" format (must check before "<")
        threshold = float(constraint[2:].strip())
        if value > threshold:
            raise ValidationError(f"{name}={value} not {constraint}. Spec {spec_section}")
    elif constraint.startswith("<"):
        # Parse "< a" format
        threshold = float(constraint[1:].strip())
        if value >= threshold:
            raise ValidationError(f"{name}={value} not {constraint}. Spec {spec_section}")


def get_constraint_summary() -> dict:
    """Get summary of all constraints for documentation.
    Returns:
        Dictionary mapping constraint categories to lists of constraints
    Examples:
        >>> summary = get_constraint_summary()
        >>> print(summary["Signal Accumulation"])
        ['lam in (0, 1)']
    """
    return {
        "Neuromodulator Separation": [
            "NE cannot modulate both precision and threshold (§2.3-2.4)",
        ],
        "Signal Accumulation": [
            "lam in (0, 1) (§3.2)",
        ],
        "Threshold Dynamics": [
            "kappa > 0 (§4.5)",
            "reset_factor in (0, 1) (§6.1)",
        ],
        "Ignition Mechanism": [
            "ignite_tau > 0 (§5.2)",
        ],
        "Continuous-Time SDE": [
            "dt > 0 (§7.4)",
            "dt ≤ min(τ_S, τ_θ, τ_Π) / 10 (§7.4)",
        ],
        "Hierarchical Architecture": [
            "timescale_k > 1 (§8.1)",
            "All τ_ℓ > 1 (§8.2)",
        ],
        "Precision System": [
            "pi_min > 0 (§2.2)",
            "pi_max > pi_min (§2.2)",
        ],
        "Numerical Stability": [
            "eps in (0, 1)",
            "eta in (0, 1]",
            "noise_std ≥ 0",
        ],
    }


def format_constraint_summary() -> str:
    """Format human-readable summary of all constraints."""
    summary = get_constraint_summary()
    lines = [
        "",
        "=" * 70,
        "APGI PARAMETER CONSTRAINTS (Spec §15)",
        "=" * 70,
    ]
    for category, constraints in summary.items():
        lines.append(f"\n{category}:")
        for constraint in constraints:
            lines.append(f"  • {constraint}")
    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def print_constraint_summary() -> None:
    """Print human-readable summary of all constraints to stdout.
    Examples:
        >>> print_constraint_summary()  # Prints constraint summary
    """
    print(format_constraint_summary())


# ---------------------------------------------------------------------------
# Canonical Parameter Ranges (Notation Appendix, "Canonical parameters
# table") — a REPORT-style compliance checker, distinct from validate_config()
# above.
#
# validate_config() enforces numerical-stability constraints (signs, bounds
# that keep the ODE/SDE integrators well-behaved) and MUST pass for every
# config the pipeline runs with — including this repo's lightweight/demo
# defaults (theta_0=1.0, pi_min=0.01, k=1.6, ...), which are deliberately
# NOT spec-canonical (see config.py's own documentation of this choice).
# Turning the Notation Appendix's canonical *scientific* ranges below into
# hard errors would make the shipped defaults unusable.
#
# check_canonical_parameter_ranges() instead reports compliance against
# those canonical ranges without raising, so they are checkable/testable
# (per the Notation Appendix's Canonical Parameters table) without breaking
# any existing configuration.
#
# IMPORTANT — symbol collision warning: this repo's config["kappa"] is the
# threshold EXPONENTIAL DECAY RATE (dimensionless, roughly (0,1)), NOT the
# ATP/bit metabolic bridge parameter κ the Notation Appendix table means by
# "kappa ≈ 100 (10-1000 ATP/bit)". That κ lives separately, e.g. in
# energy.thermodynamics.ThermodynamicTracker.kappa or as an argument to
# metabolic_cost(). This checker therefore takes each canonical quantity as
# an EXPLICIT keyword argument rather than guessing from a raw config dict,
# to avoid silently checking the wrong "kappa" against the wrong range.
# ---------------------------------------------------------------------------

CANONICAL_PARAMETER_RANGES: dict[str, tuple[float, float]] = {
    "gamma_sig": (2.0, 7.5),
    "theta_0": (0.25, 0.85),
    "pi_min": (0.1, 10.0),
    "pi_max": (0.1, 10.0),
    "beta_sm": (0.1, 1.2),
    "beta_somatic_linear": (0.5, 2.5),
    "c_theta": (0.001, 0.5),
    "kappa_atp_per_bit": (10.0, 1000.0),
    "lambda_allo": (0.001, 0.05),
    "rho_res": (0.7, 0.95),
    "k_inter_level": (50.0, 680.0),  # k≈133, Δ∈[1.7,2.8] sensitivity band
}


def check_canonical_parameter_ranges(**values: float | None) -> dict[str, Any]:
    """Report compliance of given values against the Notation Appendix's
    Canonical Parameters table. Does not raise — returns a per-parameter
    report so canonical-range compliance is checkable/testable.
    Only parameters actually passed as keyword arguments (non-None) are
    checked; recognized names are the keys of CANONICAL_PARAMETER_RANGES.
    Args:
        **values: Any subset of CANONICAL_PARAMETER_RANGES' keys, e.g.
            check_canonical_parameter_ranges(gamma_sig=1.0/config["ignite_tau"],
                                              theta_0=config["theta_base"])
    Returns:
        Dictionary with a per-parameter "results" report and an overall
        "all_in_range" boolean (True only if every CHECKED parameter is
        within its canonical range; parameters not passed are not counted
        either way).
    Raises:
        ValueError: If a keyword argument name is not a recognized
            canonical parameter.
    """
    unknown = set(values) - set(CANONICAL_PARAMETER_RANGES)
    if unknown:
        raise ValueError(
            f"unknown canonical parameter(s): {sorted(unknown)}. "
            f"Recognized: {sorted(CANONICAL_PARAMETER_RANGES)}"
        )
    results: dict[str, dict[str, Any]] = {}
    for name, value in values.items():
        if value is None:
            continue
        lo, hi = CANONICAL_PARAMETER_RANGES[name]
        in_range = lo <= value <= hi
        results[name] = {
            "value": value,
            "canonical_range": (lo, hi),
            "in_range": in_range,
        }
    all_in_range = all(r["in_range"] for r in results.values()) if results else True
    return {"results": results, "all_in_range": all_in_range}


def check_canonical_ranges(config: dict) -> list[str]:
    """Return human-readable advisories for canonical-range departures.

    Complements :func:`validate_config`, which enforces the *mathematical
    support* of each parameter (violations are errors). This function checks
    the *canonical operating band* from the Notation Appendix's Canonical
    Parameters table — departures are legitimate but noteworthy, so they are
    returned as messages for the caller to warn on and record in the run
    manifest rather than raised.

    Args:
        config: Pipeline configuration dictionary.

    Returns:
        List of advisory messages; empty when every checkable parameter sits
        inside its canonical band.
    """
    report = check_canonical_compliance_for_config(config)
    messages: list[str] = []
    for name, entry in report["results"].items():
        if entry["in_range"]:
            continue
        lo, hi = entry["canonical_range"]
        messages.append(
            f"{name}={entry['value']:.4g} is outside the canonical range "
            f"[{lo:g}, {hi:g}] (Notation Appendix, Canonical Parameters table). "
            "This is permitted — the canonical band is the operational range, "
            "not the mathematical support — but results should state the "
            "departure explicitly."
        )
    return messages


def check_canonical_compliance_for_config(config: dict) -> dict[str, Any]:
    """Convenience wrapper: extract the UNAMBIGUOUS canonical parameters
    from a pipeline config dict and check them against
    CANONICAL_PARAMETER_RANGES.
    Only maps config keys with a one-to-one, unambiguous correspondence to a
    Notation Appendix canonical symbol:
        gamma_sig            <- 1 / config["ignite_tau"]  (γ_sig = 1/τ_σ)
        theta_0               <- config["theta_base"]
        pi_min, pi_max         <- config["pi_min"], config["pi_max"]
        beta_sm                <- config["beta_sm"] (if present)
    Deliberately excluded (symbol collisions or no direct representation in
    this codebase — see module docstring above): kappa (config["kappa"] is
    the threshold decay rate, not ATP/bit), c_theta, lambda_allo, rho_res
    (lives on the reservoir object, not in the top-level pipeline config),
    k_inter_level (config["k"] is a lightweight/demo value by default; see
    hierarchy.coupling.K_INTER_LEVEL_CANONICAL for the canonical constant).
    """
    ignite_tau = config.get("ignite_tau", config.get("tau_sigma"))
    gamma_sig = (1.0 / ignite_tau) if ignite_tau else None
    return check_canonical_parameter_ranges(
        gamma_sig=gamma_sig,
        theta_0=config.get("theta_base"),
        pi_min=config.get("pi_min"),
        pi_max=config.get("pi_max"),
        beta_sm=config.get("beta_sm"),
    )
