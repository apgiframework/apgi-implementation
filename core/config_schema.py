"""Pydantic configuration schema for type-safe APGI configuration.
Provides strict validation with clear error messages for production deployments.
ROLE — this schema is a distinct, OPTIONAL validation path from the one
`APGIPipeline` actually runs at construction time. `APGIPipeline.__init__`
calls `core.validation.validate_config()` (a plain-dict validator) directly;
`APGIConfig` is not wired into the pipeline and must be invoked explicitly
by callers who want Pydantic-style strict validation (e.g. before writing a
config to disk, building a UI form, or validating user-supplied JSON) ahead
of constructing a pipeline. Kept as a complete field-for-field mirror of
config.py's CONFIG dict so it does not silently pass configs that are
missing keys the pipeline actually reads.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from core.deprecation import warn_renamed


class APGIConfig(BaseModel):
    """Production-grade APGI configuration with strict validation.
    All parameters validated at initialization with clear error messages.
    No auto-adjustments - invalid configurations raise ValidationError.
    """

    # Initial states
    S0: float = Field(default=0.0, description="Initial signal value")
    theta_0: float = Field(default=1.0, gt=0, description="Initial threshold")
    theta_base: float = Field(default=1.0, gt=0, description="Base threshold")
    sigma2_e0: float = Field(default=1.0, gt=0, description="Initial exteroceptive variance")
    sigma2_i0: float = Field(default=1.0, gt=0, description="Initial interoceptive variance")
    # Numerical stability
    eps: float = Field(default=1e-8, gt=0, lt=1, description="Numerical stability threshold")
    pi_min: float = Field(default=0.01, gt=0, description="Minimum precision")
    pi_max: float = Field(default=10.0, gt=0, description="Maximum precision")
    # EMA variance update
    alpha_e: float = Field(default=0.05, gt=0, lt=1, description="Exteroceptive EMA rate")
    alpha_i: float = Field(default=0.05, gt=0, lt=1, description="Interoceptive EMA rate")
    # Variance estimation method
    variance_method: Literal["ema", "sliding_window"] = Field(
        default="ema", description="Variance estimation method"
    )
    T_win: int = Field(default=50, gt=0, description="Sliding window size")
    # Neuromodulation
    g_ach: float = Field(default=1.0, ge=0, description="Acetylcholine gain")
    g_ne: float = Field(default=1.0, ge=0, description="Norepinephrine gain")
    # Dopaminergic bias
    beta: float = Field(default=1.15, ge=0, description="Dopaminergic bias")
    beta_da: float | None = Field(
        default=None, description="Alias for beta (backward compatibility)"
    )
    # NE double counting prevention
    ne_on_precision: bool = Field(default=True, description="NE modulates precision")
    ne_on_threshold: bool = Field(default=False, description="NE modulates threshold")
    gamma_ne: float = Field(default=0.1, ge=0, le=1, description="NE modulation strength")
    # Signal accumulation
    lam: float = Field(
        default=0.0327838995179941,
        gt=0,
        lt=1,
        description="Signal integration rate, 1 − exp(−dt/τ_S)",
    )
    signal_log_nonlinearity: bool = Field(default=True, description="Enable log stabilization")
    use_canonical_discrete_mode: bool = Field(
        default=False, description="Use discrete accumulation instead of ODE"
    )
    # Threshold update + refractory dynamics
    eta: float = Field(
        default=0.002, gt=0, le=1, description="Threshold learning rate per step (rate × dt)"
    )
    delta: float = Field(default=0.5, ge=0, description="Refractory boost")
    # Post-ignition signal reset
    reset_factor: float = Field(default=0.1, gt=0, lt=1, description="Signal reset factor (rho)")
    # Threshold adaptation timescales
    tau_theta: float = Field(
        default=20.0, gt=0, description="Threshold adaptation timescale (allostatic)"
    )
    tau_theta_recovery: float = Field(
        default=0.45, gt=0, description="Threshold recovery timescale (perceptual)"
    )
    # Hierarchical cascade tuning
    KAPPA_UP: float = Field(default=0.1, ge=0, le=1, description="Bottom-up cascade strength")
    KAPPA_DOWN: float = Field(default=0.1, ge=0, le=1, description="Top-down coupling strength")
    # Cost-value model
    use_realistic_cost: bool = Field(default=True, description="Use realistic metabolic cost")
    c0: float = Field(default=0.0, ge=0, description="Base metabolic cost")
    c1: float = Field(default=0.2, ge=0, description="Signal-dependent cost coefficient")
    c2: float = Field(default=0.5, ge=0, description="Ignition cost coefficient")
    v1: float = Field(default=0.5, ge=0, description="Exteroceptive value weight")
    v2: float = Field(default=0.5, ge=0, description="Interoceptive value weight")
    # Ignition dynamics
    ignite_tau: float = Field(default=0.5, gt=0, description="Ignition sigmoid temperature")
    tau_sigma: float | None = Field(
        default=None, description="Alias for ignite_tau (backward compatibility)"
    )
    stochastic_ignition: bool = Field(default=False, description="Enable stochastic ignition")
    # Continuous-time parameters
    tau_s: float = Field(default=0.3, gt=0, description="Signal decay time constant (s)")
    dt: float = Field(default=0.01, gt=0, description="Integration time step (s); dt₀ = 10 ms")
    r_max: float = Field(
        default=100.0, gt=0, description="Maximal ignition rate of the hazard (1/s)"
    )
    noise_std: float = Field(default=0.01, ge=0, description="SDE diffusion coefficient")
    # Generative model dynamics
    use_internal_predictions: bool = Field(default=True, description="Enable internal predictions")
    eta_pred_e: float = Field(
        default=0.01, ge=0, description="Exteroceptive prediction learning rate η_pred,e"
    )
    eta_pred_i: float = Field(
        default=0.01, ge=0, description="Interoceptive prediction learning rate η_pred,i"
    )
    # Multi-scale parameters
    # NOTE: k has two distinct spec meanings that must not be conflated
    # (Notation Appendix, Cross-Series Notation Consistency table):
    #   - Canonical inter-level coarse-graining ratio ≈133 (Δ≈2.1 log-decades
    #     per level), spanning τ_min≈100ms to τ_max≈1yr across ~5 levels.
    #   - Hasson et al. (2008) ~1.3-2.0 is the *within-level* adjacent
    #     cortical-region gradient ratio — a distinct quantity that must
    #     never be substituted for the inter-level ratio (doing so yields
    #     L≈49 instead of L≈5 in the level-count formula).
    # The upper bound here previously capped this field at <3, which made
    # the spec-mandated ≈133 value impossible to configure. Bounded only by
    # sane numerical limits now; see hierarchy.coupling.K_INTER_LEVEL_CANONICAL
    # for the canonical constant and estimate_hierarchy_levels() for the
    # level-count formula that uses it.
    timescale_k: float = Field(default=1.6, gt=1, le=1000, description="Timescale expansion factor")
    # Thermodynamic constraints
    use_thermodynamic_cost: bool = Field(default=False, description="Enable Landauer cost")
    k_boltzmann: float = Field(
        default=1.380649e-23, gt=0, description="Boltzmann constant (J/K), CODATA exact value"
    )
    T_env: float = Field(default=310.0, gt=0, description="Environmental temperature (K)")
    r_l: float = Field(
        default=1e20,
        gt=0,
        description="Erasure-overhead factor on the Landauer floor (formerly kappa_meta; absorbs J→AU)",
    )
    r_l_units: Literal["dimensionless", "joules_per_bit"] = Field(
        default="dimensionless", description="Metabolic cost units"
    )
    # BOLD calibration
    use_bold_calibration: bool = Field(default=False, description="Enable BOLD calibration")
    bold_conversion_factor: float = Field(
        default=1.2e-18, gt=0, description="Joules per 1% BOLD change per cm³"
    )
    bold_tissue_volume: float = Field(default=1.0, gt=0, description="Tissue volume in cm³")
    bold_ignition_spike_factor: float = Field(
        default=1.075, gt=0, description="Energy spike factor during ignition"
    )
    # Reservoir layer
    use_reservoir: bool = Field(default=False, description="Enable reservoir computing")
    reservoir_size: int = Field(default=100, gt=0, description="Reservoir units")
    reservoir_tau: float = Field(default=1.0, gt=0, description="Reservoir time constant")
    reservoir_spectral_radius: float = Field(
        default=0.9, gt=0, lt=1, description="Reservoir spectral radius"
    )
    reservoir_input_scale: float = Field(default=0.1, gt=0, description="Input scaling")
    reservoir_readout_method: Literal["linear", "energy"] = Field(
        default="linear", description="Reservoir readout method"
    )
    reservoir_amplification: float = Field(
        default=0.0, ge=0, description="Suprathreshold amplification strength"
    )
    # Kuramoto oscillators
    use_kuramoto: bool = Field(default=False, description="Enable Kuramoto oscillators")
    kuramoto_tau_xi: float = Field(default=1.0, gt=0, description="OU noise correlation timescale")
    kuramoto_sigma_xi: float = Field(default=0.1, ge=0, description="OU noise amplitude")
    kuramoto_reset_amount: float | None = Field(
        default=None,
        description="Legacy additive phase reset on ignition (radians). "
        "None (default) selects the canonical zero-phase reset via "
        "theta_coupling; a positive float restores the legacy additive "
        "behavior.",
    )
    # Hierarchical mode
    hierarchical_mode: Literal["off", "basic", "advanced", "full"] | None = Field(
        default=None, description="Hierarchical mode preset"
    )
    use_hierarchical: bool = Field(default=False, description="Enable hierarchical processing")
    use_hierarchical_precision_ode: bool = Field(default=False, description="Enable precision ODE")
    use_phase_modulation: bool = Field(default=False, description="Enable phase modulation")
    n_levels: int = Field(default=3, ge=1, le=10, description="Hierarchy levels")
    # Observable mapping
    use_observable_mapping: bool = Field(
        default=False, description="Enable neural/behavioral observable extraction"
    )
    # Stability analysis
    use_stability_analysis: bool = Field(default=False, description="Enable stability analysis")

    # --- Fields below were previously missing from this schema even though
    # they are live keys in config.py's CONFIG dict — this schema was an
    # incomplete snapshot that would silently validate configs missing
    # parameters the pipeline actually reads. Added for completeness so
    # APGIConfig is an accurate (optional, strict-validation) mirror of the
    # runtime config surface. ---

    # Signed nonlinear error transform φ(ε) (§6)
    alpha_plus: float = Field(default=1.0, ge=0.5, le=2.0, description="Reward/approach gain α+")
    alpha_minus: float = Field(default=1.0, ge=0.5, le=2.0, description="Threat/avoidance gain α-")
    gamma_plus: float = Field(default=2.0, ge=1.0, le=5.0, description="Reward saturation slope γ+")
    gamma_minus: float = Field(
        default=2.0, ge=1.0, le=5.0, description="Avoidance saturation slope γ-"
    )
    # Serotonin
    beta_5ht: float = Field(default=0.0, description="5-HT patience/threshold offset")
    # Somatic marker (β_SM, exponential gain — distinct from β_DA and the
    # linear-form β_somatic; see Notation Appendix "β RANGE RECONCILIATION")
    beta_sm: float = Field(default=0.3, ge=0.1, le=1.2, description="Somatic-marker gain β_SM")
    use_somatic_precision: bool = Field(
        default=False, description="Enable exponential somatic precision modulation"
    )
    M_somatic: float = Field(
        default=0.0, ge=-2.0, le=2.0, description="Initial somatic marker M(c,a)"
    )
    # Post-ignition reset (uppercase spec-mirroring aliases)
    RHO_RETAIN: float = Field(default=0.1, gt=0, lt=1, description="Signal retention fraction ρ")
    DELTA_RESET: float = Field(default=0.5, ge=0, description="Post-ignition threshold boost")
    # Precision coupling ODE (hierarchical)
    tau_pi: float = Field(default=1000.0, gt=0, description="Precision decay time constant (s)")
    C_down: float = Field(default=0.1, ge=0, description="Top-down precision coupling strength")
    C_up: float = Field(default=0.05, ge=0, description="Bottom-up precision coupling strength")
    psi_type: Literal["identity", "tanh", "softsign"] = Field(
        default="identity", description="Bottom-up nonlinear transfer ψ(ε) type"
    )
    gamma_psi: float = Field(default=2.0, ge=1.0, le=5.0, description="Gain of ψ(ε)=tanh(γ_ψ·ε)")
    # Hierarchical timescale configuration
    tau_0: float = Field(default=10.0, gt=0, description="Base timescale for index-0 level (ms)")
    k: float = Field(default=1.6, gt=1, description="Timescale ratio between levels")
    hierarchical_timescale_mode: Literal["custom", "canonical"] = Field(
        default="custom", description="Timescale derivation mode"
    )
    hierarchy_tau_min: float = Field(
        default=100.0, gt=0, description="Canonical-mode tau_min (ms, spec Level 1)"
    )
    hierarchy_tau_max: float = Field(
        default=3.0e10, gt=0, description="Canonical-mode tau_max (ms, spec Level 4)"
    )
    k_canonical: float = Field(
        default=133.0, gt=1, description="Canonical inter-level coarse-graining ratio"
    )
    # Cross-Level Threshold Resonance (Russian Doll architecture)
    use_resonance: bool = Field(default=True, description="Enable per-level resonance system")
    resonance_kappa_down: float = Field(default=0.1, ge=0, description="Top-down PAC coupling")
    resonance_kappa_up: float = Field(default=0.0, ge=0, description="Bottom-up suppression")
    resonance_phi_noise_std: float = Field(default=0.0, ge=0, description="Per-step phase jitter")
    # Reservoir extras
    reservoir_weight: float = Field(default=0.1, ge=0, description="w_res in S_global formula")
    reservoir_ridge_alpha: float = Field(
        default=1e-6, gt=0, description="Ridge regression regularization"
    )
    reservoir_replaces_signal: bool = Field(
        default=False, description="MathSpec step-15 replacement mode vs additive add-on mode"
    )
    reservoir_as_threshold: bool = Field(
        default=False, description="Use reservoir as the threshold source (Mode B, spec §10)"
    )
    # Active Inference Action Loop (§19)
    use_active_inference: bool = Field(default=False, description="Enable perception-action loop")
    ai_on_ignition_only: bool = Field(default=True, description="Fire only on B_t=1")
    ai_n_actions: int = Field(default=3, gt=0, description="Number of candidate policies K")
    ai_policy_precision: float = Field(
        default=2.0, gt=0, description="Boltzmann sharpness γ_policy"
    )
    ai_w_epistemic: float = Field(default=1.0, ge=0, description="Weight on epistemic term")
    ai_w_pragmatic: float = Field(default=1.0, ge=0, description="Weight on pragmatic term")
    ai_w_metabolic: float = Field(default=0.5, ge=0, description="Weight on metabolic cost term")
    ai_sensory_feedback_rate: float = Field(default=0.1, ge=0, description="Channel-1 scale")
    ai_metabolic_feedback_rate: float = Field(default=0.1, ge=0, description="Channel-2 scale")
    ai_precision_update_rate: float = Field(default=0.05, ge=0, description="Channel-3 scale")
    ai_action_params: list[list[float]] | None = Field(
        default=None, description="Custom action parameter table (K, 3), or None for default"
    )
    # Metabolic state M(t) — distinct from the somatic marker M(c,a) above
    metabolic_state0: float = Field(default=0.5, ge=0, le=1, description="Initial metabolic state")
    c_metabolic_state: float = Field(
        default=0.0, ge=0, description="C(t) coefficient on metabolic_state(t)"
    )
    # Circadian/ultradian threshold modulation (§27, peripheral)
    use_circadian_modulation: bool = Field(default=False, description="Enable circadian rhythm")
    circadian_t0: float = Field(default=0.0, description="Initial circadian clock time (s)")
    circadian_A_circ: float = Field(default=0.1, ge=0, description="Circadian amplitude")
    circadian_T_circ: float = Field(default=86400.0, gt=0, description="Circadian period (s)")
    circadian_phi_circ: float = Field(default=0.0, description="Circadian phase offset (rad)")
    circadian_A_ultradian: float = Field(default=0.05, ge=0, description="Ultradian amplitude")
    circadian_T_ultradian: float = Field(default=5400.0, gt=0, description="Ultradian period (s)")
    circadian_phi_ultradian: float = Field(default=0.0, description="Ultradian phase offset (rad)")
    # Reference range tuples (Notation Appendix canonical ranges; informational
    # only — not consumed as scalar parameters anywhere in the pipeline)
    BETA_SM_RANGE: tuple[float, float] = Field(default=(0.1, 1.2))
    BETA_SM_WAKING_RANGE: tuple[float, float] = Field(default=(0.3, 0.8))
    BETA_SOMATIC_LINEAR_RANGE: tuple[float, float] = Field(default=(0.5, 2.5))
    TAU_THETA_ALLOSTATIC: float = Field(default=20.0, gt=0)
    TAU_THETA_RECOVERY: float = Field(default=0.45, gt=0)

    @field_validator("pi_max")
    @classmethod
    def validate_pi_range(cls, v: float, info: ValidationInfo) -> float:
        """Ensure pi_max > pi_min."""
        if info.data.get("pi_min") is not None and v <= info.data["pi_min"]:
            raise ValueError("pi_max must be greater than pi_min")
        return v

    @model_validator(mode="after")
    def validate_ne_separation(self) -> APGIConfig:
        """Prevent NE double-counting (Spec §2.3-2.4)."""
        if self.ne_on_precision and self.ne_on_threshold:
            raise ValueError(
                "NE cannot modulate both precision and threshold. "
                "Set exactly one to True. Spec §2.3-2.4."
            )
        return self

    @model_validator(mode="after")
    def validate_ne_threshold_stability(self) -> APGIConfig:
        """Validate NE threshold mode parameters for stability."""
        if self.ne_on_threshold and self.gamma_ne >= 0.1:
            raise ValueError(
                f"ne_on_threshold=True with gamma_ne={self.gamma_ne} causes threshold instability. "
                "Use gamma_ne <= 0.01 or a faster threshold decay. Spec §4.4."
            )
        return self

    @model_validator(mode="after")
    def validate_dt_stability(self) -> APGIConfig:
        """Validate integration step size (Spec §7.4)."""
        min_tau = min(self.tau_s, self.tau_theta, self.tau_pi)  # all in s
        max_dt = min_tau / 10.0
        if self.dt > max_dt:
            raise ValueError(
                f"dt={self.dt} exceeds max {max_dt:.4f}. "
                f"Spec §7.4 requires dt <= min(τ_S, τ_θ, τ_Π) / 10."
            )
        return self

    @model_validator(mode="before")
    @classmethod
    def rename_withdrawn_keys(cls, data: Any) -> Any:
        """κ_e and κ_i are withdrawn (Notation Appendix); accept them for one release."""
        if isinstance(data, dict):
            for old, new in (("kappa_e", "eta_pred_e"), ("kappa_i", "eta_pred_i")):
                if old in data:
                    warn_renamed(old, new, stacklevel=5)
                    data = {k: v for k, v in data.items() if k != old} | {new: data[old]}
        return data

    @model_validator(mode="after")
    def validate_learning_rates(self) -> APGIConfig:
        """Validate generative model learning rates (Spec §1.4)."""
        if self.use_internal_predictions:
            max_eta = 2.0 / self.pi_max
            if self.eta_pred_e >= max_eta:
                raise ValueError(
                    f"eta_pred_e={self.eta_pred_e} >= {max_eta:.4f}. "
                    f"Spec §1.4 requires η_pred,e < 2/Π_max for stability."
                )
            if self.eta_pred_i >= max_eta:
                raise ValueError(
                    f"eta_pred_i={self.eta_pred_i} >= {max_eta:.4f}. "
                    f"Spec §1.4 requires η_pred,i < 2/Π_max for stability."
                )
        return self

    @model_validator(mode="after")
    def apply_backward_compat(self) -> APGIConfig:
        """Apply backward compatibility aliases and ensure consistency."""
        # 1. Somatic bias (beta/beta_da)
        # beta_da takes precedence when it differs from beta, or when beta is
        # still at its own default (so an explicit beta_da wins over an
        # unset beta).
        if self.beta_da is not None and (abs(self.beta_da - self.beta) > 0.01 or self.beta == 1.15):
            self.beta = self.beta_da
        # 2. Ignition temperature (tau_sigma/ignite_tau)
        # Same precedence rule for the spec-preferred tau_sigma name.
        if self.tau_sigma is not None and (
            abs(self.tau_sigma - self.ignite_tau) > 0.01 or self.ignite_tau == 0.5
        ):
            self.ignite_tau = self.tau_sigma
        return self

    def to_dict(self) -> dict:
        """Convert to dictionary for legacy compatibility."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> APGIConfig:
        """Create from dictionary."""
        return cls(**data)


def create_production_config(strict: bool = True, **overrides: Any) -> APGIConfig | dict:
    """Create a production-ready configuration.
    Args:
        strict: If True, return APGIConfig (strict validation). If False, return dict.
        **overrides: Configuration overrides
    Returns:
        APGIConfig instance or dictionary based on strict mode
    """
    config = APGIConfig(**overrides)
    if strict:
        return config
    return config.to_dict()
