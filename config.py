CONFIG = {
    # --- Reproducibility -------------------------------------------------
    # Seed for the pipeline's generator. Every stochastic element (ignition
    # sampling, SDE noise, reservoir weights, Kuramoto phases) draws from it,
    # so a run is bit-for-bit reproducible from this value alone. None =
    # non-deterministic (OS entropy); set an int for any published result.
    "seed": None,
    # Fail closed on spec violations. strict=True (default) raises on a
    # configuration that violates the mathematical support of any parameter,
    # so an invalid run can never masquerade as a valid one. Set False only
    # for exploratory work — the departure is recorded in the run manifest.
    "strict": True,
    # Initial states
    "S0": 0.0,
    # θ₀ canonical range [0.25, 0.85] AU (Notation Appendix, Canonical
    # Parameters table). 0.55 is the band midpoint. The previous default of
    # 1.0 sat outside the canonical range, so every out-of-the-box run was
    # non-canonical.
    "theta_0": 0.55,
    "theta_base": 0.55,
    "sigma2_e0": 1.0,
    "sigma2_i0": 1.0,
    # Numerical stability
    "eps": 1e-8,
    # Spec-mandated precision clamp [Π_min, Π_max] = [0.1, 10] AU (MathSpec
    # §2). Applied after EVERY precision computation, including after each
    # neuromodulatory gain.
    "pi_min": 0.1,
    "pi_max": 10.0,
    # Global θ clamp (numerical stability, MathSpec §15). Externalized rather
    # than hardcoded so sensitivity analyses can widen it explicitly.
    "theta_clip_min": 0.1,
    "theta_clip_max": 20.0,
    # EMA variance update
    "alpha_e": 0.05,
    "alpha_i": 0.05,
    # Variance estimation method
    "variance_method": "ema",  # "ema" or "sliding_window"
    "T_win": 50,  # Sliding window size (only used if variance_method == "sliding_window")
    # Neuromodulation — four transmitter systems (§8)
    # ACh  → Π_e gain:         Π_e_eff = g_ACh · Π_e           (exteroceptive precision)
    # NE   → Π_i gain:         Π_i_eff = g_NE  · Π_i           (interoceptive precision)
    # DA   → error bias:       z_i_eff = z_i + β_DA             (reward prediction error)
    # 5-HT → threshold offset: θ_eff   = θ    + β_5HT           (patience / uncertainty tolerance)
    "g_ach": 1.0,
    "g_ne": 1.0,
    "beta_5ht": 0.0,  # 5-HT patience offset; 0 = no serotonergic modulation (§8.4)
    # --- Dopaminergic additive bias β_DA (Notation Appendix: "Statistical
    # estimators & reserved α-family" table; MathSpec §2 "Dopamine") ---
    # z_i_eff = z_i + β_DA. Unconstrained real (ℝ), NOT a somatic-marker
    # parameter — despite the historical key names below, this is the
    # dopaminergic reward-prediction-error bias, distinct from β_SM and
    # β_somatic (see the Somatic Marker block further down).
    "beta": 1.15,  # β_DA — alias for beta_da (backward compatibility)
    "beta_da": 1.15,  # β_DA — spec-preferred name (alias for backward compatibility)
    # --- Somatic marker parameters — THREE mathematically distinct β
    # symbols per Notation Appendix "β RANGE RECONCILIATION" / Cross-Series
    # Notation Consistency table. Never merge these into one range:
    #   β_SM       — exponential gain in Π_i_eff = Π_i_baseline·exp(β_SM·M);
    #                canonical range [0.1, 1.2], typical waking sub-band
    #                [0.3, 0.8]. This is the parameter actually consumed by
    #                compute_interoceptive_precision_exponential() via the
    #                "beta_sm" key below — the canonical, primary form.
    #   β_somatic  — linear weight in the alternate/reduced accumulation
    #                form S += β_somatic·Π_i·|ε_i| (valid only when
    #                |β·M| << 1; MathSpec §3); canonical range [0.5, 2.5].
    #                Not wired to an active code path by default — the
    #                canonical exponential form (β_SM) is used instead, per
    #                spec's own preference for the non-linear form.
    #   β_DA       — see above; NOT a somatic parameter despite historically
    #                sharing this comment block.
    "beta_sm": 0.3,  # β_SM, waking sub-band [0.3, 0.8] ⊂ canonical [0.1, 1.2]
    "BETA_SM_RANGE": (0.1, 1.2),  # canonical range for β_SM
    "BETA_SM_WAKING_RANGE": (0.3, 0.8),  # typical waking sub-band for β_SM
    "BETA_SOMATIC_LINEAR_RANGE": (0.5, 2.5),  # canonical range for β_somatic (linear form)
    # NE modulates BOTH Π_sal (precision) AND η_θ (threshold gain) per spec.
    # Enabling both simultaneously is allowed by the spec but requires γ_NE·g_NE ≤ 0.5.
    # Default: precision only (ne_on_threshold=False) to avoid accidental instability.
    # Set ne_on_threshold=True AND lower gamma_ne (≤0.01) to enable both pathways.
    "ne_on_precision": True,
    "ne_on_threshold": False,
    # NE modulation strength — spec constraint: γ_NE · g_NE ≤ 0.5
    "gamma_ne": 0.1,
    # Threshold mean-reversion timescale τ_θ (s). The per-step decay exponent
    # is γ_θ·dt = dt/τ_θ, computed by decay_exponent() below, never stored:
    # the former per-step constant "kappa" = 0.15 at dt = 0.5 s implied
    # τ_θ ≈ 3.3 s, outside the published 10–30 s (Math Spec p5, worked value
    # 20 s), and bare κ is reserved for ATP/bit (Notation Appendix §11).
    # A legacy config["kappa"] is still honoured, with a deprecation warning.
    "tau_theta": 20.0,
    # Signal accumulation.
    # λ is the discrete leaky-integration rate. MathSpec §3 gives the EXACT
    # correspondence to the continuous form: λ = 1 − exp(−dt/τ_S); the
    # approximation λ ≈ dt/τ_S holds only when dt = 1. With dt = 0.5 and
    # τ_S = 5.0 below, λ = 1 − exp(−0.1) = 0.0952. The former default of 0.2
    # was ~2× the value its own comment ("dt/tau_s ≈ lam") claimed, so the
    # discrete and continuous formulations disagreed by a factor of two.
    # Use lam_from_tau_s() in this module to keep them consistent.
    "lam": 0.0327838995179941,  # = lam_from_tau_s(dt=0.01, tau_s=0.3)
    "signal_log_nonlinearity": True,
    "use_canonical_discrete_mode": False,  # Use discrete leaky accumulation instead of ODE
    # Threshold update + refractory dynamics
    # η_θ per step = continuous rate × dt (Math Spec p3: "continuous rate
    # η_θ/dt"). 0.002 = 0.2 s⁻¹ × 0.01 s, the rate the former default (0.1
    # per 0.5 s step) implied.
    "eta": 0.002,
    "delta": 0.5,
    # 9) Post-ignition signal reset (§6)
    "RHO_RETAIN": 0.1,  # Matches Spec §6.1; strong reset (retains 10%, clears 90%)
    "DELTA_RESET": 0.5,  # Baseline threshold boost post-ignition
    "reset_factor": 0.1,  # Alias for RHO_RETAIN (backward compatibility)
    # 10) Threshold Adaptation Timescales (Audit HIGH-4)
    "TAU_THETA_ALLOSTATIC": 20.0,  # 10-30s range; governing θ₀ (metabolic)
    "TAU_THETA_RECOVERY": 0.45,  # 300-600ms range; perceptual timescale (Level 1)
    # 11) Hierarchical Cascade Tuning
    "KAPPA_UP": 0.1,  # Bottom-up cascade strength (suppression)
    "KAPPA_DOWN": 0.1,  # Top-down coupling strength (PAC)
    # Cost-value model
    "use_realistic_cost": True,
    "c0": 0.0,
    "c1": 0.2,
    "c2": 0.5,
    "v1": 0.5,
    "v2": 0.5,
    # Ignition dynamics
    "ignite_tau": 0.5,  # Spec alias: tau_sigma
    "tau_sigma": 0.5,  # Spec-preferred name (alias for backward compatibility)
    "stochastic_ignition": False,
    # Maximal ignition rate r_max (s⁻¹) of the dt-invariant hazard
    # P(B=1) = 1 − exp(−r_max·Bₜ·dt) (Math Spec Eq. 35). The corpus fits it per
    # study; 100 s⁻¹ = 1/dt₀ makes the hazard agree with the former per-step
    # Bernoulli(Bₜ) to first order at the reference step dt₀ = 10 ms.
    "r_max": 100.0,
    # Continuous-time signal ODE / SDE parameters
    # τ_S = 300 ms, the worked value (Notation Appendix p2; range 200–500 ms).
    # The former 5.0 s lay outside the published range.
    "tau_s": 0.3,
    # dt₀ = 10 ms, the reference step at which θ₀ and τ_σ are calibrated
    # (Math Spec p1). Rule: dt ≤ min(τ_S, τ_θ, τ_Π)/10 (Notation Appendix p16);
    # ≤ 1 ms when the reservoir is active (Math Spec p26).
    "dt": 0.01,
    "noise_std": 0.01,  # SDE diffusion coefficient
    # Generative model dynamics (§1.4)
    "use_internal_predictions": True,
    "kappa_e": 0.01,
    "kappa_i": 0.01,
    # Multi-scale recommendation parameter
    "timescale_k": 1.6,
    # Hierarchical architecture (§8)
    # Prefer setting "hierarchical_mode" to one of: off/basic/advanced/full (see pipeline.py).
    "hierarchical_mode": None,
    "use_hierarchical": False,
    # Per-level arrays represent spec levels 1..L only (the APGI ignition
    # hierarchy). Index 0 = spec Level 1 (first APGI ignition level), NOT
    # spec Level 0 (the sub-APGI reflexive brainstem/collicular substrate,
    # 10-100ms, which never participates in ignition and has no slot here).
    "n_levels": 3,
    "tau_0": 10.0,  # Base timescale for index-0 level (ms). Lightweight/demo
    # value — does not by itself claim the Notation Appendix's Level-1 span
    # (100ms-3s); for a spec-canonical timescale range use
    # hierarchical_timescale_mode="canonical" below.
    # k: inter-level timescale ratio. IMPORTANT — two distinct spec
    # quantities share the symbol k and must not be conflated (Notation
    # Appendix, Cross-Series Notation Consistency): the canonical inter-level
    # coarse-graining ratio is k≈133 (Δ≈2.1 log-decades/level); Hasson et al.
    # (2008)'s ~1.3-2.0 is the *within-level* adjacent-region gradient, a
    # distinct quantity that must never be substituted here (doing so yields
    # L≈49 instead of L≈5 in the level-count formula). The default below
    # (1.6) is a lightweight/demo value, not the canonical inter-level ratio.
    "k": 1.6,  # Timescale ratio between levels (see note above)
    # hierarchical_timescale_mode: "custom" (default) uses n_levels/tau_0/k
    # above verbatim. "canonical" derives (n_levels, taus) from the spec
    # formula L = floor(log(tau_max/tau_min)/log(k)) + 1 with k≈133,
    # tau_min≈100ms (Level 1), tau_max≈1yr (Level 4) — see
    # hierarchy.coupling.build_canonical_hierarchy_timescales().
    "hierarchical_timescale_mode": "custom",
    "hierarchy_tau_min": 100.0,  # ms, spec Level 1 lower bound (canonical mode)
    "hierarchy_tau_max": 3.0e10,  # ms, ~1 year, spec Level 4 upper bound (canonical mode)
    "k_canonical": 133.0,  # Notation Appendix canonical inter-level ratio (canonical mode)
    # Precision coupling ODE (continuous-time Π cascade)
    "use_hierarchical_precision_ode": False,
    "tau_pi": 1000.0,
    "C_down": 0.1,
    "C_up": 0.05,
    # Bottom-up nonlinear transfer ψ(ε_{ℓ-1}) in precision coupling ODE
    # Options: "identity", "tanh", "softsign".
    # Canonical is "tanh": MathSpec §2 and the Notation Appendix both define
    # ψ(ε) = tanh(γ_ψ·ε) and require saturation ("ψ′(ε) > 0 and bounded —
    # monotone amplification with saturation; prevents unbounded gain at large
    # errors"). "identity" is unbounded and violates that property, so it is
    # retained only as an explicit diagnostic opt-out.
    "psi_type": "tanh",
    # γ_ψ: gain of ψ(ε)=tanh(γ_ψ·ε) when psi_type="tanh". MathSpec glossary
    # canonical range [1.0, 5.0]; Notation Appendix Canonical Parameters
    # table gives [1, 3] — configurable since the two documents disagree
    # slightly on the exact bound. Only used when psi_type="tanh".
    "gamma_psi": 2.0,
    # Thermodynamic constraints (§11)
    "use_thermodynamic_cost": False,  # Enable Landauer's principle
    "k_boltzmann": 1.380649e-23,  # Boltzmann constant (J/K), CODATA exact value
    "T_env": 310.0,  # Environmental temperature (K, body temp)
    # Erasure-overhead factor on the Landauer floor, formerly κ_meta (deprecated,
    # Notation Appendix §11: R_L replaces κ_meta). Absorbs the Joules→AU conversion.
    "r_l": 1e20,
    "r_l_units": "dimensionless",  # "dimensionless" or "joules_per_bit"
    # BOLD fMRI calibration for energy conversion
    "use_bold_calibration": False,  # Enable BOLD-to-Joule conversion
    "bold_conversion_factor": 1.2e-18,  # Joules per 1% BOLD change per cm³ tissue
    "bold_tissue_volume": 1.0,  # Tissue volume in cm³
    "bold_ignition_spike_factor": 1.075,  # 7.5% energy spike during ignition (5-10% range)
    # Signed Nonlinear Prediction Error Transform φ(ε) (§6)
    # α⁺, α⁻ ∈ [0.5, 2.0] — valence-specific amplitude gain
    # γ⁺, γ⁻ ∈ [1.0, 5.0] — saturation steepness
    # Symmetric defaults (α⁺=α⁻, γ⁺=γ⁻) recover the unsigned |ε| approximation
    "alpha_plus": 1.0,  # reward/approach gain
    "alpha_minus": 1.0,  # threat/avoidance gain
    "gamma_plus": 2.0,  # reward saturation slope
    "gamma_minus": 2.0,  # avoidance saturation slope
    # Reservoir layer (§10)
    "use_reservoir": False,  # Enable reservoir computing layer
    "reservoir_size": 100,  # Number of reservoir units
    "reservoir_tau": 1.0,  # Base time constant (ms)
    "reservoir_spectral_radius": 0.9,  # Spectral radius (must be < 1)
    "reservoir_input_scale": 0.1,  # Input weight scaling
    "reservoir_readout_method": "linear",  # "linear" or "energy"
    "reservoir_amplification": 0.0,  # Suprathreshold amplification strength
    "reservoir_ridge_alpha": 1e-6,  # Ridge regression regularization
    "reservoir_weight": 0.1,  # w_res in S_global = Σw_l·S_l + w_res·S_res (spec §10)
    # False (default): additive add-on mode (Dynamical Formulation §17 multi-level
    #   formula) — reservoir contributes w_res*S_res alongside hierarchical levels.
    # True: MathSpec §13 step-15 replacement mode — S(t) = W_out^T·x(t) fully
    #   replaces the explicit S_inst/leaky-accumulation of steps 8-9, matching
    #   the single-scale spec form exactly (not additive).
    "reservoir_replaces_signal": False,
    # Kuramoto oscillators (§9)
    "use_kuramoto": False,  # Enable Kuramoto oscillators with phase noise
    "kuramoto_tau_xi": 1.0,  # OU noise correlation timescale (ms)
    "kuramoto_sigma_xi": 0.1,  # OU noise amplitude (rad/ms)
    # Phase reset on ignition. None (default) applies the spec-canonical
    # reset to zero-phase (Δφ_reset = −φ_level(t)), matching MathSpec §9's
    # "canonical reset target is Δφ_reset = −φ_level(t) (reset to
    # zero-phase)". Set to an explicit radian value (e.g. 3.14159 for a
    # fixed +π kick) to use the legacy fixed-additive-kick semantics instead.
    "kuramoto_reset_amount": None,
    # Observable mapping (§14)
    "use_observable_mapping": False,  # Enable neural/behavioral observable extraction
    # Stability analysis (§7)
    "use_stability_analysis": False,  # Enable fixed-point stability analysis
    # Cross-Level Threshold Resonance — Russian Doll Architecture (§8 / §9)
    # True = per-level S_l accumulators + θ_l = θ_{0,l}·(1+κ·Π_{l+1}·cos(φ_{l+1})) (spec §8)
    # False = single aggregated S, parallel accumulator fallback (non-affective / diagnostic only)
    # Has no effect when use_hierarchical=False.
    "use_resonance": True,
    "resonance_kappa_down": 0.1,  # Top-down coupling κ_down (threshold + phase entrainment)
    "resonance_kappa_up": 0.0,  # Bottom-up suppression after level-0 ignition
    "resonance_phi_noise_std": 0.0,  # Per-step phase jitter (biological noise)
    # Colored (OU) vs white phase noise when resonance_phi_noise_std > 0.
    # MathSpec §9 prefers OU ("more consistent with neural oscillation
    # variability than white noise"); True (default) uses OU. No effect
    # while resonance_phi_noise_std=0 (the default, noise disabled).
    "resonance_use_ou_phase_noise": True,
    # Active Inference Action Loop (§19)
    "use_active_inference": False,  # Enable perception-action loop
    "ai_on_ignition_only": True,  # Fire only on Bₜ=1 (True) or every step (False)
    "ai_n_actions": 3,  # Number of discrete candidate policies K
    "ai_policy_precision": 2.0,  # Boltzmann sharpness γ_policy
    "ai_w_epistemic": 1.0,  # Weight on epistemic term of EFE(a)
    "ai_w_pragmatic": 1.0,  # Weight on pragmatic term of EFE(a)
    "ai_w_metabolic": 0.5,  # Weight on metabolic cost term
    "ai_sensory_feedback_rate": 0.1,  # Scale of channel-1 Δx̂ₑ (sensory)
    "ai_metabolic_feedback_rate": 0.1,  # Scale of channel-2 ΔM (somatic marker + metabolic state)
    "ai_precision_update_rate": 0.05,  # Scale of channel-3 ΔΣ (epistemic)
    "ai_action_params": None,  # None → use default 3-action table
    # Metabolic state M(t) (Notation Appendix; Dynamical Formulation §13-14)
    # — DISTINCT from the somatic marker M(c,a) (config["M_somatic"]) despite
    # sharing the glyph M. Continuous, normalized [0, 1]. Feeds C(t) via
    # c_metabolic_state, closing active-inference channel 2's "motor
    # actions change metabolic state M, modulating θ_t via allostatic
    # feedback" loop. c_metabolic_state=0.0 is an exact no-op (preserves
    # C(t) exactly as before this was added) unless explicitly enabled.
    "metabolic_state0": 0.5,
    "c_metabolic_state": 0.0,
    # Circadian/ultradian theta modulation (§27, peripheral prediction —
    # core/circadian.py). Off by default (no effect on existing configs).
    "use_circadian_modulation": False,
    "circadian_t0": 0.0,  # seconds
    "circadian_A_circ": 0.1,
    "circadian_T_circ": 86400.0,  # 24h in seconds
    "circadian_phi_circ": 0.0,
    "circadian_A_ultradian": 0.05,
    "circadian_T_ultradian": 5400.0,  # 90 min in seconds
    "circadian_phi_ultradian": 0.0,
    # Bounded diagnostic history. self.history previously grew without limit
    # (~113 bytes/step → ~1.1 GB at 1e7 steps), which is reachable: Levels 3-4
    # span hours-to-months of simulated time and DFA needs N ≥ 1000. None =
    # unbounded (legacy); an int keeps the most recent N samples in a ring
    # buffer. 1e6 samples ≈ 110 MB, ample for every published analysis here.
    "history_maxlen": 1_000_000,
}


def decay_exponent(config: dict) -> float:
    """Per-step threshold decay exponent γ_θ·dt = dt/τ_θ (Math Spec Eq. 32).

    A legacy ``config["kappa"]`` (the former stored per-step exponent) takes
    precedence when present, since CONFIG no longer defines it, and emits an
    :class:`APGIDeprecationWarning`.
    """
    from core.deprecation import warn_renamed

    if "kappa" in config:  # CONFIG has no "kappa", so a present key was set explicitly
        warn_renamed("kappa", "tau_theta (exponent = dt/tau_theta)")
        return float(config["kappa"])
    if "tau_theta" in config:
        tau_theta = float(config["tau_theta"])
        if tau_theta <= 0:
            raise ValueError(f"tau_theta must be > 0, got {tau_theta}")
        return float(config.get("dt", 1.0)) / tau_theta
    raise KeyError("config needs 'tau_theta'")


def lam_from_tau_s(dt: float, tau_s: float) -> float:
    """Exact discrete leaky rate λ = 1 − exp(−dt/τ_S) (MathSpec §3).

    The spec is explicit that this identity — not the λ ≈ dt/τ_S
    approximation — is what makes the discrete and continuous formulations
    agree: "the discrete leaky rate is exactly λ = 1 − exp(−dt/τ_S); the
    approximation τ_S ≈ 1/λ holds only when dt = 1".

    Args:
        dt: Integration timestep, must be > 0.
        tau_s: Signal decay timescale τ_S, must be > 0.

    Returns:
        λ ∈ (0, 1).

    Raises:
        ValueError: If ``dt`` or ``tau_s`` is not positive.
    """
    import math

    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    if tau_s <= 0:
        raise ValueError(f"tau_s must be > 0, got {tau_s}")
    return 1.0 - math.exp(-dt / tau_s)
