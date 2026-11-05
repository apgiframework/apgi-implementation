from __future__ import annotations

import warnings
from collections import deque
from collections.abc import MutableSequence
from dataclasses import dataclass, field

import numpy as np

from active_inference.policy import ActiveInferenceAgent
from core.allostatic import allostatic_threshold_ode
from core.dynamics import compute_precision_coupled_noise_std, update_prediction, update_signal_ode
from core.finite import require_finite_scalar, require_optional_finite_scalar
from core.ignition import compute_ignition_probability, detect_ignition_event, sample_ignition_state
from core.logging_config import get_logger
from core.manifest import build_manifest, write_manifest
from core.phi_transform import phi_transform, phi_transform_array
from core.precision import (
    apply_ach_gain,
    apply_dopamine_bias_to_error,
    apply_ne_gain,
    clamp,
    compute_interoceptive_precision_exponential,
    compute_precision,
    update_mean_ema,
)
from core.preprocessing import RunningStats, compute_prediction_error
from core.rng import resolve_rng
from core.signal import instantaneous_signal_phi, integrate_signal_leaky, stabilize_signal_log
from core.thermodynamics import compute_landauer_cost
from core.threshold import (
    apply_ne_threshold_modulation,
    apply_refractory_boost,
    apply_serotonin_threshold_offset,
    compute_information_value,
    compute_metabolic_cost,
    compute_metabolic_cost_realistic,
    threshold_decay,
    update_threshold_discrete,
)
from core.validation import (
    CanonicalRangeWarning,
    ValidationError,
    check_canonical_ranges,
    validate_config,
)
from hierarchy.coupling import HierarchicalPrecisionNetwork
from hierarchy.multiscale import (
    aggregate_multiscale_signal_phi,
    build_timescales,
    multiscale_weights,
    update_multiscale_feature,
)
from hierarchy.resonance import NestedResonanceSystem, build_resonance_system
from stats.hurst import estimate_hurst_robust
from stats.spectral_model import validate_pink_noise

logger = get_logger("apgi.pipeline")


@dataclass
class PrecisionState:
    sigma2_e: float
    sigma2_i: float
    pi_e: float = 1.0
    pi_i: float = 1.0
    mu_e: float = field(default=0.0)  # EMA mean for exteroceptive errors
    mu_i: float = field(default=0.0)  # EMA mean for interoceptive errors


@dataclass
class HierarchicalState:
    """State for hierarchical precision and threshold coupling."""

    n_levels: int = 1
    pis: list[float] = None  # type: ignore[assignment]
    thetas: list[float] = None  # type: ignore[assignment]
    phases: list[float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.pis is None:
            self.pis = [1.0] * self.n_levels
        if self.thetas is None:
            self.thetas = [1.0] * self.n_levels
        if self.phases is None:
            self.phases = [0.0] * self.n_levels


def _new_history_buffer(maxlen: int | None) -> MutableSequence[float]:
    """Create a diagnostic history buffer.

    Args:
        maxlen: Maximum retained samples, or ``None`` for an unbounded list.

    Returns:
        A ``deque`` bounded to ``maxlen`` (oldest samples evicted first), or a
        plain ``list`` when ``maxlen`` is ``None``. Both satisfy the
        ``append``/``len``/indexing/iteration the analysis paths use, and
        ``np.asarray`` accepts either, so downstream code is unaffected.

    Raises:
        ValueError: If ``maxlen`` is not ``None`` and not >= 1.
    """
    if maxlen is None:
        return []
    if maxlen < 1:
        raise ValueError(f"history_maxlen must be >= 1 or None, got {maxlen}")
    return deque(maxlen=maxlen)


class APGIPipeline:
    """APGI one-step update implementing the full corrected mathematical pipeline."""

    #: Spec-mandated ceiling on γ_NE · g_NE (MathSpec §2). Keeps the
    #: multiplicative NE threshold modulation sub-dominant to the allostatic
    #: update, preventing non-linear instability when both NE pathways run.
    NE_CLAMP_MAX: float = 0.5

    M: float
    history: dict[str, MutableSequence[float]]
    mu_e_levels: np.ndarray
    mu_i_levels: np.ndarray
    sigma2_e_levels: np.ndarray
    sigma2_i_levels: np.ndarray
    n_levels: int
    taus: np.ndarray

    def manifest(self, **extra: object) -> dict:
        """Return a provenance manifest for this pipeline's run.

        Records the seed, the RESOLVED config and its hash, the git commit,
        the interpreter and every numerical dependency version, plus any spec
        departures accumulated during construction. Write it beside any figure
        or table this pipeline produces so the result can be regenerated.

        Args:
            **extra: Additional run-specific fields (step count, output paths).

        Returns:
            JSON-serialisable manifest dictionary.
        """
        return build_manifest(
            self.config,
            extra=dict(extra) or None,
            validation_warnings=self._validation_warnings,
        )

    def write_manifest(self, path: str, **extra: object) -> object:
        """Write this run's provenance manifest to ``path`` as JSON.

        Args:
            path: Destination file; parent directories are created as needed.
            **extra: Additional run-specific fields.

        Returns:
            The path written.
        """
        return write_manifest(
            path,
            self.config,
            extra=dict(extra) or None,
            validation_warnings=self._validation_warnings,
        )

    def _enforce_ne_clamp(self) -> None:
        """Enforce the mandatory NE sub-dominance clamp γ_NE · g_NE ≤ 0.5.

        MathSpec §2: "Implementations MUST enforce the hard clamp
        γ_NE · g_NE ≤ 0.5 at every timestep to prevent non-linear instability;
        this is a strict architectural constraint, not an optional toggle.
        Concretely, if γ_NE · g_NE(t) > 0.5, rescale g_NE(t) ← 0.5 / γ_NE
        before the threshold update."

        Rescaling (rather than raising) is what the spec prescribes, so a
        configuration that overshoots is corrected and reported rather than
        rejected. The clamp only binds when the multiplicative threshold
        pathway is active; with ne_on_threshold=False, g_NE scales precision
        only and no instability can arise from this product.
        """
        if not self.config.get("ne_on_threshold", False):
            return
        gamma_ne = float(self.config.get("gamma_ne", 0.1))
        g_ne = float(self.config.get("g_ne", 1.0))
        if gamma_ne <= 0.0:
            return
        product = gamma_ne * g_ne
        if product > self.NE_CLAMP_MAX:
            rescaled = self.NE_CLAMP_MAX / gamma_ne
            msg = (
                f"γ_NE·g_NE = {product:.3f} exceeds the spec-mandated ceiling "
                f"{self.NE_CLAMP_MAX}; rescaling g_NE {g_ne:.3f} → {rescaled:.3f} "
                "(MathSpec §2, hard clamp)."
            )
            self.config["g_ne"] = rescaled
            self._validation_warnings.append(msg)
            warnings.warn(msg, RuntimeWarning, stacklevel=3)
            logger.warning("ne_clamp_applied", product=product, g_ne_rescaled=rescaled)

    def _warn_ne_threshold_stiffness(self) -> None:
        """Warn when NE threshold modulation is stiff relative to θ recovery.

        This is an empirical stability heuristic, NOT a spec constraint: with
        the multiplicative NE pathway active, a large γ_NE combined with a
        slow threshold decay κ lets θ ratchet upward faster than it relaxes.

        It only warns. The previous implementation silently rewrote both
        ``gamma_ne`` and ``kappa`` behind the caller's back, so a run could
        report parameters it had not actually used — a reproducibility hazard
        for an artifact whose outputs back published figures. Choosing the
        remedy is the caller's decision; naming it precisely is ours.
        """
        if not self.config.get("ne_on_threshold", False):
            return
        gamma_ne = float(self.config.get("gamma_ne", 0.1))
        kappa = float(self.config.get("kappa", 0.15))
        if gamma_ne >= 0.1 and kappa <= 0.15:
            msg = (
                f"ne_on_threshold=True with gamma_ne={gamma_ne} and kappa={kappa} "
                "risks threshold instability: multiplicative NE gain can raise θ "
                "faster than the decay relaxes it. Prefer gamma_ne <= 0.01 or "
                "kappa >= 0.15 (§4.4). Not auto-corrected — the configuration "
                "you pass is the configuration that runs."
            )
            self._validation_warnings.append(msg)
            warnings.warn(msg, RuntimeWarning, stacklevel=3)

    def _apply_hierarchical_preset(self, config: dict) -> dict:
        """Apply hierarchical mode preset to configuration.
        Simplifies hierarchical system configuration by allowing users to set
        a single 'hierarchical_mode' parameter instead of three separate flags.
        Modes:
            - 'off': Disable all hierarchical features
            - 'basic': Enable hierarchical only (use_hierarchical=True)
            - 'advanced': Enable hierarchical + precision ODE (use_hierarchical=True,
                         use_hierarchical_precision_ode=True)
            - 'full': Enable all hierarchical features (use_hierarchical=True,
                     use_hierarchical_precision_ode=True, use_phase_modulation=True)
        Args:
            config: Configuration dictionary
        Returns:
            Updated configuration with hierarchical flags set
        Raises:
            ValueError: If hierarchical_mode has unknown value
        """
        mode = config.get("hierarchical_mode")
        # Only apply preset if hierarchical_mode is explicitly set
        if mode is None:
            return config
        if mode == "off":
            config.update(
                {
                    "use_hierarchical": False,
                    "use_hierarchical_precision_ode": False,
                    "use_phase_modulation": False,
                }
            )
        elif mode == "basic":
            config.update(
                {
                    "use_hierarchical": True,
                    "use_resonance": True,
                    "use_hierarchical_precision_ode": False,
                    "use_phase_modulation": False,
                }
            )
        elif mode == "advanced":
            config.update(
                {
                    "use_hierarchical": True,
                    "use_resonance": True,
                    "use_hierarchical_precision_ode": True,
                    "use_phase_modulation": False,
                }
            )
        elif mode == "full":
            config.update(
                {
                    "use_hierarchical": True,
                    "use_resonance": True,
                    "use_hierarchical_precision_ode": True,
                    "use_phase_modulation": True,
                }
            )
        else:
            raise ValueError(
                f"Unknown hierarchical_mode: {mode}. Must be one of: 'off', 'basic', 'advanced', 'full'"
            )
        return config

    def _compute_per_level_errors(
        self,
        epsilon_e: float,
        epsilon_i: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-level z-scores for hierarchical system.
        Spec §7: Each level ℓ processes errors at its own timescale τ_ℓ.
        This version is fully vectorized for performance (§4.1).
        Args:
            epsilon_e: Exteroceptive prediction error
            epsilon_i: Interoceptive prediction error
        Returns:
            (z_e_levels, z_i_levels) — z-scores at each level (np.ndarray)
        """
        if not self.use_hierarchical:
            # Single-scale case
            z_e = (epsilon_e - self.state.mu_e) / (self.state.sigma2_e**0.5 + self.config["eps"])
            z_i = (epsilon_i - self.state.mu_i) / (self.state.sigma2_i**0.5 + self.config["eps"])
            return np.array([z_e]), np.array([z_i])
        # Multi-scale: fully vectorized EMA updates
        alphas = 1.0 / self.taus
        # Vectorized mean updates
        self.mu_e_levels = (1.0 - alphas) * self.mu_e_levels + alphas * epsilon_e
        self.mu_i_levels = (1.0 - alphas) * self.mu_i_levels + alphas * epsilon_i
        # Vectorized variance updates (centered)
        self.sigma2_e_levels = (1.0 - alphas) * self.sigma2_e_levels + alphas * (
            epsilon_e - self.mu_e_levels
        ) ** 2
        self.sigma2_i_levels = (1.0 - alphas) * self.sigma2_i_levels + alphas * (
            epsilon_i - self.mu_i_levels
        ) ** 2
        # Vectorized z-scores
        z_e_levels = (epsilon_e - self.mu_e_levels) / (
            np.sqrt(self.sigma2_e_levels) + self.config["eps"]
        )
        z_i_levels = (epsilon_i - self.mu_i_levels) / (
            np.sqrt(self.sigma2_i_levels) + self.config["eps"]
        )
        return z_e_levels, z_i_levels

    def __init__(self, config: dict):
        # Copy to avoid mutating the caller's dict
        self.config = dict(config)
        # Apply hierarchical mode preset if specified
        self.config = self._apply_hierarchical_preset(self.config)
        # Support spec-preferred parameter names with backward compatibility
        if "beta_da" in self.config and "beta" not in self.config:
            self.config["beta"] = self.config["beta_da"]
        if "tau_sigma" in self.config and "ignite_tau" not in self.config:
            self.config["ignite_tau"] = self.config["tau_sigma"]
        # Validate configuration against spec constraints (§15).
        #
        # FAIL CLOSED. A configuration that violates the specification is a
        # defect, not a preference: silently continuing produces numbers that
        # look plausible but are not APGI (e.g. λ=1.5 breaks the leaky
        # integrator outright while step() still returns finite values). Since
        # this package is the cited artifact for the paper series, an invalid
        # run must never masquerade as a valid one.
        #
        # `strict=False` is the documented, explicit opt-out for exploratory
        # work; it downgrades violations to warnings and is recorded in the
        # run manifest so any output produced under it is traceable.
        self.strict = bool(self.config.get("strict", self.config.get("strict_mode", True)))
        self._validation_warnings: list[str] = []
        try:
            validate_config(self.config)
            logger.debug("config_validation_passed", strict=self.strict)
        except ValidationError as e:
            if self.strict:
                raise
            self._validation_warnings.append(str(e))
            warnings.warn(
                f"Config validation failed: {e}. Continuing because strict=False; "
                "results from this run are NOT spec-conformant.",
                RuntimeWarning,
                stacklevel=2,
            )
        # Canonical-range advisories (Notation Appendix). These never abort a
        # run — out-of-band-but-in-support values are legitimate for sensitivity
        # analysis — but they are surfaced and recorded in the manifest.
        for msg in check_canonical_ranges(self.config):
            self._validation_warnings.append(msg)
            warnings.warn(msg, CanonicalRangeWarning, stacklevel=2)
        # Spec constraint (MathSpec §2): γ_NE · g_NE ≤ 0.5 is a hard clamp that
        # implementations MUST enforce at every timestep — "a strict
        # architectural constraint, not an optional toggle". The spec also
        # prescribes the remedy: rescale g_NE ← 0.5/γ_NE rather than abort.
        # Both NE pathways may be active simultaneously provided this holds;
        # the previous blanket ban on ne_on_precision+ne_on_threshold was
        # stricter than the spec and contradicted config.py's own guidance.
        self._enforce_ne_clamp()
        self._warn_ne_threshold_stiffness()
        self.S = float(self.config["S0"])
        # One generator drives every stochastic element of this pipeline
        # (ignition sampling, SDE noise, reservoir weights, phase noise), so a
        # run is bit-for-bit reproducible from config["seed"] alone.
        self.seed = self.config.get("seed")
        self.rng = resolve_rng(self.config.get("rng", self.seed))
        self.theta = float(self.config["theta_0"])
        self.theta_dot = 0.0  # For continuous ODE tracking
        # Internalized predictions per §1.4 if enabled
        self.x_hat_e = float(self.config.get("x_hat_e0", 0.0))
        self.x_hat_i = float(self.config.get("x_hat_i0", 0.0))
        self.state = PrecisionState(
            sigma2_e=float(self.config["sigma2_e0"]),
            sigma2_i=float(self.config["sigma2_i0"]),
        )
        self.B_prev = 0
        self.t = 0.0  # Continuous time (phase tracking + SDE)
        # Initialize sliding-window stats if variance_method is "sliding_window"
        self.stats_e = None
        self.stats_i = None
        if self.config.get("variance_method", "ema") == "sliding_window":
            T_win = self.config.get("T_win", 50)
            self.stats_e = RunningStats(window_size=T_win)
            self.stats_i = RunningStats(window_size=T_win)
        # Hierarchical state if enabled
        self.hierarchical = None
        self.hierarchical_network = None
        self.use_hierarchical = self.config.get("use_hierarchical", False)
        if self.use_hierarchical:
            # Per-level arrays (taus, phi_*_levels, weights, ...) represent
            # spec levels 1..L only (the APGI ignition hierarchy). Index 0 of
            # every such array is spec Level 1 (first APGI ignition level),
            # NOT spec Level 0 (the sub-APGI reflexive brainstem/collicular
            # substrate, 10-100ms). Level 0 is architecturally excluded from
            # ignition and from these arrays entirely (Notation Appendix,
            # "Level Architecture"); it is never allocated a slot here.
            #
            # hierarchical_timescale_mode selects how (n_levels, tau_0, k)
            # are resolved:
            #   "custom"    (default) — use n_levels/tau_0/k from config
            #                verbatim. These defaults (tau_0=10ms, k=1.6,
            #                n_levels=3) are a lightweight, non-canonical
            #                configuration for tests/demos, not a claim that
            #                the resulting span matches the Notation
            #                Appendix's Level 1-4 timescale table.
            #   "canonical" — derive (n_levels, taus) from the spec formula
            #                L = floor(log(tau_max/tau_min)/log(k)) + 1 using
            #                the canonical k≈133 inter-level ratio, spanning
            #                tau_min≈100ms (Level 1) to tau_max≈1yr (Level 4).
            if self.config.get("hierarchical_timescale_mode", "custom") == "canonical":
                from hierarchy.coupling import (
                    K_INTER_LEVEL_CANONICAL,
                    TAU_MAX_CANONICAL_MS,
                    TAU_MIN_CANONICAL_MS,
                    build_canonical_hierarchy_timescales,
                )

                k_used = self.config.get("k_canonical", K_INTER_LEVEL_CANONICAL)
                self.taus = build_canonical_hierarchy_timescales(
                    self.config.get("hierarchy_tau_min", TAU_MIN_CANONICAL_MS),
                    self.config.get("hierarchy_tau_max", TAU_MAX_CANONICAL_MS),
                    k_used,
                )
                n_levels = len(self.taus)
            else:
                n_levels = self.config.get("n_levels", 3)
                k_used = self.config.get("k", 1.6)
                self.taus = build_timescales(self.config.get("tau_0", 10.0), k_used, n_levels)
            self.hierarchical = HierarchicalState(n_levels=n_levels)
            # Per-level statistics for true hierarchical error processing
            self.n_levels = n_levels
            self.mu_e_levels = np.zeros(n_levels)
            self.mu_i_levels = np.zeros(n_levels)
            self.sigma2_e_levels = np.ones(n_levels)
            self.sigma2_i_levels = np.ones(n_levels)
            # Initialize hierarchical feature states for signal aggregation
            self.phi_e_levels = np.zeros(n_levels)
            self.phi_i_levels = np.zeros(n_levels)
            # w_ℓ = k^{-ℓ}/Z is defined over the *same* ratio used to build
            # taus, so canonical mode must not silently fall back to
            # config["k"] here — that would decouple weight decay from the
            # actual timescale spacing.
            self.weights = multiscale_weights(n_levels, k_used)
            # Initialize oscillatory phases for threshold modulation (basic mode)
            self.phase_levels = np.zeros(n_levels)
            self.omega_levels = 2 * np.pi / self.taus  # Natural frequencies from timescales
            # Initialize HierarchicalPrecisionNetwork for proper per-level error computation
            # Only enable when use_hierarchical_precision_ode is explicitly True
            if self.config.get("use_hierarchical_precision_ode", False):
                psi_type = self.config.get("psi_type", "identity")
                # γ_ψ: gain of the bottom-up modulation function ψ(ε) =
                # tanh(γ_ψ·ε) (MathSpec glossary: γ_ψ ∈ [1.0, 5.0]; Notation
                # Appendix Canonical Parameters table: γ_ψ ∈ [1, 3] — the
                # two documents give slightly different bounds, so this is
                # left user-configurable rather than hardcoded to either.
                # Previously this was silently γ_ψ≡1 (bare tanh), never
                # exposed as a parameter.
                gamma_psi = self.config.get("gamma_psi", 2.0)
                if psi_type == "identity" or psi_type is None:
                    psi_fn = None
                elif psi_type == "tanh":

                    def psi_fn(x: np.ndarray, _g: float = gamma_psi) -> np.ndarray:  # type: ignore[misc]
                        return np.tanh(_g * x)  # type: ignore[no-any-return]

                elif psi_type == "softsign":

                    def psi_fn(x: np.ndarray) -> np.ndarray:  # type: ignore[misc]
                        return x / (1.0 + np.abs(x))  # type: ignore[no-any-return]

                else:
                    raise ValueError(
                        f"Unknown psi_type: {psi_type}. Must be one of: 'identity', 'tanh', 'softsign'"
                    )
                self.hierarchical_network = HierarchicalPrecisionNetwork(
                    n_levels=n_levels,
                    taus=self.taus,
                    tau_pi=self.config.get("tau_pi", 1000.0),
                    C_down=self.config.get("C_down", 0.1),
                    C_up=self.config.get("C_up", 0.05),
                    psi=psi_fn,
                )
        else:
            # Single-scale: initialize single z-score pair
            self.n_levels = 1
            self.taus = np.array([1.0])
            self.mu_e_levels = np.zeros(1)
            self.mu_i_levels = np.zeros(1)
            self.sigma2_e_levels = np.ones(1)
            self.sigma2_i_levels = np.ones(1)
        # Cross-Level Threshold Resonance (§8/§9) — Russian Doll architecture
        # Active by default whenever use_hierarchical=True (spec §8 canonical mode).
        # Per-level S_l accumulators and live phase-modulated thresholds replace the
        # single aggregated S of "parallel accumulator" mode.  Disable explicitly
        # with use_resonance=False to fall back to the aggregated path.
        self.resonance_system: NestedResonanceSystem | None = None
        if self.use_hierarchical and self.config.get("use_resonance", True):
            self.resonance_system = build_resonance_system(
                n_levels=self.n_levels,
                taus=self.taus,
                theta_base=self.config["theta_base"],
                dt=self.config.get("dt", 1.0),
                kappa_down=self.config.get("resonance_kappa_down", 0.1),
                kappa_up=self.config.get("resonance_kappa_up", 0.0),
                phi_noise_std=self.config.get("resonance_phi_noise_std", 0.0),
                use_ou_phase_noise=self.config.get("resonance_use_ou_phase_noise", True),
            )
        # Somatic marker state if enabled
        self.M = self.config.get("M_somatic", 0.0)  # Somatic marker M(c,a) ∈ [-2, +2]
        # Continuous metabolic state M(t) ∈ [0, 1] (Notation Appendix /
        # Dynamical Formulation §13-14) — DISTINCT from the somatic marker
        # above despite sharing the glyph M. Feeds C(t) below (metabolic
        # cost), which is how active-inference channel 2 ("motor actions
        # change metabolic state M, modulating θ_t via allostatic
        # feedback") actually reaches the threshold.
        self.metabolic_state = float(self.config.get("metabolic_state0", 0.5))
        # Circadian/ultradian threshold rhythm (§27, peripheral prediction) —
        # off by default (None), preserving existing behavior exactly.
        self.circadian_regulator = None
        if self.config.get("use_circadian_modulation", False):
            from core.circadian import CircadianRegulator

            self.circadian_regulator = CircadianRegulator(
                t0=self.config.get("circadian_t0", 0.0),
                dt=self.config.get("dt", 1.0),
                A_circ=self.config.get("circadian_A_circ", 0.1),
                T_circ=self.config.get("circadian_T_circ", 86400.0),
                phi_circ=self.config.get("circadian_phi_circ", 0.0),
                A_ultradian=self.config.get("circadian_A_ultradian", 0.05),
                T_ultradian=self.config.get("circadian_T_ultradian", 5400.0),
                phi_ultradian=self.config.get("circadian_phi_ultradian", 0.0),
            )
        # Reservoir layer if enabled (§10)
        self.reservoir = None
        if self.config.get("use_reservoir", False):
            from reservoir.liquid_state_machine import LiquidStateMachine

            self.reservoir = LiquidStateMachine(
                N=self.config.get("reservoir_size", 100),
                M=2,  # [z_e, z_i]
                tau_res=self.config.get("reservoir_tau", 1.0),
                spectral_radius=self.config.get("reservoir_spectral_radius", 0.9),
                input_scale=self.config.get("reservoir_input_scale", 0.1),
                rng=self.rng,
            )
        # Kuramoto oscillators if enabled (§9)
        self.kuramoto = None
        if self.config.get("use_kuramoto", False):
            from oscillation.kuramoto import HierarchicalKuramotoSystem

            n_levels = self.config.get("n_levels", 3)
            # Pass the pipeline generator through config so the oscillator
            # bank's initial phases and OU noise derive from the run seed.
            self.kuramoto = HierarchicalKuramotoSystem(
                n_levels=n_levels,
                config={**self.config, "rng": self.rng},
            )
        # Observable mapping if enabled (§14)
        self.neural_observables = None
        self.behavioral_observables = None
        self.prediction_validator = None
        if self.config.get("use_observable_mapping", False):
            from validation.observable_mapping import (
                BehavioralObservableExtractor,
                KeyTestablePredictionValidator,
                NeuralObservableExtractor,
            )

            self.neural_observables = NeuralObservableExtractor()
            self.behavioral_observables = BehavioralObservableExtractor()
            self.prediction_validator = KeyTestablePredictionValidator(
                tau_sigma=self.config.get("ignite_tau", 0.5)
            )
        # Stability analyzer if enabled (§7)
        self.stability_analyzer = None
        if self.config.get("use_stability_analysis", False):
            from analysis.stability import StabilityAnalyzer

            self.stability_analyzer = StabilityAnalyzer(self.config)
        # Active Inference Action Loop (§19)
        self.active_inference_agent: ActiveInferenceAgent | None = None
        if self.config.get("use_active_inference", False):
            ai_params = self.config.get("ai_action_params", None)
            if ai_params is not None:
                ai_params = np.asarray(ai_params, dtype=float)
            self.active_inference_agent = ActiveInferenceAgent(
                n_actions=self.config.get("ai_n_actions", 3),
                action_params=ai_params,
                policy_precision=self.config.get("ai_policy_precision", 2.0),
                w_epistemic=self.config.get("ai_w_epistemic", 1.0),
                w_pragmatic=self.config.get("ai_w_pragmatic", 1.0),
                w_metabolic=self.config.get("ai_w_metabolic", 0.5),
                sensory_feedback_rate=self.config.get("ai_sensory_feedback_rate", 0.1),
                metabolic_feedback_rate=self.config.get("ai_metabolic_feedback_rate", 0.1),
                precision_update_rate=self.config.get("ai_precision_update_rate", 0.05),
                alpha_pos=self.config.get("alpha_plus", 1.0),
                alpha_neg=self.config.get("alpha_minus", 1.0),
                gamma_pos=self.config.get("gamma_plus", 2.0),
                gamma_neg=self.config.get("gamma_minus", 2.0),
            )
        # Diagnostic history. Bounded by default: an unbounded list grows at
        # ~113 bytes/step (~1.1 GB at 1e7 steps), and Levels 3-4 span
        # hours-to-months of simulated time. config["history_maxlen"]=None
        # restores unbounded (legacy) behaviour.
        _maxlen = self.config.get("history_maxlen", 1_000_000)
        self.history_maxlen = _maxlen
        self.history = {
            "S": _new_history_buffer(_maxlen),
            "theta": _new_history_buffer(_maxlen),
            "B": _new_history_buffer(_maxlen),
        }
        # Multiscale signal tracking for spectral validation
        self.multiscale_signal_history: list[float] = []
        self.per_level_history: list[list[float]] = []
        # Thermodynamic cost history if enabled (§11)
        if self.config.get("use_thermodynamic_cost", False):
            self.history["C_landauer"] = _new_history_buffer(self.history_maxlen)
            self.history["bits_erased"] = _new_history_buffer(self.history_maxlen)

    def step(
        self,
        x_e: float,
        x_i: float,
        x_hat_e: float | None = None,
        x_hat_i: float | None = None,
    ) -> dict:
        # AUD-DEF-01 / APGI-PRO-002: reject malformed inputs before ANY state,
        # statistics, RNG or history mutation, so a rejected call is atomic.
        x_e = require_finite_scalar("x_e", x_e)
        x_i = require_finite_scalar("x_i", x_i)
        x_hat_e = require_optional_finite_scalar("x_hat_e", x_hat_e)
        x_hat_i = require_optional_finite_scalar("x_hat_i", x_hat_i)
        # Use provided predictions or internalized ones per §1.4
        _x_hat_e = x_hat_e if x_hat_e is not None else self.x_hat_e
        _x_hat_i = x_hat_i if x_hat_i is not None else self.x_hat_i
        # 1) Raw prediction errors
        z_e = compute_prediction_error(x_e, _x_hat_e)
        z_i = compute_prediction_error(x_i, _x_hat_i)
        # 3) Proper z-score normalization: (z - μ) / (σ + eps)
        # Using previous stats for the current sample's normalization (standard filtering)
        eps = self.config["eps"]
        z_e_n_pre = (z_e - self.state.mu_e) / (self.state.sigma2_e**0.5 + eps)
        z_i_n_pre = (z_i - self.state.mu_i) / (self.state.sigma2_i**0.5 + eps)
        # 3a-φ) Signed nonlinear transform §6 — φ(ε) = α·tanh(γ·ε), valence-asymmetric
        _a_pos = self.config.get("alpha_plus", 1.0)
        _a_neg = self.config.get("alpha_minus", 1.0)
        _g_pos = self.config.get("gamma_plus", 2.0)
        _g_neg = self.config.get("gamma_minus", 2.0)
        phi_e_n = phi_transform(z_e_n_pre, _a_pos, _a_neg, _g_pos, _g_neg)
        # Interoceptive z-score includes dopamine bias if configured
        z_i_eff_pre = apply_dopamine_bias_to_error(z_i_n_pre, self.config["beta"])
        phi_i_eff = phi_transform(z_i_eff_pre, _a_pos, _a_neg, _g_pos, _g_neg)
        # 4) Online mean + variance update — spec Method A (EMA) or Method B
        # (sliding window), selected via config["variance_method"].
        if self.stats_e is not None and self.stats_i is not None:
            # Method B — Sliding Window (MathSpec §1): explicit window of
            # length T_win, Bessel-corrected sample variance (RunningStats
            # uses ddof=1), as the spec recommends for small T_win.
            self.stats_e.update(z_e)
            self.stats_i.update(z_i)
            self.state.mu_e = self.stats_e.mean()
            self.state.mu_i = self.stats_i.mean()
            self.state.sigma2_e = self.stats_e.variance()
            self.state.sigma2_i = self.stats_i.variance()
        else:
            # Method A (preferred) — two-step EMA
            # Step 1: μ_e(t+1) = (1-α)·μ_e(t) + α·ε_e(t)
            # Step 2: σ²_e(t+1) = (1-α)·σ²_e(t) + α·(ε_e(t) − μ_e(t))²
            # Using raw errors (not φ-transformed) for unbiased variance estimation.
            alpha_e = self.config["alpha_e"]
            alpha_i = self.config["alpha_i"]
            # MathSpec §1 literal indexing: Step 2's variance update uses μ(t),
            # the mean BEFORE Step 1's update — not μ(t+1). Capture the
            # pre-update mean here so Step 2 below matches
            # σ²(t+1) = (1-α)σ²(t) + α·(ε(t) − μ(t))² exactly.
            mu_e_prev = self.state.mu_e
            mu_i_prev = self.state.mu_i
            # Step 1: update means from raw errors — μ(t+1) = (1-α)μ(t) + α·ε(t)
            self.state.mu_e = update_mean_ema(self.state.mu_e, z_e, alpha_e)
            self.state.mu_i = update_mean_ema(self.state.mu_i, z_i, alpha_i)
            # Step 2: centered variance using the PRE-update mean μ(t) — omitting
            # centering entirely produces E[ε²], not variance; using μ(t+1)
            # instead of μ(t) is a literal (if minor) deviation from the spec's
            # indexing that this pre-update capture corrects.
            self.state.sigma2_e = (1.0 - alpha_e) * self.state.sigma2_e + alpha_e * (
                z_e - mu_e_prev
            ) ** 2
            self.state.sigma2_i = (1.0 - alpha_i) * self.state.sigma2_i + alpha_i * (
                z_i - mu_i_prev
            ) ** 2
        # 5) Precision with clamping (§7.2)
        pi_e = compute_precision(
            self.state.sigma2_e,
            self.config["eps"],
            self.config["pi_min"],
            self.config["pi_max"],
        )
        pi_i = compute_precision(
            self.state.sigma2_i,
            self.config["eps"],
            self.config["pi_min"],
            self.config["pi_max"],
        )
        self.state.pi_e = pi_e
        self.state.pi_i = pi_i
        # Final normalized signals for downstream use
        z_e_n = z_e_n_pre
        z_i_n = z_i_n_pre
        z_i_eff = z_i_eff_pre
        # 6) Neuromodulation (§8)
        # Π_min/Π_max clamp (§7.2) must be re-applied after EVERY
        # neuromodulatory gain, not just the somatic-exponential path — an
        # unbounded g_ach/g_ne (both configurable, no upper bound on the
        # gain itself) would otherwise let Π_e_eff/Π_i_eff exceed the
        # spec-mandated ceiling with nothing downstream catching it.
        pi_e_eff = clamp(
            apply_ach_gain(pi_e, self.config["g_ach"]),
            self.config["pi_min"],
            self.config["pi_max"],
        )
        # Interoceptive precision: exponential somatic form or linear NE gain
        if self.config.get("use_somatic_precision", False):
            # β_SM (exponential gain), NOT β_DA or β_somatic (linear form) —
            # see config.py's Somatic marker parameters block.
            _beta_sm = float(self.config.get("beta_sm") or self.config.get("beta_somatic") or 0.3)
            pi_i_eff = compute_interoceptive_precision_exponential(
                pi_i,
                _beta_sm,
                self.M,
                self.config["pi_min"],
                self.config["pi_max"],
            )
        elif self.config.get("ne_on_precision", False):
            pi_i_eff = clamp(
                apply_ne_gain(pi_i, self.config["g_ne"]),
                self.config["pi_min"],
                self.config["pi_max"],
            )
        else:
            pi_i_eff = pi_i
        # 5b) Hierarchical precision ODE if enabled
        if self.hierarchical_network is not None and self.config.get(
            "use_hierarchical_precision_ode", False
        ):
            dt = self.config.get("dt", 1.0)
            # Compute per-level z-scores for both channels via vectorized EMA (§7)
            z_e_levels, z_i_levels = self._compute_per_level_errors(z_e, z_i)
            # Full precision cascade: apply φ transform per level (§6, §7)
            epsilon_e_levels = phi_transform_array(z_e_levels, _a_pos, _a_neg, _g_pos, _g_neg)
            epsilon_i_levels = phi_transform_array(z_i_levels, _a_pos, _a_neg, _g_pos, _g_neg)
            epsilon_drive = epsilon_e_levels + epsilon_i_levels
            # Bottom-up ψ coupling can use a different channel than the drive term (§8)
            # Options: combined (default), extero, intero
            bu_source = self.config.get("precision_bottom_up_source", "combined")
            if bu_source == "combined":
                epsilon_bottom_up = epsilon_drive
            elif bu_source == "extero":
                epsilon_bottom_up = epsilon_e_levels
            elif bu_source == "intero":
                epsilon_bottom_up = epsilon_i_levels
            else:
                raise ValueError(
                    "precision_bottom_up_source must be one of: 'combined', 'extero', 'intero'"
                )
            # Step the hierarchical network using actual combined per-level errors
            pis_new, phi_new = self.hierarchical_network.step(
                epsilon_new=epsilon_drive,
                dt=dt,
                alpha_gain=self.config.get("alpha_gain", 0.1),
                epsilon_bottom_up=epsilon_bottom_up,
            )
            # Update hierarchical state with both precision and phase
            if self.hierarchical is None:  # pragma: no cover  # Type guard for mypy
                raise RuntimeError("hierarchical is None")
            self.hierarchical.pis = pis_new.tolist()
            self.hierarchical.phases = phi_new.tolist()
        # 5c) Update hierarchical feature states for signal aggregation
        if self.use_hierarchical:
            dt = self.config.get("dt", 1.0)
            for level in range(self.n_levels):
                # Update exteroceptive and interoceptive features at each timescale
                self.phi_e_levels[level] = update_multiscale_feature(
                    self.phi_e_levels[level], z_e_n, self.taus[level]
                )
                self.phi_i_levels[level] = update_multiscale_feature(
                    self.phi_i_levels[level], z_i_eff, self.taus[level]
                )
            # Advance oscillatory phases every step in basic hierarchical mode.
            # Without this, phases stay at 0 (cos(0)=1 constant) and the PAC
            # threshold modulation from §8 produces no oscillatory windows.
            # The precision-ODE branch already updates phases; resonance_system
            # manages its own phases internally — skip in both those cases.
            if self.hierarchical_network is None and self.resonance_system is None:
                _kappa_ph = self.config.get(
                    "kappa_phase", self.config.get("resonance_kappa_down", 0.1)
                )
                _phases = np.array(self.hierarchical.phases)  # type: ignore[union-attr]
                for _l in range(self.n_levels):
                    _dphi = self.omega_levels[_l] * dt
                    if _l < self.n_levels - 1:
                        _dphi += _kappa_ph * np.sin(_phases[_l + 1] - _phases[_l]) * dt
                    _phases[_l] = (_phases[_l] + _dphi) % (2.0 * np.pi)
                self.hierarchical.phases = _phases.tolist()  # type: ignore[union-attr]
        # 6) Instantaneous signal (diagnostic) + SDE integration via ODE drift
        # Resolve tau_s / dt / adaptive_noise_std early — needed in both hierarchical
        # and single-scale paths.
        dt = self.config.get("dt", 1.0)
        tau_s = self.config.get("tau_s", 5.0)
        adaptive_noise_std = compute_precision_coupled_noise_std(pi_e_eff, pi_i_eff)
        # Step Kuramoto oscillators here (before S integration) so Γ gates S_inst
        # per spec §12 rather than being applied post-hoc to accumulated self.S.
        # Kuramoto phase dynamics are independent of the within-step APGI state,
        # so stepping early is safe and produces the correct causal ordering.
        _kuramoto_gamma = 1.0  # default: no gating (single-scale scalar use)
        _kuramoto_gamma_per_level: np.ndarray | None = None  # hierarchical per-level use
        _kuramoto_result_cache: dict | None = None
        if self.kuramoto is not None:
            _kuramoto_result_cache = self.kuramoto.step(dt=dt)
            _kuramoto_gamma = self.kuramoto.oscillators.compute_gamma()
            # Spec §8/§12: Γ^(l)(t) is level-specific, not one global scalar
            # shared by every level — use the per-level gate for the
            # hierarchical S_inst^(l) = Π^(l)·φ(ε^(l))·Γ^(l) aggregation below.
            _kuramoto_gamma_per_level = self.kuramoto.oscillators.compute_gamma_per_level()
        # Use hierarchical aggregation if enabled, otherwise single-scale
        if self.use_hierarchical:
            # Spec §12: S_inst^(l) = Π_l · φ(ε_l) · Γ^(l)
            # Per-level φ(ε_l) is computed from per-level z-scores so that each
            # hierarchy level contributes its own instantaneous error signal.
            # phi_e_levels / phi_i_levels are the slow EMA features retained for
            # threshold-cascade computation; they are not used here.
            _z_e_lev, _z_i_lev = self._compute_per_level_errors(z_e, z_i)
            _phi_e_lev = phi_transform_array(_z_e_lev, _a_pos, _a_neg, _g_pos, _g_neg)
            _phi_i_lev = phi_transform_array(_z_i_lev, _a_pos, _a_neg, _g_pos, _g_neg)
            # Combine exteroceptive and interoceptive per-level signals, then
            # gate by Γ — elementwise per-level when Kuramoto is active
            # (Γ^(l) genuinely differs per level), else the neutral scalar 1.0.
            # Falls back to the scalar gate if the Kuramoto oscillator count
            # doesn't match the hierarchy's level count (independently
            # configurable), rather than raising a shape-mismatch error.
            if (
                _kuramoto_gamma_per_level is not None
                and len(_kuramoto_gamma_per_level) == self.n_levels
            ):
                phi_combined = (_phi_e_lev + _phi_i_lev) * _kuramoto_gamma_per_level
            else:
                phi_combined = (_phi_e_lev + _phi_i_lev) * _kuramoto_gamma
            # Use hierarchical precision if available, otherwise compute from per-level variance
            if self.hierarchical_network is not None:
                pi_levels = self.hierarchical_network.pi
            else:
                # Basic hierarchical mode: compute per-level precision from variance estimates
                # This creates multi-scale precision structure even without precision ODE
                pi_levels = np.array(
                    [
                        compute_precision(
                            self.sigma2_e_levels[i] + self.sigma2_i_levels[i],
                            self.config["eps"],
                            self.config["pi_min"],
                            self.config["pi_max"],
                        )
                        for i in range(self.n_levels)
                    ]
                )
            S_inst = aggregate_multiscale_signal_phi(phi_combined, pi_levels, self.weights)
            # Cross-Level Threshold Resonance (§8/§9): advance phases and thresholds.
            #
            # ARCHITECTURE: The resonance system owns two separate concerns:
            #   (a) Phase dynamics + threshold modulation  ← used here
            #   (b) Per-level leaky S accumulators         ← NOT used for self.S
            #
            # Why (b) is bypassed: lambda_l = dt/tau_l.  For large tau_l (e.g. tau_0=5,
            # dt=0.002 → lambda=0.0004), the leaky accumulator delivers S_inst×0.0004
            # per step — 33× slower than the ODE (dt/tau_s = 0.013).  The signal never
            # reaches threshold within a realistic simulation horizon.
            # self.S is instead accumulated via the ODE (same as single-scale) directly
            # below.  The resonance system's theta (phase-modulated) is still used for
            # the ignition check, preserving the hierarchical threshold architecture.
            if self.resonance_system is not None:
                # Pass current instantaneous signal × tau_s as S_inst so the resonance
                # system's per-level S arrays equilibrate at the same amplitude as the ODE
                # signal (S* = tau_s × Π·φ for both paths).  This keeps resonance_S_levels
                # meaningful for diagnostic outputs and the maturity cascade metric.
                # self.S is still accumulated via the ODE below — ignition is independent.
                _S_inst_diag = np.full(
                    self.n_levels, (pi_e_eff * phi_e_n + pi_i_eff * phi_i_eff) * tau_s
                )
                self.resonance_system.step(_S_inst_diag, pi_levels, dt, noise_std=0.0)
                # Sync live phases and precisions back to the hierarchical state so
                # that hierarchical_threshold_modulation uses current resonance phases.
                if self.hierarchical is None:  # pragma: no cover
                    raise RuntimeError("hierarchical is None")
                self.hierarchical.phases = self.resonance_system.phi.tolist()
                self.hierarchical.pis = self.resonance_system.pi.tolist()
        else:
            S_inst = instantaneous_signal_phi(phi_e_n, phi_i_eff, pi_e_eff, pi_i_eff)
        # Gate single-scale ODE inputs by Γ so S_inst = Π·φ(ε)·Γ per spec §12.
        _phi_e_gated = phi_e_n * _kuramoto_gamma
        _phi_i_gated = phi_i_eff * _kuramoto_gamma
        # Wire dynamics.py (update_signal_ode) OR discrete leaky accumulation
        # dt, tau_s, and adaptive_noise_std already resolved above.
        # Choose between ODE (continuous) or discrete leaky accumulation.
        # When resonance is active it handles phases/thresholds only — S is always
        # accumulated via the ODE so that ignition dynamics are parameter-independent.
        if self.resonance_system is not None:
            # ODE accumulation (resonance manages phases+thresholds, not S)
            self.S = update_signal_ode(
                self.S,
                _phi_e_gated,
                _phi_i_gated,
                pi_e_eff,
                pi_i_eff,
                tau_s,
                dt,
                adaptive_noise_std,
                rng=self.rng,
            )
        elif self.config.get("use_canonical_discrete_mode", False):
            # Discrete canonical mode: S(t+1) = (1-λ)S(t) + λ·S_inst(t)·Γ per spec §12
            lam = self.config["lam"]
            self.S = integrate_signal_leaky(self.S, S_inst * _kuramoto_gamma, lam)
        else:
            # ODE mode: dS/dt = -S/τ_S + Π_e·φ(ε_e)·Γ + Π_i·φ(ε_i)·Γ + σ·dW
            self.S = update_signal_ode(
                self.S,
                _phi_e_gated,
                _phi_i_gated,
                pi_e_eff,
                pi_i_eff,
                tau_s,
                dt,
                adaptive_noise_std,
                rng=self.rng,
            )
        self.S = stabilize_signal_log(self.S, enabled=self.config["signal_log_nonlinearity"])
        # 7) Cost/value and threshold update (canonical spec semantics)
        # Mode A: Standard allostatic threshold (default)
        # Mode B: Reservoir-as-threshold (spec-explicit alternative per §10)
        use_reservoir_threshold = self.config.get("reservoir_as_threshold", False)
        if self.config["use_realistic_cost"]:
            # Check if BOLD calibration should be used
            bold_calibration = None
            if self.config.get("use_bold_calibration", False):
                bold_calibration = {
                    "bold_signal_change": 2.0,  # Default 2% BOLD change
                    "conversion_factor": self.config.get("bold_conversion_factor", 1.2e-18),
                    "tissue_volume": self.config.get("bold_tissue_volume", 1.0),
                    "ignition_spike_factor": self.config.get("bold_ignition_spike_factor", 1.075),
                }
            C_t = compute_metabolic_cost_realistic(
                self.S,
                self.B_prev,
                self.config["c1"],
                self.config["c2"],
                enforce_landauer=self.config.get("use_thermodynamic_cost", False),
                kappa_meta=self.config.get("kappa_meta", 1e20),
                kappa_units=self.config.get("kappa_units", "dimensionless"),
                bold_calibration=(bold_calibration if bold_calibration is not None else {}),
            )
        else:
            # C(t) = c1·S(t) + c2·B(t-1) per spec (c2=0 if not set)
            C_t = compute_metabolic_cost(
                self.S, self.B_prev, self.config["c1"], self.config.get("c2", 0.0)
            )
        # Dynamical Formulation §13: C(t) = c1·S(t) + c2·B(t-1) + c3·A(t)
        # (action-execution-cost term), generalized here to the continuous
        # metabolic state M(t). c_metabolic_state=0.0 (default) is an exact
        # no-op, preserving C(t) as computed above for existing configs;
        # setting it > 0 closes active-inference channel 2's "motor actions
        # change metabolic state M, modulating θ_t via allostatic feedback"
        # loop (§19) — self.metabolic_state is updated by that channel's
        # feedback below, distinct from the somatic marker self.M.
        C_t += self.config.get("c_metabolic_state", 0.0) * self.metabolic_state
        # 7a) Thermodynamic cost validation (§11)
        C_landauer = 0.0
        bits_erased = 0.0
        if self.config.get("use_thermodynamic_cost", False):
            from core.thermodynamics import compute_information_bits

            C_landauer = compute_landauer_cost(
                self.S,
                self.config["eps"],
                k_b=self.config.get("k_boltzmann", 1.38e-23),
                T_env=self.config.get("T_env", 310.0),
                kappa_meta=self.config.get("kappa_meta", 1e20),
                kappa_units=self.config.get("kappa_units", "dimensionless"),
            )
            bits_erased = compute_information_bits(self.S, self.config["eps"])
            self.history["C_landauer"].append(C_landauer)
            self.history["bits_erased"].append(bits_erased)
        # V(t) = v1|z_e| + v2|z_i_eff| — raw prediction errors per spec
        V_t = compute_information_value(z_e_n, z_i_eff, self.config["v1"], self.config["v2"])
        # 7b/8) Threshold and Ignition: Standard or Reservoir-as-Threshold mode
        # Spec §10: Reservoir can serve as alternative execution path
        # Set True only when the ODE-drift branch below actually folds
        # δ·B(t-1) into its drift term, so step 9's discrete boost knows to
        # stand down and avoid double-applying the same refractory term.
        ode_drift_owns_refractory = False
        if use_reservoir_threshold:
            # Mode B: Reservoir-as-threshold (spec-explicit alternative)
            # Reservoir dynamics replace allostatic threshold computation
            if self.reservoir is None:
                raise ValueError("Reservoir mode enabled but reservoir not initialized")
            u_res = np.array([z_e_n, z_i_eff])
            self.reservoir.step(
                u_res,
                tau=self.config.get("reservoir_tau", 1.0),
                dt=dt,
                precision=pi_e_eff,
                S_target=self.S,
                theta=self.theta,
                A_amp=self.config.get("reservoir_amplification", 0.0),
            )
            # Reservoir readout serves as effective threshold
            S_reservoir = self.reservoir.readout(
                method=self.config.get("reservoir_readout_method", "linear")
            )
            # Map reservoir signal to effective threshold
            theta_next = self.config["theta_base"] * (
                1.0 + self.config.get("reservoir_theta_scale", 0.1) * S_reservoir
            )
            # Skip allostatic update - reservoir provides dynamics
            p_ignite = compute_ignition_probability(self.S, theta_next, self.config["ignite_tau"])
            if self.config["stochastic_ignition"]:
                B_t = sample_ignition_state(p_ignite, rng=self.rng)
            else:
                B_t = int(detect_ignition_event(self.S, theta_next))
        else:
            # Mode A: Standard allostatic threshold (canonical spec semantics)
            # 7b) Threshold update: discrete or continuous ODE per APGI spec §4
            # Canonical semantics: θ(t+1) = θ(t) + η[C(t) - V(t)]
            # Refractory boost δ·B(t) applied AFTER ignition (step 9) per spec §6
            if self.config.get("use_continuous_threshold_ode", False):
                # Continuous ODE form. By default, refractory boost (δ·B(t)) is
                # handled post-ignition (step 9). When use_ode_refractory_drift=True
                # the impulse is included inside the ODE drift for closer alignment
                # with the analytic form in §7.4: dθ/dt = … + δ_reset·B(t)
                if self.config.get("use_ode_refractory_drift", False):
                    _ode_delta = self.config["delta"]
                    _ode_B = self.B_prev
                    ode_drift_owns_refractory = True
                else:
                    _ode_delta = 0.0
                    _ode_B = 0
                theta_next = allostatic_threshold_ode(
                    theta=self.theta,
                    theta_0=self.config["theta_base"],
                    gamma=1.0 / self.config.get("tau_theta", 1000.0),
                    B_prev=_ode_B,
                    delta=_ode_delta,
                    C=C_t,
                    V=V_t,
                    eta=self.config["eta"],
                    dt=dt,
                    noise_std=self.config.get("noise_std", 0.01),
                    rng=self.rng,
                )
            else:
                # Discrete form: θ_next = θ + η(C-V) [refractory in step 9]
                theta_next = update_threshold_discrete(
                    self.theta,
                    C_t,
                    V_t,
                    self.config["eta"],
                    delta=0.0,  # Refractory handled in step 9
                    B_prev=0,  # Refractory handled in step 9
                )
            # 7b-decay) Step 12 — mean reversion & exponential decay.
            #   θ(t+1) ← θ₀ + (θ(t+1) − θ₀)·exp(−γ_θ)
            # MathSpec §13 orders this AFTER the allostatic update (step 11)
            # and BEFORE the NE modulation (13), the 5-HT offset (14) and the
            # ignition decision (15). It previously ran at the very END of the
            # timestep, so the ignition test at step 15 saw an un-decayed
            # threshold. The two orderings are NOT equivalent: decay is affine
            # (θ ↦ θ₀ + (θ−θ₀)e^{−γ}) and the allostatic update is a
            # translation, and affine maps do not commute with translations —
            # deferring the decay scales the cost-benefit increment by an
            # extra e^{−γ_θ} and applies the NE/5-HT offsets to a threshold
            # one decay-step out of phase.
            theta_next = threshold_decay(
                theta_next, self.config["theta_base"], self.config["kappa"]
            )
            # 7c) Step 13 — NE multiplicative threshold modulation (§4.4).
            # g_NE has already been clamped so that γ_NE·g_NE ≤ 0.5.
            if self.config.get("ne_on_threshold", False):
                theta_next = apply_ne_threshold_modulation(
                    theta_next, self.config["g_ne"], self.config["gamma_ne"]
                )
            # 7e) Serotonergic threshold offset: θ_eff = θ + β_5HT (spec §8.4)
            # 5-HT encodes patience/uncertainty-tolerance; β_5HT > 0 raises ignition bar.
            beta_5ht = self.config.get("beta_5ht", 0.0)
            if beta_5ht != 0.0:
                theta_next = apply_serotonin_threshold_offset(theta_next, beta_5ht)
            # 7f) Circadian/ultradian threshold modulation (§27, peripheral —
            # Severity C: a null result here leaves the core framework
            # intact). Additive offset, same position as the 5-HT offset
            # above (after the allostatic/NE update, before hierarchical
            # modulation and the ignition decision). core/circadian.py's
            # CircadianRegulator was previously unwired (dead code).
            if self.circadian_regulator is not None:
                theta_next = theta_next + self.circadian_regulator.theta_offset()
                self.circadian_regulator.tick()
            # 7d) Hierarchical threshold modulation (PAC + Cascade) if enabled (§8.4)
            # Apply to ALL hierarchical modes for consistency
            if self.hierarchical is not None:
                # Prepare inputs for hierarchical threshold computation.
                # Level 0's baseline is the allostatically-updated theta_next
                # (already includes the C(t)/V(t) update, NE and 5-HT
                # corrections computed above) — per MathSpec §4/§8 the PAC
                # formula modulates a baseline that is itself governed by the
                # allostatic ODE; silently substituting the static config
                # theta_base here would discard the allostatic update
                # whenever hierarchical mode is active. Other levels have no
                # independent C/V tracking in this implementation, so they
                # keep the static baseline. The global clip below (§7.4)
                # still bounds the composed result, so this does not
                # reintroduce unbounded growth.
                theta_0_levels = np.ones(self.n_levels) * self.config["theta_base"]
                theta_0_levels[0] = theta_next
                # Level-specific signals for bottom-up cascade (if enabled)
                S_levels = np.zeros(self.n_levels)
                S_levels[0] = self.S
                if self.n_levels > 1:
                    S_levels[1:] = phi_transform_array(
                        self.phi_e_levels[1:], _a_pos, _a_neg, _g_pos, _g_neg
                    ) + phi_transform_array(self.phi_i_levels[1:], _a_pos, _a_neg, _g_pos, _g_neg)
                # Use hierarchical precision if available, otherwise compute from per-level variance
                if self.hierarchical_network is not None:
                    _pi_levels = self.hierarchical_network.pi
                else:
                    _pi_levels = np.array(
                        [
                            compute_precision(
                                self.sigma2_e_levels[i] + self.sigma2_i_levels[i],
                                self.config["eps"],
                                self.config["pi_min"],
                                self.config["pi_max"],
                            )
                            for i in range(self.n_levels)
                        ]
                    )
                if self.resonance_system is not None:
                    # Feed the allostatically-updated threshold in as the
                    # level-0 baseline, then read the PAC-modulated result.
                    self.resonance_system.set_baseline(0, theta_0_levels[0])
                    theta_next = self.resonance_system.primary_threshold
                    self.hierarchical.thetas = self.resonance_system.theta.tolist()
                else:
                    # Compute full hierarchical threshold set using modular component
                    from hierarchy.coupling import bottom_up_threshold_cascade
                    from oscillation.threshold_modulation import hierarchical_threshold_modulation

                    thetas_mod = hierarchical_threshold_modulation(
                        thetas=theta_0_levels,
                        pis=_pi_levels,
                        phases=np.array(self.hierarchical.phases),
                        kappa_down=self.config.get("kappa_phase", 0.1),
                    )
                    # Apply bottom-up cascade if enabled
                    if self.config.get("kappa_up", 0.0) > 0:
                        for level in range(1, self.n_levels):
                            thetas_mod[level] = bottom_up_threshold_cascade(
                                theta_ell=thetas_mod[level],
                                S_ell_minus_1=S_levels[level - 1],
                                theta_ell_minus_1=thetas_mod[level - 1],
                                kappa_up=self.config.get("kappa_up", 0.0),
                            )
                    # Use the modulated threshold for the primary ignition level (level 0)
                    theta_next = float(thetas_mod[0])
                    self.hierarchical.thetas = thetas_mod.tolist()
            # Global threshold clamping for stability (§15). Bounds are
            # externalized to config so sensitivity analyses can widen them
            # explicitly rather than editing a hardcoded literal.
            theta_next = float(
                np.clip(
                    theta_next,
                    self.config.get("theta_clip_min", 0.1),
                    self.config.get("theta_clip_max", 20.0),
                )
            )
            # 8) Step 15 — ignition decision (§5). Draws from the pipeline's
            # own generator so the ignition train is reproducible from the seed.
            p_ignite = compute_ignition_probability(self.S, theta_next, self.config["ignite_tau"])
            if self.config["stochastic_ignition"]:
                B_t = sample_ignition_state(p_ignite, rng=self.rng)
            else:
                B_t = int(detect_ignition_event(self.S, theta_next))
        # 8b) Compute ignition margin using pre-reset values (§14)
        # Must compute BEFORE signal reset and refractory boost
        # Margin at ignition decision: Δ(t) = S(t) - θ(t) (pre-refractory)
        S_at_ignition = self.S  # Save pre-reset signal
        theta_at_ignition = theta_next  # Save pre-refractory threshold
        ignition_margin = S_at_ignition - theta_at_ignition
        # Post-ignition signal reset (§6)
        # Spec: S ← ρ_S·S on ignition, ρ_S ∈ [0.1, 0.5] (attentional blink bounds)
        if B_t == 1:
            reset_factor = self.config.get("reset_factor", 0.1)
            if not (0.1 <= reset_factor <= 0.5):
                raise ValueError(
                    f"reset_factor (ρ_S) must be in [0.1, 0.5] per spec, got {reset_factor}. "
                    "ρ_S < 0.1: near-complete erasure (incompatible with working memory). "
                    "ρ_S > 0.5: insufficient reset (attentional blink > 500 ms)."
                )
            self.S = self.S * reset_factor
            # Resonance-level ignition reset (§17): refractory boost on resonance theta.
            # self.S was already reset via reset_factor above; resonance.primary_signal
            # is NOT used (S is ODE-accumulated), so only the theta boost is applied.
            if self.resonance_system is not None:
                delta_ref_val = self.config.get("DELTA_RESET", self.config.get("delta", 0.5)) or 0.5
                self.resonance_system.apply_level_ignition(
                    level=0,
                    rho_S=float(self.config.get("RHO_RETAIN", reset_factor) or reset_factor),
                    delta_refractory=float(delta_ref_val),
                )
        # 9) Post-ignition threshold dynamics (§6)
        # Canonical spec semantics:
        #   9a) Refractory boost: θ ← θ + δ·B(t)  [using CURRENT B_t]
        #   9b) Decay to baseline: θ ← θ_base + (θ - θ_base)·e^{-κ}
        # 9a is SKIPPED when use_ode_refractory_drift=True: in that mode the
        # ODE branch above already folds δ·B(t-1) into its drift term every
        # step (see 7b). Applying this discrete boost as well would add
        # δ·B(t-1)'s contribution twice — once via the ODE drift at step t,
        # and again via this line at step t-1 (self.theta already carries
        # that boost into step t before the ODE runs). The two mechanisms are
        # mutually exclusive alternate formulations of the same refractory
        # term, never additive.
        if not ode_drift_owns_refractory:
            theta_next = apply_refractory_boost(theta_next, B_t, self.config["delta"])
        # NOTE: the mean-reversion decay is NOT applied here. It now runs at
        # step 7b-decay, before the ignition decision, per MathSpec §13's
        # step-12-before-step-15 ordering. The δ_reset boost applied above is
        # carried into the next timestep, where that decay begins relaxing it
        # toward θ₀ — which is exactly the refractory recovery the spec
        # describes ("the exponential decay then brings the elevated threshold
        # back toward θ₀ over subsequent timesteps").
        # 10) Internal Prediction Update per §1.4 Generative Model Dynamics
        # Support both flag names for backward compatibility
        use_generative_update = self.config.get(
            "use_generative_model_update", False
        ) or self.config.get("use_internal_predictions", False)
        if use_generative_update:
            self.x_hat_e = update_prediction(
                self.x_hat_e,
                z_e,
                pi_e,
                self.config.get("kappa_e", 0.01),
                self.config["pi_max"],
            )
            self.x_hat_i = update_prediction(
                self.x_hat_i,
                z_i,
                pi_i,
                self.config.get("kappa_i", 0.01),
                self.config["pi_max"],
            )
        # 11) Reservoir layer update (§10) - add-on mode only
        # When reservoir_as_threshold=True, reservoir is handled in step 7b/8
        S_reservoir = 0.0
        if self.reservoir is not None and not use_reservoir_threshold:
            u_res = np.array([z_e_n, z_i_eff])
            self.reservoir.step(
                u_res,
                tau=self.config.get("reservoir_tau", 1.0),
                dt=dt,
                precision=pi_e_eff,
                S_target=self.S,
                theta=self.theta,
                A_amp=self.config.get("reservoir_amplification", 0.0),
            )
            S_reservoir = self.reservoir.readout(
                method=self.config.get("reservoir_readout_method", "linear")
            )
            if self.config.get("reservoir_replaces_signal", False):
                # MathSpec §13 step 15: "Reservoir readout (optional):
                # S(t) = W_out^T·x(t); REPLACES the explicit accumulation of
                # Steps 8-9 when active." True replacement — not an
                # additive term — matching the single-scale spec form. This
                # is a distinct mode from the default below, which instead
                # implements the Dynamical Formulation §17 multi-level
                # formula S_global(t) = Σ_l w_l·S_inst^(l)(t) + w_res·S_res(t),
                # where the reservoir is one additive term alongside the
                # hierarchical levels rather than a full substitute.
                self.S = S_reservoir
            else:
                # Dynamical Formulation §17: S_global(t) = Σ_l w_l·S_inst^(l)(t) + w_res·S_res(t)
                # Integrate reservoir contribution into the main signal accumulator.
                w_res = self.config.get("reservoir_weight", 0.1)
                self.S = self.S + w_res * S_reservoir
        # 18) Active Inference Action Loop (§19) — Step 18 of APGI Formulation.
        # Runs AFTER all signal-processing steps (including reservoir readout) so
        # that policy selection sees the fully updated state.  Fires on ignition
        # or every step depending on ai_on_ignition_only config.
        # Closes the perception-action cycle through three feedback channels:
        #   channel 1: Δx̂ₑ (sensory, φ-weighted)  → prediction shifted by φ(z)
        #   channel 2: ΔM  (metabolic)             → somatic marker updated
        #   channel 3: ΔΣ  (epistemic)             → sigma2 reduced, Πₜ raised next step
        ai_result: dict | None = None
        if self.active_inference_agent is not None:
            _ai_on_ignition_only = self.config.get("ai_on_ignition_only", True)
            _should_run_ai = (not _ai_on_ignition_only) or (B_t == 1)
            if _should_run_ai:
                policy = self.active_inference_agent.select_policy(
                    sigma2_e=self.state.sigma2_e,
                    sigma2_i=self.state.sigma2_i,
                    S=self.S,
                    theta=theta_next,
                )
                feedback = self.active_inference_agent.apply_action_feedback(
                    action_idx=policy.action_idx,
                    z_e=z_e,
                    z_i=z_i,
                    sigma2_e=self.state.sigma2_e,
                    sigma2_i=self.state.sigma2_i,
                    M=self.M,
                )
                # Apply channel 1: sensory prediction shift (φ-weighted, valence-asymmetric)
                self.x_hat_e += feedback.delta_x_hat_e
                self.x_hat_i += feedback.delta_x_hat_i
                # Apply channel 2: somatic-marker change (feeds interoceptive precision)
                self.M = float(np.clip(self.M + feedback.delta_M, -2.0, 2.0))
                # Apply channel 2 (metabolic-state branch): distinct continuous
                # M(t) that feeds C(t) -> theta_t next step (see c_metabolic_state
                # above) — this is the pathway §19 describes as "modulating
                # θ_t via allostatic feedback", separate from the somatic
                # marker's effect on precision.
                self.metabolic_state = float(
                    np.clip(self.metabolic_state + feedback.delta_metabolic_state, 0.0, 1.0)
                )
                # Apply channel 3: epistemic precision update (floor at near-zero)
                self.state.sigma2_e = max(self.state.sigma2_e + feedback.delta_sigma2_e, 1e-6)
                self.state.sigma2_i = max(self.state.sigma2_i + feedback.delta_sigma2_i, 1e-6)
                ai_result = {
                    "ai_action_idx": policy.action_idx,
                    "ai_action_label": policy.action_label,
                    # EFE (Expected Free Energy) is distinct from variational
                    # free energy F — "ai_efe_values" is the spec-correct
                    # name; "ai_F_values" kept as a deprecated alias for
                    # backward compatibility with existing consumers of the
                    # result dict.
                    "ai_efe_values": policy.efe_values,
                    "ai_F_values": policy.efe_values,
                    "ai_p_policies": policy.p_policies,
                    "ai_expected_free_energy": policy.expected_free_energy,
                    "ai_delta_x_hat_e": feedback.delta_x_hat_e,
                    "ai_delta_M": feedback.delta_M,
                    "ai_delta_sigma2_e": feedback.delta_sigma2_e,
                }
        self.theta = theta_next
        self.B_prev = B_t
        self.t += self.config.get("dt", 1.0)
        # Update history for validation
        self.history["S"].append(self.S)
        self.history["theta"].append(self.theta)
        self.history["B"].append(float(B_t))
        result = {
            "z_e": z_e,
            "z_i": z_i,
            "z_e_norm": z_e_n,
            "z_i_norm": z_i_n,
            "mu_e": self.state.mu_e,
            "mu_i": self.state.mu_i,
            "pi_e": pi_e,
            "pi_i": pi_i,
            "pi_e_eff": pi_e_eff,
            "pi_i_eff": pi_i_eff,
            "z_i_eff": z_i_eff,
            "S_inst": S_inst,
            "S": self.S,
            "C": C_t,
            "V": V_t,
            "theta": self.theta,
            "ignition_margin": ignition_margin,
            "p_ignite": p_ignite,
            "B": B_t,
            "theta_dot": self.theta_dot,
            "x_hat_e": self.x_hat_e,
            "x_hat_i": self.x_hat_i,
            "M_somatic": self.M,
            "metabolic_state": self.metabolic_state,
        }
        # Add thermodynamic info if enabled
        if self.config.get("use_thermodynamic_cost", False):
            result["C_landauer"] = C_landauer
            result["bits_erased"] = bits_erased
        # Add BOLD calibration info if enabled
        if self.config.get("use_bold_calibration", False) and bold_calibration is not None:
            result["bold_calibration"] = bold_calibration
        # Add reservoir info if enabled
        if self.reservoir is not None:
            result["S_reservoir"] = S_reservoir
            result["reservoir_state_norm"] = float(np.linalg.norm(self.reservoir.x))
        # Add Kuramoto oscillator info if enabled (§9)
        # Oscillators were already stepped before S integration (above); use cached result.
        if self.kuramoto is not None and _kuramoto_result_cache is not None:
            result["kuramoto_phases"] = _kuramoto_result_cache["phases"].tolist()
            result["kuramoto_synchronization"] = _kuramoto_result_cache["synchronization"]
            # Γ was already applied inside S_inst (not retroactively on accumulated self.S)
            result["kuramoto_gamma"] = _kuramoto_gamma
            if _kuramoto_gamma_per_level is not None:
                result["kuramoto_gamma_per_level"] = _kuramoto_gamma_per_level.tolist()
            # Apply phase reset on ignition
            if B_t == 1:
                # Broadcast phase reset across the full hierarchy when enabled.
                # kuramoto_broadcast_ignition=True triggers ignition-induced phase
                # resets at all levels with hierarchical distance decay (§9.2).
                broadcast_ignition = self.config.get("kuramoto_broadcast_ignition", False)
                if broadcast_ignition:
                    # Full-hierarchy broadcast: reset every level with decay
                    for _lvl in range(self.kuramoto.n_levels):
                        self.kuramoto.apply_ignition_reset(
                            level=_lvl,
                            broadcast=True,
                        )
                else:
                    # Default: reset only primary level 0
                    self.kuramoto.apply_ignition_reset(level=0)
        # Add observable mapping if enabled (§14)
        # Use pre-reset values for correct margin computation
        if self.neural_observables is not None:
            neural_obs = self.neural_observables.step(S_at_ignition, theta_at_ignition, B_t)
            result["neural_gamma_power"] = neural_obs["gamma_power"]
            result["neural_erp_amplitude"] = neural_obs["erp_amplitude"]
            result["neural_ignition_rate"] = neural_obs["ignition_rate"]
        if self.behavioral_observables is not None:
            behavioral_obs = self.behavioral_observables.step(S_at_ignition, theta_at_ignition, B_t)
            result["behavioral_rt_variability"] = behavioral_obs["rt_variability"]
            result["behavioral_response_criterion"] = behavioral_obs["response_criterion"]
            result["behavioral_decision_rate"] = behavioral_obs["decision_rate"]
        if self.prediction_validator is not None:
            pred_result = self.prediction_validator.step(S_at_ignition, theta_at_ignition, B_t)
            result["prediction_margin"] = pred_result["delta"]
            result["prediction_p_ign"] = pred_result["p_ign"]
        # Add active inference output if enabled (§19)
        if ai_result is not None:
            result.update(ai_result)
        # Add stability analysis if enabled (§7)
        if self.stability_analyzer is not None:
            self.stability_analyzer.step(self.S, self.theta)
        if self.hierarchical is not None:
            result["hierarchical_pis"] = self.hierarchical.pis.copy()
            result["hierarchical_phases"] = self.hierarchical.phases.copy()
            result["hierarchical_thetas"] = self.hierarchical.thetas.copy()
        if self.resonance_system is not None:
            result["resonance_phases"] = self.resonance_system.phi.tolist()
            result["resonance_thetas"] = self.resonance_system.theta.tolist()
            result["resonance_S_levels"] = self.resonance_system.S.tolist()
            result["resonance_ignition_windows"] = self.resonance_system.ignition_windows.tolist()
            result["resonance_modulation_depth"] = self.resonance_system.modulation_depth.tolist()
        return result

    def validate(self) -> dict:
        """Perform statistical validation on the accumulated history.

        Returns:
            A dict whose ``status`` is ``"insufficient_data"``, ``"success"``
            (every analysis ran) or ``"partial"`` (one or more sub-analyses
            raised; they are listed in ``failed_analyses`` and ``certifiable``
            is False). Exceptions in the core Hurst/PSD analyses propagate.
        """
        if len(self.history["theta"]) < 64:
            return {"status": "insufficient_data"}
        # Validate threshold 1/f / fractal dynamics (§12/§22)
        theta_arr = np.array(self.history["theta"])
        fs = 1.0 / self.config.get("dt", 1.0)
        # Estimate Hurst exponent (robust spectral + DFA depending on implementation)
        hurst_val = estimate_hurst_robust(theta_arr, fs=fs)
        # Compute PSD once for downstream validators
        from stats.hurst import welch_periodogram

        freqs, psd = welch_periodogram(theta_arr, fs=fs)
        pink_stats = validate_pink_noise(freqs, psd)
        # Lorentzian superposition validation (§12)
        lorentzian_stats: dict | None = None
        if self.use_hierarchical:
            try:
                from stats.spectral_model import fit_lorentzian_superposition

                # Reuse the timescales actually used to build the hierarchy
                # (self.taus), rather than recomputing from config — the two
                # would silently diverge under hierarchical_timescale_mode="canonical".
                taus_ms = self.taus
                # Use seconds for spectral model
                fit = fit_lorentzian_superposition(freqs, psd, taus_ms / 1000.0)
                lorentzian_stats = {
                    "r_squared": fit.get("r_squared", 0.0),
                    "r_squared_log": fit.get("r_squared_log", 0.0),
                    "fit_method": fit.get("fit_method", "unknown"),
                }
            except Exception as e:
                lorentzian_stats = {"status": "error", "error": str(e)}
        # Avalanche power-law validation (spec target exponent ~1.5) (§22)
        avalanche_stats: dict | None = None
        if "B" in self.history and len(self.history["B"]) >= 64:
            try:
                from stats.avalanche import validate_avalanche_power_law

                avalanche_stats = validate_avalanche_power_law(
                    np.asarray(self.history["B"]),
                    alpha_target=1.5,
                    tolerance=float(self.config.get("avalanche_alpha_tolerance", 0.3)),
                    xmin=int(self.config.get("avalanche_xmin", 1)),
                    min_avalanches=int(self.config.get("avalanche_min_avalanches", 30)),
                )
            except Exception as e:
                avalanche_stats = {"status": "error", "error": str(e)}
        # AUD-DEF-03 / APGI-INT-003: a sub-analysis that raised must not be
        # reported under an overall "success". The outcome is "partial" when
        # the core analyses completed but an optional one failed; consumers
        # (certification, reports) must treat anything but "success" as not
        # certifiable.
        failed = [
            name
            for name, stats in (
                ("lorentzian_superposition", lorentzian_stats),
                ("avalanche_power_law", avalanche_stats),
            )
            if isinstance(stats, dict) and stats.get("status") == "error"
        ]
        return {
            "status": "partial" if failed else "success",
            "failed_analyses": failed,
            "certifiable": not failed,
            "hurst_exponent": hurst_val,
            "is_pink_noise": pink_stats["is_pink_noise"],
            "beta": pink_stats["beta"],
            "data_points": len(theta_arr),
            "lorentzian_superposition": lorentzian_stats,
            "avalanche_power_law": avalanche_stats,
        }
