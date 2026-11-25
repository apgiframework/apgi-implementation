"""Fixed-point stability analysis for APGI system.
Implements Jacobian computation and eigenvalue analysis per spec §7.5.
Discrete system at no-ignition fixed point (affective paradigm):
    S* = (1-λ)S* + λ·Π·φ(z)  →  S* = Π·φ(z)/(λ)
    θ* = θ* + η[C - V]        →  C = V at equilibrium
where φ(ε) = α·tanh(γ·ε) is the asymmetric valence-specific transform (§6).
Jacobian: J = [[1-λ, 0], [ηc₁λ, e^{-κ}]]
Stability requires: |λ₁| < 1 and |λ₂| < 1
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from core.deprecation import RenamedKeyDict, renamed_kwargs
from core.numerics import safe_corrcoef
from core.phi_transform import phi_transform

# Suppress LAPACK warnings
warnings.filterwarnings("ignore", message=".*On entry to DLASCL.*")


@renamed_kwargs(kappa="decay_exponent")
def compute_jacobian_discrete(
    lam: float,
    decay_exponent: float,
    c1: float,
    eta: float,
) -> np.ndarray:
    """Compute Jacobian of discrete APGI system.
    At the no-ignition fixed point, the system is:
        S(t+1) = (1-λ)S(t) + λ·Π·φ(z)
        θ(t+1) = θ(t) + η[C(t) - V(t)] + e^{-d}(θ(t) - θ_base),  d = γ_θ·dt
    Linearizing around fixed point:
        J = [[∂S'/∂S, ∂S'/∂θ],
             [∂θ'/∂S, ∂θ'/∂θ]]
    Spec §7.5: J = [[1-λ, 0], [ηc₁λ, e^{-d}]]
    Args:
        lam: Signal integration rate λ ∈ (0,1)
        decay_exponent: Per-step threshold decay exponent d = γ_θ·dt > 0 (formerly ``kappa``)
        c1: Cost coefficient c₁ > 0
        eta: Threshold learning rate η > 0
    Returns:
        2x2 Jacobian matrix
    """
    J = np.array(
        [
            [1.0 - lam, 0.0],
            [eta * c1 * lam, np.exp(-decay_exponent)],
        ]
    )
    return J


def compute_eigenvalues(J: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute eigenvalues and eigenvectors of Jacobian.
    Args:
        J: Jacobian matrix
    Returns:
        Tuple of (eigenvalues, eigenvectors)
    """
    try:
        with np.errstate(all="ignore"):
            eigs, vecs = np.linalg.eig(J)
            return eigs, vecs
    except (np.linalg.LinAlgError, FloatingPointError):
        # Fallback: return identity-based eigenvalues for 2x2 matrix
        eigs = np.array([J[0, 0], J[1, 1]])
        vecs = np.eye(2)
        return eigs, vecs


def check_stability(
    config: dict,
    verbose: bool = False,
) -> dict[str, Any]:
    """Check fixed-point stability against spec constraints.
    Spec §7.5 Stability Conditions:
    - |λ₁| = 1 - λ < 1  ⟺  λ > 0 ✓ (always satisfied)
    - |λ₂| = e^{-κ} < 1  ⟺  κ > 0 ✓ (always satisfied)
    For stability, both eigenvalues must have magnitude < 1.
    Args:
        config: Configuration dictionary
        verbose: Print detailed analysis
    Returns:
        Dictionary with stability analysis results
    """
    # Extract parameters
    lam = config.get("lam", 0.2)
    from config import decay_exponent

    d = decay_exponent({"tau_theta": 20.0, "dt": 0.01, **config})  # γ_θ·dt = dt/τ_θ
    c1 = config.get("c1", 0.2)
    eta = config.get("eta", 0.1)
    # Compute Jacobian
    J = compute_jacobian_discrete(lam, d, c1, eta)
    # Compute eigenvalues
    eigs, vecs = compute_eigenvalues(J)
    # Check stability
    eig_magnitudes = np.abs(eigs)
    stable = np.all(eig_magnitudes < 1.0)
    max_eig = np.max(eig_magnitudes)
    # Compute stability margin (how close to instability)
    stability_margin = 1.0 - max_eig
    if verbose:
        print("\n" + "=" * 60)
        print("APGI Fixed-Point Stability Analysis (Spec §7.5)")
        print("=" * 60)
        print("\nParameters:")
        print(f"  λ (integration rate):     {lam:.4f}")
        print(f"  d = γ_θ·dt (decay exp.):  {d:.4f}")
        print(f"  c₁ (cost coefficient):    {c1:.4f}")
        print(f"  η (learning rate):        {eta:.4f}")
        print("\nJacobian at no-ignition fixed point:")
        print(f"  J = [[{J[0, 0]:.4f}, {J[0, 1]:.4f}],")
        print(f"       [{J[1, 0]:.4f}, {J[1, 1]:.4f}]]")
        print("\nEigenvalues:")
        for i, eig in enumerate(eigs):
            print(f"  λ_{i + 1} = {eig:.4f} (|λ| = {np.abs(eig):.4f})")
        print("\nStability:")
        print(f"  Max eigenvalue magnitude: {max_eig:.4f}")
        print(f"  Stability margin:         {stability_margin:.4f}")
        print(f"  Status:                   {'STABLE ✓' if stable else 'UNSTABLE ✗'}")
        print("\nConstraint Verification (Spec §7.5):")
        print(f"  λ > 0:                    {lam > 0} ✓")
        print(f"  d > 0:                    {d > 0} ✓")
        print(f"  |λ₁| = 1-λ < 1:           {1 - lam < 1} ✓")
        print(f"  |λ₂| = e^(-d) < 1:        {np.exp(-d) < 1} ✓")
        print("=" * 60 + "\n")
    return {
        "jacobian": J,
        "eigenvalues": eigs,
        "eigenvectors": vecs,
        "eigenvalue_magnitudes": eig_magnitudes,
        "stable": bool(stable),
        "max_eigenvalue": float(max_eig),
        "stability_margin": float(stability_margin),
        "constraints_satisfied": RenamedKeyDict(
            {
                "lambda_positive": bool(lam > 0),
                "decay_exponent_positive": bool(d > 0),
                "lambda1_stable": bool(1 - lam < 1),
                "lambda2_stable": bool(np.exp(-d) < 1),
            },
            {"kappa_positive": "decay_exponent_positive"},
        ),
    }


def compute_fixed_point(
    config: dict,
) -> dict[str, float]:
    """Compute fixed point of the system.
    At equilibrium (no ignition):
        S* = Π·φ(z) / λ  (from S = (1-λ)S + λ·Π·φ(z), §6)
        θ* = θ_base      (from θ = θ + η[C - V] with C = V)
    Args:
        config: Configuration dictionary
    Returns:
        Dictionary with fixed point values
    """
    # Signal fixed point depends on precision and φ-transformed error magnitude
    pi_typical = 1.0
    z_typical = 1.0
    lam = config.get("lam", 0.2)
    phi_z = phi_transform(
        z_typical,
        alpha_pos=config.get("alpha_plus", 1.0),
        alpha_neg=config.get("alpha_minus", 1.0),
        gamma_pos=config.get("gamma_plus", 2.0),
        gamma_neg=config.get("gamma_minus", 2.0),
    )
    S_star = (pi_typical * phi_z) / (lam + 1e-8)
    # Threshold fixed point
    theta_base = config.get("theta_base", 1.0)
    theta_star = theta_base
    return {
        "S_star": float(S_star),
        "theta_star": float(theta_star),
    }


def analyze_bifurcation(
    config: dict,
    param_name: str,
    param_range: tuple[float, float],
    n_points: int = 50,
) -> dict[str, Any]:
    """Analyze bifurcation as parameter varies.
    Computes stability and eigenvalues across parameter range
    to identify bifurcation points.
    Args:
        config: Base configuration
        param_name: Parameter to vary (e.g., "lam", "tau_theta")
        param_range: (min, max) values for parameter
        n_points: Number of points to sample
    Returns:
        Dictionary with bifurcation analysis
    """
    param_values = np.linspace(param_range[0], param_range[1], n_points)
    eigenvalues_list = []
    stability_list = []
    for param_val in param_values:
        # Create modified config
        cfg = config.copy()
        cfg[param_name] = param_val
        # Compute stability
        result = check_stability(cfg, verbose=False)
        eigenvalues_list.append(result["eigenvalue_magnitudes"])
        stability_list.append(result["stable"])
    eigenvalues_array = np.array(eigenvalues_list)
    stability_array = np.array(stability_list)
    # Find bifurcation points (transitions in stability)
    stability_changes = np.diff(stability_array.astype(int))
    bifurcation_indices = np.where(stability_changes != 0)[0]
    bifurcation_params = []
    if len(bifurcation_indices) > 0:
        for idx in bifurcation_indices:
            bifurcation_params.append(float(param_values[idx]))
    return {
        "parameter_name": param_name,
        "parameter_values": param_values.tolist(),
        "eigenvalue_magnitudes": eigenvalues_array.tolist(),
        "stability": stability_array.tolist(),
        "bifurcation_points": bifurcation_params,
        "stable_region": {
            "min": float(param_values[0]) if stability_array[0] else None,
            "max": float(param_values[-1]) if stability_array[-1] else None,
        },
    }


def validate_system_dynamics(
    config: dict,
    S_history: np.ndarray,
    theta_history: np.ndarray,
) -> dict[str, Any]:
    """Validate system dynamics against theoretical predictions.
    Compares observed dynamics with linearized predictions
    near fixed point.
    Args:
        config: Configuration dictionary
        S_history: Signal history
        theta_history: Threshold history
    Returns:
        Dictionary with validation results
    """
    if len(S_history) < 100:
        return {
            "valid": False,
            "reason": "Insufficient data (need >= 100 samples)",
        }
    # Compute Jacobian
    J = compute_jacobian_discrete(
        config.get("lam", 0.2),
        _decay_exponent_of(config),
        config.get("c1", 0.2),
        config.get("eta", 0.1),
    )
    # Compute fixed point
    fp = compute_fixed_point(config)
    S_star = fp["S_star"]
    theta_star = fp["theta_star"]
    # Compute deviations from fixed point
    S_dev = S_history - S_star
    theta_dev = theta_history - theta_star
    # Compute observed dynamics
    S_dev_next = S_dev[1:]
    theta_dev_next = theta_dev[1:]
    S_dev_curr = S_dev[:-1]
    theta_dev_curr = theta_dev[:-1]
    # Predict next state using Jacobian
    state_curr = np.array([S_dev_curr, theta_dev_curr])
    state_pred = J @ state_curr
    # Compute prediction error
    state_obs = np.array([S_dev_next, theta_dev_next])
    prediction_error = np.mean((state_pred - state_obs) ** 2)
    # Compute correlation between predicted and observed
    corr_S = safe_corrcoef(state_pred[0], state_obs[0])
    corr_theta = safe_corrcoef(state_pred[1], state_obs[1])
    return {
        "valid": True,
        "fixed_point": fp,
        "prediction_error": float(prediction_error),
        "correlation_S": float(corr_S),
        "correlation_theta": float(corr_theta),
        "mean_correlation": float((corr_S + corr_theta) / 2),
        "linearization_valid": float((corr_S + corr_theta) / 2) > 0.5,
    }


def measure_criticality_signatures(
    S_history: np.ndarray,
    theta: float,
    baseline_window: int = 50,
    min_samples: int = 100,
) -> dict[str, Any]:
    """Measure the three simultaneous phase-transition signatures required by §10.
    Per APGI spec §10, ignition near ρ_crit ≈ 1 must produce all three:
    1. Φ-discontinuity (Cohen's d > 0.8):
       Signal variance jumps discontinuously at threshold crossing.
       Proxy for integrated information Φ — variance is used because
       direct Φ computation scales exponentially with system size (§24.3).
    2. Elevated susceptibility (variance ratio > 1.5):
       Current-window variance exceeds baseline by factor > 1.5.
       Near criticality, susceptibility (order-parameter variance) diverges —
       a signature of second-order phase transitions.
    3. Critical slowing (autocorrelation increase > 20%):
       Lag-1 autocorrelation of S(t) increases relative to baseline.
       Systems near a bifurcation point recover slowly from perturbations
       (critical slowing down, Scheffer et al., 2009).
    All three must co-occur; failure on any single signature across two
    independent datasets constitutes partial falsification (§10).
    Args:
        S_history: Time series of signal values S(t)
        theta: Ignition threshold for sub/supra-threshold split
        baseline_window: Samples to use for baseline statistics
        min_samples: Minimum samples required for reliable estimation
    Returns:
        Dictionary with keys:
            cohens_d, cohens_d_criterion (> 0.8),
            susceptibility_ratio, susceptibility_criterion (> 1.5),
            autocorr_increase, autocorr_criterion (> 0.20),
            all_criteria_met (all three co-occur — spec §10 requirement),
            n_subthreshold, n_suprathreshold, plus intermediate statistics
    """
    S = np.asarray(S_history, dtype=float)
    if len(S) < min_samples:
        return {
            "cohens_d": float("nan"),
            "cohens_d_criterion": False,
            "susceptibility_ratio": None,
            "susceptibility_criterion": False,
            "autocorr_increase": None,
            "autocorr_criterion": False,
            "all_criteria_met": False,
            "n_subthreshold": 0,
            "n_suprathreshold": 0,
            "error": f"Insufficient samples: need {min_samples}, got {len(S)}",
        }
    sub_mask = theta > S
    S_sub = S[sub_mask]
    S_supra = S[~sub_mask]
    n_sub, n_supra = len(S_sub), len(S_supra)
    # --- Signature 1: Cohen's d on sub/supra mean difference (Φ discontinuity proxy) ---
    if n_sub >= 2 and n_supra >= 2:
        var_sub = float(np.var(S_sub, ddof=1))
        var_supra = float(np.var(S_supra, ddof=1))
        pooled_sd = float(
            np.sqrt(((n_sub - 1) * var_sub + (n_supra - 1) * var_supra) / (n_sub + n_supra - 2))
        )
        cohens_d = abs(float(np.mean(S_supra)) - float(np.mean(S_sub))) / (pooled_sd + 1e-10)
    else:
        cohens_d = float("nan")
    # --- Signature 2: Susceptibility — variance ratio current/baseline ---
    half = max(2, len(S) // 4)
    win = min(baseline_window, half)
    baseline_arr = S[:win]
    current_arr = S[-win:]
    var_baseline = float(np.var(baseline_arr, ddof=1)) if len(baseline_arr) >= 2 else float("nan")
    var_current = float(np.var(current_arr, ddof=1)) if len(current_arr) >= 2 else float("nan")
    susceptibility_ratio: float | None
    if not np.isnan(var_baseline) and var_baseline > 1e-12:
        susceptibility_ratio = float(var_current / var_baseline)
    else:
        susceptibility_ratio = None

    # --- Signature 3: Critical slowing — lag-1 autocorrelation increase ---
    def _lag1_ac(x: np.ndarray) -> float:
        if len(x) < 3:
            return float("nan")
        xc = x - x.mean()
        denom = float(np.dot(xc, xc))
        return float(np.dot(xc[:-1], xc[1:]) / denom) if denom > 1e-12 else float("nan")

    ac_baseline = _lag1_ac(baseline_arr)
    ac_current = _lag1_ac(current_arr)
    autocorr_increase: float | None
    if not np.isnan(ac_baseline) and abs(ac_baseline) > 1e-6:
        autocorr_increase = float((ac_current - ac_baseline) / abs(ac_baseline))
    else:
        autocorr_increase = None
    # --- Evaluate all three criteria ---
    cohens_d_criterion = (not np.isnan(cohens_d)) and bool(cohens_d > 0.8)
    susceptibility_criterion = susceptibility_ratio is not None and bool(susceptibility_ratio > 1.5)
    autocorr_criterion = autocorr_increase is not None and bool(autocorr_increase > 0.20)
    all_criteria_met = cohens_d_criterion and susceptibility_criterion and autocorr_criterion
    return {
        "cohens_d": float(cohens_d) if not np.isnan(cohens_d) else None,
        "cohens_d_criterion": cohens_d_criterion,
        "susceptibility_ratio": susceptibility_ratio,
        "susceptibility_criterion": susceptibility_criterion,
        "autocorr_increase": autocorr_increase,
        "autocorr_criterion": autocorr_criterion,
        "all_criteria_met": all_criteria_met,
        "n_subthreshold": n_sub,
        "n_suprathreshold": n_supra,
        "var_baseline": float(var_baseline) if not np.isnan(var_baseline) else None,
        "var_current": float(var_current) if not np.isnan(var_current) else None,
        "ac_baseline": float(ac_baseline) if not np.isnan(ac_baseline) else None,
        "ac_current": float(ac_current) if not np.isnan(ac_current) else None,
    }


class StabilityAnalyzer:
    """Comprehensive stability analysis tool for APGI system."""

    def __init__(self, config: dict):
        """Initialize stability analyzer.
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.history: dict[str, list[float]] = {
            "S": [],
            "theta": [],
        }

    def step(self, S: float, theta: float) -> None:
        """Record state for analysis.
        Args:
            S: Signal value
            theta: Threshold value
        """
        self.history["S"].append(S)
        self.history["theta"].append(theta)

    def analyze(self, verbose: bool = False) -> dict[str, Any]:
        """Perform comprehensive stability analysis.
        Args:
            verbose: Print detailed output
        Returns:
            Dictionary with complete analysis
        """
        # Fixed-point stability
        stability = check_stability(self.config, verbose=verbose)
        # Fixed point
        fixed_point = compute_fixed_point(self.config)
        # Dynamics validation (if enough history)
        if len(self.history["S"]) >= 100:
            dynamics = validate_system_dynamics(
                self.config,
                np.array(self.history["S"]),
                np.array(self.history["theta"]),
            )
        else:
            dynamics = {
                "valid": False,
                "reason": "Insufficient history",
            }
        return {
            "stability": stability,
            "fixed_point": fixed_point,
            "dynamics_validation": dynamics,
        }


def _decay_exponent_of(config: dict) -> float:
    """Per-step threshold decay exponent γ_θ·dt from a config (legacy 'kappa' honoured)."""
    from config import decay_exponent

    return decay_exponent({"tau_theta": 20.0, "dt": 0.01, **config})
