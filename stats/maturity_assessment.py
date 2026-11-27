"""Automated maturity assessment system for APGI implementation.
Implements comprehensive maturity scoring combining:
- Hierarchical Architecture (§8) assessment
- Statistical Validation (§12) assessment
- Overall system health scoring
- Diagnostic recommendations
Spec §15: Design Constraints and Maturity Metrics
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.logging_config import get_logger
from core.numerics import safe_corrcoef
from stats.spectral_extraction import SpectralSignature, extract_1f_signature

logger = get_logger("apgi.maturity")


@dataclass
class MaturityScore:
    """Result of maturity assessment."""

    hierarchical_score: float  # §8 maturity (0-100)
    statistical_score: float  # §12 maturity (0-100)
    overall_score: float  # Overall maturity (0-100)
    # Component scores
    pac_score: float  # Phase-amplitude coupling quality
    cascade_score: float  # Threshold cascade effectiveness
    spectral_score: float  # 1/f signature quality
    coherence_score: float  # Cross-level coherence
    # Diagnostic info
    issues: list[str]  # List of identified issues
    recommendations: list[str]  # Improvement recommendations
    # Detailed metrics
    spectral_signature: SpectralSignature | None = None
    hierarchical_coupling_strength: float = 0.0
    cascade_effectiveness: float = 0.0


def assess_hierarchical_architecture(
    signal_levels: list[np.ndarray],
    theta_levels: list[np.ndarray],
    phi_levels: list[np.ndarray],
    pi_levels: list[np.ndarray],
    fs: float = 1.0,
) -> tuple[float, float, float, list[str], list[str]]:
    """Assess hierarchical architecture maturity (§8).
    Evaluates:
    - Phase-amplitude coupling quality
    - Threshold cascade effectiveness
    - Cross-level coherence
    - Timescale separation
    Args:
        signal_levels: List of signals at each hierarchy level
        theta_levels: List of threshold trajectories
        phi_levels: List of phase trajectories
        pi_levels: List of precision trajectories
        fs: Sampling frequency (Hz)
    Returns:
        (pac_score, cascade_score, coherence_score, issues, recommendations)
    """
    issues = []
    recommendations = []
    n_levels = len(signal_levels)
    # 1. Phase-Amplitude Coupling Quality
    pac_score = 0.0
    if n_levels > 1:
        # Measure correlation between phase and threshold modulation
        pac_correlations = []
        for ell in range(n_levels - 1):
            # Correlation between higher level phase and this level's threshold
            if len(phi_levels[ell + 1]) > 0 and len(theta_levels[ell]) > 0:
                min_len = min(len(phi_levels[ell + 1]), len(theta_levels[ell]))
                if min_len > 1:
                    a = np.asarray(phi_levels[ell + 1][:min_len], dtype=float)
                    b = np.asarray(theta_levels[ell][:min_len], dtype=float)
                    if np.std(a) > 0 and np.std(b) > 0:
                        corr = safe_corrcoef(a, b)
                        if not np.isnan(corr):
                            pac_correlations.append(abs(corr))
        if pac_correlations:
            pac_score = float(np.mean(pac_correlations) * 100)
        else:
            pac_score = 0.0
            issues.append("No phase-amplitude coupling detected")
            recommendations.append("Increase kappa_down coupling strength")
    # 2. Threshold Cascade Effectiveness (IMPROVED)
    cascade_score = 0.0
    if n_levels > 1:
        # Measure actual suppression magnitude when lower levels are suprathreshold
        cascade_effects = []
        for ell in range(1, n_levels):
            if len(signal_levels[ell - 1]) > 0 and len(theta_levels[ell]) > 0:
                min_len = min(len(signal_levels[ell - 1]), len(theta_levels[ell]))
                if min_len > 10:  # Need sufficient samples
                    S_lower = np.asarray(signal_levels[ell - 1][:min_len], dtype=float)
                    theta_upper = np.asarray(theta_levels[ell][:min_len], dtype=float)

                    # Identify periods when lower level is suprathreshold
                    # Use dynamic threshold based on lower level statistics
                    lower_threshold = float(np.median(S_lower)) + 0.5 * np.std(S_lower)
                    suprathresh_mask = S_lower > lower_threshold

                    if np.sum(suprathresh_mask) > 5:  # Need enough suprathreshold samples
                        # Compare threshold during suprathreshold vs baseline periods
                        theta_supra = theta_upper[suprathresh_mask]
                        theta_baseline = theta_upper[~suprathresh_mask]

                        baseline_mean = (
                            float(np.mean(theta_baseline)) if len(theta_baseline) > 0 else 1.0
                        )
                        if baseline_mean > 1e-6:
                            suppression = 1.0 - (float(np.mean(theta_supra)) / baseline_mean)
                            # A suppression is a fraction in [0, 1]: thresholds are ≥ θ_floor = 0,
                            # so θ̄_supra ≥ 0 bounds it by 1. Clipping keeps the score in
                            # [0, 100] for inputs outside that domain (F17).
                            cascade_effects.append(min(1.0, max(0.0, suppression)))

        if cascade_effects:
            cascade_score = float(np.mean(cascade_effects) * 100)
        else:
            cascade_score = 0.0
            issues.append("No threshold cascade detected")
            recommendations.append("Increase kappa_up cascade strength")
    # 3. Cross-Level Coherence
    coherence_score = 0.0
    if n_levels > 1:
        try:
            from scipy import signal as scipy_signal  # type: ignore

            coherences = []
            for ell in range(n_levels - 1):
                if len(signal_levels[ell]) > 100 and len(signal_levels[ell + 1]) > 100:
                    # Ensure nperseg doesn't exceed signal length
                    min_len = min(len(signal_levels[ell]), len(signal_levels[ell + 1]))
                    nperseg = min(256, min_len // 4)
                    nperseg = max(nperseg, 8)  # Minimum for meaningful analysis
                    f, coh = scipy_signal.coherence(
                        signal_levels[ell], signal_levels[ell + 1], fs=fs, nperseg=nperseg
                    )
                    # Average coherence in mid-frequency range
                    mid_idx = len(coh) // 2
                    coherences.append(np.mean(coh[mid_idx - 10 : mid_idx + 10]))
            coherence_score = float(np.mean(coherences) * 100) if coherences else 0.0
        except Exception:
            coherence_score = 0.0
    # Compute overall hierarchical score
    _ = float(0.4 * pac_score + 0.3 * cascade_score + 0.3 * coherence_score)
    # Check for issues
    if pac_score < 20:
        issues.append("Weak phase-amplitude coupling")
    if cascade_score < 20:
        issues.append("Weak threshold cascade")
    if coherence_score < 20:
        issues.append("Low cross-level coherence")
    return pac_score, cascade_score, coherence_score, issues, recommendations


def assess_statistical_validation(
    signal: np.ndarray,
    signal_levels: list[np.ndarray] | None = None,
    fs: float = 1.0,
) -> tuple[float, float, float, list[str], list[str], SpectralSignature | None]:
    """Assess statistical validation maturity (§12).
    Evaluates:
    - 1/f spectral signature quality
    - Spectral exponent confidence
    - Per-level spectral consistency
    - Goodness of fit
    Args:
        signal: Main signal time series
        signal_levels: Optional list of per-level signals
        fs: Sampling frequency (Hz)
    Returns:
        (spectral_score, confidence_score, consistency_score, issues, recommendations, signature)
    """
    issues = []
    recommendations = []
    spectral_signature = None
    # Check for very short signals - return 0 scores for unreliable data
    if len(signal) < 100:
        issues.append("Signal too short for reliable spectral analysis")
        recommendations.append("Increase signal length to at least 100 samples")
        return (0.0, 0.0, 0.0, issues, recommendations, None)
    # 1. Extract 1/f signature
    try:
        spectral_signature = extract_1f_signature(signal, fs=fs, n_bootstrap=100)
        # Spectral score based on pink noise detection and R²
        if spectral_signature.is_pink_noise:
            spectral_score = float(spectral_signature.r_squared * 100)
        else:
            spectral_score = float(spectral_signature.r_squared * 50)
        # Confidence score
        confidence_score = float(spectral_signature.confidence * 100)
        # Check for issues
        if spectral_signature.beta < 0.8:
            issues.append("Spectral exponent too low (white noise)")
            recommendations.append("Increase hierarchical coupling strength")
        elif spectral_signature.beta > 1.5:
            issues.append("Spectral exponent too high (brown noise)")
            recommendations.append("Decrease hierarchical coupling strength")
        if spectral_signature.r_squared < 0.7:
            issues.append("Poor spectral fit (R² < 0.7)")
            recommendations.append("Verify signal preprocessing and filtering")
    except Exception as e:
        spectral_score = 0.0
        confidence_score = 0.0
        issues.append(f"Spectral extraction failed: {str(e)}")
        recommendations.append("Check signal quality and length")
    # 2. Per-level spectral consistency
    consistency_score = 0.0
    if signal_levels is not None and len(signal_levels) > 1:
        try:
            level_betas = []
            for sig in signal_levels:
                try:
                    sig_extract = extract_1f_signature(sig, fs=fs, n_bootstrap=50)
                    level_betas.append(sig_extract.beta)
                except Exception:  # nosec B110
                    pass
            if len(level_betas) > 1:
                # Consistency: low variance in spectral exponents
                beta_std = np.std(level_betas)
                consistency_score = float(max(0, 100 - beta_std * 50))  # type: ignore
            else:
                consistency_score = 0.0
        except Exception:
            consistency_score = 0.0
    # Compute overall statistical score
    _ = float(0.5 * spectral_score + 0.3 * confidence_score + 0.2 * consistency_score)
    return (
        spectral_score,
        confidence_score,
        consistency_score,
        issues,
        recommendations,
        spectral_signature,
    )


def assess_overall_maturity(
    signal: np.ndarray,
    signal_levels: list[np.ndarray] | None = None,
    theta_levels: list[np.ndarray] | None = None,
    phi_levels: list[np.ndarray] | None = None,
    pi_levels: list[np.ndarray] | None = None,
    fs: float = 1.0,
) -> MaturityScore:
    """Comprehensive maturity assessment combining all components.
    Args:
        signal: Main signal time series
        signal_levels: Optional list of per-level signals
        theta_levels: Optional list of threshold trajectories
        phi_levels: Optional list of phase trajectories
        pi_levels: Optional list of precision trajectories
        fs: Sampling frequency (Hz)
    Returns:
        MaturityScore with all metrics and recommendations
    """
    # Initialize defaults
    if signal_levels is None:
        signal_levels = [signal]
    if theta_levels is None:
        theta_levels = [np.zeros_like(signal)]
    if phi_levels is None:
        phi_levels = [np.zeros_like(signal)]
    if pi_levels is None:
        pi_levels = [np.ones_like(signal)]
    # Assess hierarchical architecture
    pac_score, cascade_score, coherence_score, hier_issues, hier_recs = (
        assess_hierarchical_architecture(signal_levels, theta_levels, phi_levels, pi_levels, fs=fs)
    )
    hierarchical_score = float(0.4 * pac_score + 0.3 * cascade_score + 0.3 * coherence_score)
    # Assess statistical validation
    spectral_score, confidence_score, consistency_score, stat_issues, stat_recs, sig = (
        assess_statistical_validation(signal, signal_levels, fs=fs)
    )
    statistical_score = float(
        0.5 * spectral_score + 0.3 * confidence_score + 0.2 * consistency_score
    )
    # Compute overall score
    overall_score = float(0.5 * hierarchical_score + 0.5 * statistical_score)
    # Combine issues and recommendations
    all_issues = hier_issues + stat_issues
    all_recommendations = hier_recs + stat_recs
    # Add general recommendations based on overall score
    if overall_score < 50:
        all_recommendations.append("System maturity is low; review all parameters")
    elif overall_score < 70:
        all_recommendations.append("System maturity is moderate; focus on weak components")
    return MaturityScore(
        hierarchical_score=hierarchical_score,
        statistical_score=statistical_score,
        overall_score=overall_score,
        pac_score=pac_score,
        cascade_score=cascade_score,
        spectral_score=spectral_score,
        coherence_score=coherence_score,
        issues=all_issues,
        recommendations=all_recommendations,
        spectral_signature=sig,
        hierarchical_coupling_strength=pac_score / 100.0,
        cascade_effectiveness=cascade_score / 100.0,
    )


def log_maturity_assessment(score: MaturityScore) -> None:
    """Log maturity assessment results via structured logging."""
    logger.info(
        "maturity_assessment",
        overall_score=score.overall_score,
        hierarchical_score=score.hierarchical_score,
        statistical_score=score.statistical_score,
        pac_score=score.pac_score,
        cascade_score=score.cascade_score,
        spectral_score=score.spectral_score,
        coherence_score=score.coherence_score,
    )
    if score.spectral_signature:
        logger.info(
            "spectral_analysis",
            spectral_beta=score.spectral_signature.beta,
            hurst_exponent=score.spectral_signature.hurst,
            is_pink_noise=score.spectral_signature.is_pink_noise,
            r_squared=score.spectral_signature.r_squared,
        )
    if score.issues:
        for issue in score.issues:
            logger.warning("maturity_issue", issue=issue)
    if score.recommendations:
        for i, rec in enumerate(score.recommendations, 1):
            logger.info("maturity_recommendation", number=i, recommendation=rec)


def format_maturity_assessment(score: MaturityScore) -> str:
    """Format maturity assessment as human-readable string.
    Returns:
        Formatted assessment string for display
    """
    lines = [
        "",
        "=" * 80,
        "APGI SYSTEM MATURITY ASSESSMENT",
        "=" * 80,
        f"\nOVERALL MATURITY: {score.overall_score:.1f}/100",
        f"  Hierarchical Architecture (§8): {score.hierarchical_score:.1f}/100",
        f"  Statistical Validation (§12): {score.statistical_score:.1f}/100",
        "\nCOMPONENT SCORES:",
        f"  Phase-Amplitude Coupling: {score.pac_score:.1f}/100",
        f"  Threshold Cascade: {score.cascade_score:.1f}/100",
        f"  Spectral Signature: {score.spectral_score:.1f}/100",
        f"  Cross-Level Coherence: {score.coherence_score:.1f}/100",
    ]
    if score.spectral_signature:
        lines.extend(
            [
                "\nSPECTRAL ANALYSIS:",
                f"  Spectral Exponent (β): {score.spectral_signature.beta:.3f}",
                f"  Hurst Exponent (H): {score.spectral_signature.hurst:.3f}",
                f"  Pink Noise: {'✓ YES' if score.spectral_signature.is_pink_noise else '✗ NO'}",
                f"  Goodness of Fit (R²): {score.spectral_signature.r_squared:.3f}",
            ]
        )
    if score.issues:
        lines.append(f"\nISSUES ({len(score.issues)}):")
        for issue in score.issues:
            lines.append(f"  ⚠ {issue}")
    else:
        lines.append("\nISSUES: None detected ✓")
    if score.recommendations:
        lines.append(f"\nRECOMMENDATIONS ({len(score.recommendations)}):")
        for i, rec in enumerate(score.recommendations, 1):
            lines.append(f"  {i}. {rec}")
    lines.extend(["", "=" * 80, ""])
    return "\n".join(lines)


def print_maturity_assessment(score: MaturityScore) -> None:
    """Pretty-print maturity assessment results (backward compatibility)."""
    print(format_maturity_assessment(score))


def get_maturity_rating(score: float) -> str:
    """Get human-readable maturity rating."""
    if score >= 90:
        return "Excellent (90-100)"
    elif score >= 80:
        return "Very Good (80-89)"
    elif score >= 70:
        return "Good (70-79)"
    elif score >= 60:
        return "Satisfactory (60-69)"
    elif score >= 50:
        return "Acceptable (50-59)"
    else:
        return "Needs Improvement (<50)"
