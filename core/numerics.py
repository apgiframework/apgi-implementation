"""Numerically defensive helpers for degenerate inputs.

APGI analyses routinely run on series that can be constant or near-constant —
a threshold that never moves, an ignition train with no ignitions, a
subthreshold run with zero variance. NumPy and LAPACK signal these cases with
warnings (``invalid value encountered in divide``, ``Mean of empty slice``) or,
for LAPACK, with text written straight to stderr (``On entry to DLASCL,
parameter number 4 had an illegal value``) that no Python warning filter can
intercept.

The previous approach was to silence those warnings wholesale in ``pytest.ini``
and via ``warnings.filterwarnings`` at import time. That hid the degeneracy
rather than handling it: a correlation of NaN would flow onward and be reported
as a result. These helpers detect the degenerate case up front and return an
explicit, documented value instead.

Examples
--------
>>> import numpy as np
>>> safe_corrcoef(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0]))
1.0
>>> safe_corrcoef(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0]))
0.0
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "safe_corrcoef",
    "is_degenerate",
    "safe_nperseg",
    "safe_log",
    "finite_mask",
]

#: Series whose standard deviation is at or below this are treated as constant.
DEGENERACY_TOL = 1e-12


def is_degenerate(x: np.ndarray, tol: float = DEGENERACY_TOL) -> bool:
    """Return whether a series is too degenerate for variance-based statistics.

    Args:
        x: Input array.
        tol: Standard-deviation threshold below which the series counts as
            constant.

    Returns:
        True if the series is empty, contains non-finite values, or is
        (numerically) constant.
    """
    arr = np.asarray(x, dtype=float)
    if arr.size < 2:
        return True
    if not np.all(np.isfinite(arr)):
        return True
    return bool(np.std(arr) <= tol)


def safe_corrcoef(
    x: np.ndarray, y: np.ndarray, default: float = 0.0, tol: float = DEGENERACY_TOL
) -> float:
    """Pearson correlation that is defined on degenerate input.

    ``np.corrcoef`` computes ``cov/(σx·σy)``; when either series is constant
    the denominator is zero, emitting "invalid value encountered in divide" and
    returning NaN. A NaN correlation then propagates silently into reported
    validation metrics.

    Args:
        x: First series.
        y: Second series.
        default: Value returned for degenerate input. 0.0 (no linear
            association) is the honest reading: a constant series carries no
            information about covariation.
        tol: Degeneracy tolerance.

    Returns:
        Pearson r, or ``default`` when either series is degenerate or the
        result is non-finite.
    """
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"x and y must have the same shape, got {a.shape} and {b.shape}")
    if is_degenerate(a, tol) or is_degenerate(b, tol):
        return float(default)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.corrcoef(a, b)[0, 1]
    return float(r) if np.isfinite(r) else float(default)


def safe_nperseg(signal_length: int, requested: int | None = None, minimum: int = 8) -> int:
    """Clamp a Welch segment length to something the signal can support.

    ``scipy.signal.welch`` warns and silently rewrites ``nperseg`` when it
    exceeds the signal length. Clamping up front keeps the parameter that was
    actually used explicit.

    Args:
        signal_length: Number of samples available.
        requested: Desired segment length; defaults to ``min(256, N // 4)``.
        minimum: Floor on the returned value.

    Returns:
        A segment length in ``[minimum, signal_length]`` (or ``signal_length``
        when the signal is shorter than ``minimum``).

    Raises:
        ValueError: If ``signal_length`` < 1.
    """
    if signal_length < 1:
        raise ValueError(f"signal_length must be >= 1, got {signal_length}")
    target = requested if requested is not None else min(256, max(1, signal_length // 4))
    return int(max(min(target, signal_length), min(minimum, signal_length)))


def safe_log(x: np.ndarray, floor: float = float(np.finfo(float).tiny)) -> np.ndarray:
    """Elementwise natural log with a positive floor.

    ``np.log`` of zero emits "divide by zero encountered in log" and yields
    ``-inf``, which then poisons any downstream ``polyfit``. Flooring the input
    keeps the result finite and the fit well-posed.

    Args:
        x: Input array.
        floor: Smallest value permitted before taking the log.

    Returns:
        ``log(max(x, floor))``.
    """
    result: np.ndarray = np.log(np.maximum(np.asarray(x, dtype=float), floor))
    return result


def finite_mask(*arrays: np.ndarray) -> np.ndarray:
    """Return a boolean mask selecting positions finite in every input array.

    Use before ``polyfit``/``eigvals`` so NaN and inf never reach LAPACK, which
    reports them as ``DLASCL`` errors on stderr that cannot be caught as
    Python warnings.

    Args:
        *arrays: Arrays of identical shape.

    Returns:
        Boolean mask, True where all inputs are finite.

    Raises:
        ValueError: If no arrays are given or shapes differ.
    """
    if not arrays:
        raise ValueError("at least one array is required")
    mask: np.ndarray = np.isfinite(np.asarray(arrays[0], dtype=float))
    for arr in arrays[1:]:
        other = np.asarray(arr, dtype=float)
        if other.shape != mask.shape:
            raise ValueError(f"shape mismatch: {other.shape} vs {mask.shape}")
        mask &= np.isfinite(other)
    return mask
