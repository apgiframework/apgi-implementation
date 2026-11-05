"""Shared finite/shape/domain checks for public scientific entry points (AUD-DEF-01).

APGI-PRO-002 and APGI-DAT-001/002 require malformed inputs to be rejected
*before* any model, session, statistics, RNG or history state changes. Every
public boundary that accepts numbers from a caller — ``APGIPipeline.step``,
``APGIModel.step``/``run``, ``SessionEngine.advance``, the estimators — calls
these helpers first, so a rejected call leaves no partial state behind.

The helpers raise :class:`NonFiniteInputError`, a :class:`ValueError` subclass,
so existing ``except ValueError`` handlers keep working while callers that need
to distinguish a data-quality rejection can catch the specific type.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real
from typing import Any

import numpy as np

__all__ = [
    "NonFiniteInputError",
    "require_finite_scalar",
    "require_optional_finite_scalar",
    "require_finite_array",
    "require_positive_finite",
]


class NonFiniteInputError(ValueError):
    """A caller-supplied value is NaN, infinite, non-numeric or of the wrong shape."""


def require_finite_scalar(name: str, value: Any) -> float:
    """Return ``value`` as a finite ``float`` or raise.

    ``bool`` is rejected: ``True``/``False`` passed where a stimulus is expected is
    almost always a caller bug, and silently treating it as 1.0/0.0 would hide it.
    """
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (Real, np.number)):
        raise NonFiniteInputError(f"{name} must be a real number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        raise NonFiniteInputError(f"{name} must be finite, got {out!r}")
    return out


def require_optional_finite_scalar(name: str, value: Any) -> float | None:
    """Like :func:`require_finite_scalar` but passes ``None`` through."""
    return None if value is None else require_finite_scalar(name, value)


def require_positive_finite(name: str, value: Any) -> float:
    """Return ``value`` as a finite float strictly greater than zero, or raise."""
    out = require_finite_scalar(name, value)
    if out <= 0.0:
        raise NonFiniteInputError(f"{name} must be > 0, got {out!r}")
    return out


def require_finite_array(
    name: str,
    value: Sequence[float] | np.ndarray,
    *,
    ndim: int | None = 1,
    length: int | None = None,
    allow_empty: bool = True,
) -> np.ndarray:
    """Return ``value`` as a float array after shape and finiteness checks.

    Args:
        name: Field name used in error messages.
        value: Array-like input.
        ndim: Required number of dimensions (``None`` skips the check).
        length: Required length of the first axis (``None`` skips the check).
        allow_empty: Whether a zero-length array is acceptable.

    Raises:
        NonFiniteInputError: On non-numeric data, wrong shape, empty input when
            not allowed, or any NaN/inf element. The message names the first
            offending index so the record can be quarantined.
    """
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise NonFiniteInputError(f"{name} is not numeric: {exc}") from exc
    if ndim is not None and arr.ndim != ndim:
        raise NonFiniteInputError(f"{name} must be {ndim}-D, got shape {arr.shape}")
    if length is not None and arr.shape[0] != length:
        raise NonFiniteInputError(f"{name} must have length {length}, got {arr.shape[0]}")
    if not allow_empty and arr.size == 0:
        raise NonFiniteInputError(f"{name} must not be empty")
    bad = ~np.isfinite(arr)
    if bad.any():
        first = tuple(int(i) for i in np.argwhere(bad)[0])
        idx = first[0] if len(first) == 1 else first
        raise NonFiniteInputError(
            f"{name} contains {int(bad.sum())} non-finite value(s); first at index {idx}"
        )
    return arr
