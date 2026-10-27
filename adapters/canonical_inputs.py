"""Canonical inputs (S1-IMP-10.1).

Everything that drives a model enters through :func:`to_canonical_inputs`, which
accepts the shapes callers actually have — arrays, mappings with a range of
historical key names, pandas-like frames — and returns one validated
:class:`CanonicalInputs`. Invalid data fail here, once, with a message naming the
offending field, rather than as a NaN several hundred steps into a run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["CanonicalInputs", "to_canonical_inputs", "EXTERO_ALIASES", "INTERO_ALIASES"]

#: Accepted names for the exteroceptive stream, first match wins.
EXTERO_ALIASES: tuple[str, ...] = ("x_e", "exteroceptive", "stimulus", "x")
#: Accepted names for the interoceptive stream, first match wins.
INTERO_ALIASES: tuple[str, ...] = ("x_i", "interoceptive")


@dataclass(frozen=True)
class CanonicalInputs:
    """Validated model inputs.

    Attributes:
        x_e: Exteroceptive input, 1-D float array.
        x_i: Interoceptive input (same length) or ``None``.
        dt: Step length in seconds, or ``None`` if not supplied.
    """

    x_e: np.ndarray
    x_i: np.ndarray | None = None
    dt: float | None = None

    @property
    def n_steps(self) -> int:
        return int(self.x_e.shape[0])

    def as_dict(self) -> dict[str, Any]:
        return {
            "x_e": self.x_e.tolist(),
            "x_i": None if self.x_i is None else self.x_i.tolist(),
            "dt": self.dt,
        }


def _column(obj: Any, name: str) -> Any:
    """Fetch ``name`` from a mapping or frame-like object, else ``None``."""
    if isinstance(obj, Mapping):
        return obj.get(name)
    try:
        if name in obj.columns:
            return obj[name]
    except AttributeError:
        pass
    return None


def _as_vector(value: Any, field: str, allow_nan: bool) -> np.ndarray:
    try:
        arr = np.asarray(value.to_numpy() if hasattr(value, "to_numpy") else value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field}: cannot convert to a float array ({exc})") from exc
    if arr.ndim != 1:
        raise ValueError(f"{field}: expected a 1-D sequence, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{field}: must not be empty")
    bad = ~np.isfinite(arr)
    if bad.any() and not (allow_nan and np.isnan(arr[bad]).all()):
        idx = int(np.flatnonzero(bad)[0])
        raise ValueError(f"{field}: non-finite value at index {idx} ({arr[idx]!r})")
    return arr


def to_canonical_inputs(
    data: Any,
    *,
    x_i: Sequence[float] | np.ndarray | None = None,
    dt: float | None = None,
    allow_nan: bool = False,
) -> CanonicalInputs:
    """Normalise ``data`` to :class:`CanonicalInputs`.

    Args:
        data: One of: a 1-D sequence (exteroceptive only); a 2-D array with
            columns ``(x_e, x_i)``; a mapping or frame with any of
            :data:`EXTERO_ALIASES` / :data:`INTERO_ALIASES` (and optionally
            ``dt``); or an existing :class:`CanonicalInputs`.
        x_i: Interoceptive stream when ``data`` is a bare sequence.
        dt: Step length; overrides a ``dt`` carried in ``data``. Must be > 0.
        allow_nan: Permit NaN (never ±inf), e.g. for explicit missing samples.

    Raises:
        ValueError: On missing/empty/non-finite data, length mismatch or ``dt <= 0``.
        KeyError: If a mapping/frame has no exteroceptive field.
    """
    if isinstance(data, CanonicalInputs):
        return to_canonical_inputs(
            data.x_e, x_i=data.x_i, dt=dt if dt is not None else data.dt, allow_nan=allow_nan
        )

    carried_dt: Any = None
    xe_raw: Any = None
    xi_raw: Any = x_i
    if isinstance(data, Mapping) or hasattr(data, "columns"):
        for name in EXTERO_ALIASES:
            xe_raw = _column(data, name)
            if xe_raw is not None:
                break
        if xe_raw is None:
            raise KeyError(f"no exteroceptive field; expected one of {EXTERO_ALIASES}")
        if xi_raw is None:
            for name in INTERO_ALIASES:
                xi_raw = _column(data, name)
                if xi_raw is not None:
                    break
        carried_dt = _column(data, "dt") if isinstance(data, Mapping) else None
    else:
        arr = np.asarray(data, dtype=float) if not hasattr(data, "to_numpy") else data.to_numpy()
        if arr.ndim == 2:
            if arr.shape[1] not in (1, 2):
                raise ValueError(f"2-D input must have 1 or 2 columns, got {arr.shape[1]}")
            xe_raw = arr[:, 0]
            if arr.shape[1] == 2 and xi_raw is None:
                xi_raw = arr[:, 1]
        else:
            xe_raw = arr

    xe = _as_vector(xe_raw, "x_e", allow_nan)
    xi = None if xi_raw is None else _as_vector(xi_raw, "x_i", allow_nan)
    if xi is not None and xi.shape != xe.shape:
        raise ValueError(f"x_i length {xi.shape[0]} does not match x_e length {xe.shape[0]}")
    step = dt if dt is not None else carried_dt
    if step is not None:
        step = float(step)
        if not np.isfinite(step) or step <= 0:
            raise ValueError(f"dt must be a positive finite number, got {step!r}")
    return CanonicalInputs(xe, xi, step)
