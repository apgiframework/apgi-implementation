"""Deprecation helpers for names retired by the APGI Notation Appendix v1.1.

Bare κ is reserved for the metabolic bridge (ATP/bit) and κ_meta is
deprecated (Notation Appendix §11). Production code used both names for other
quantities; the old names stay accepted for one release and emit
:class:`APGIDeprecationWarning`. See apgi-lean known-conflicts PD-KAPPA-SYMBOL.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class APGIDeprecationWarning(DeprecationWarning):
    """A retired APGI parameter or config name was used."""


def warn_renamed(old: str, new: str, stacklevel: int = 3) -> None:
    warnings.warn(
        f"'{old}' is deprecated (Notation Appendix v1.1); use '{new}'. "
        "The old name will be removed in the next release.",
        APGIDeprecationWarning,
        stacklevel=stacklevel,
    )


def renamed_kwargs(**renames: str) -> Callable[[F], F]:
    """Accept deprecated keyword names: ``@renamed_kwargs(kappa="decay_exponent")``."""

    def deco(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for old, new in renames.items():
                if old in kwargs:
                    if new in kwargs:
                        raise TypeError(
                            f"{fn.__name__}() got both '{old}' and its replacement '{new}'"
                        )
                    warn_renamed(old, new)
                    kwargs[new] = kwargs.pop(old)
            return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return deco


def config_value(config: Mapping[str, Any], new: str, old: str, default: Any) -> Any:
    """Read ``config[new]``, unless the deprecated ``config[old]`` was set explicitly.

    CONFIG no longer defines the old key, so its presence means a caller set it;
    it then wins over the default-filled new key, with a warning.
    """
    if old in config:
        warn_renamed(old, new)
        return config[old]
    return config.get(new, default)
