"""Deprecation helpers for names retired by the APGI Notation Appendix v1.1.

Bare κ is reserved for the metabolic bridge (ATP/bit) and κ_meta is
deprecated (Notation Appendix §11); κ_e, κ_i and κ_gen are withdrawn in favour
of η_pred,e, η_pred,i and η_gen. Production code used these names for other
quantities; the old names stay accepted for one release and emit
:class:`APGIDeprecationWarning`. See apgi-lean known-conflicts PD-KAPPA-SYMBOL
and PD-KAPPA-RESIDUAL.
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


class RenamedKeyDict(dict[str, Any]):
    """A result dict whose retired keys still resolve, with a warning, for one release.

    ``d[old]`` and ``d.get(old)`` return ``d[new]`` and warn; ``old in d`` is False,
    so serialised output carries only the new names.
    """

    def __init__(self, data: Mapping[str, Any], renamed: Mapping[str, str]) -> None:
        super().__init__(data)
        self._renamed = dict(renamed)

    def __missing__(self, key: str) -> Any:
        new = self.__dict__.get("_renamed", {}).get(key)
        if new is None:
            raise KeyError(key)
        warn_renamed(key, new, stacklevel=4)
        return self[new]

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        if key in self:
            return dict.__getitem__(self, key)
        if key in self.__dict__.get("_renamed", {}):
            return self[key]
        return default


def config_value(config: Mapping[str, Any], new: str, old: str, default: Any) -> Any:
    """Read ``config[new]``, unless the deprecated ``config[old]`` was set explicitly.

    CONFIG no longer defines the old key, so its presence means a caller set it;
    it then wins over the default-filled new key, with a warning.
    """
    if old in config:
        warn_renamed(old, new)
        return config[old]
    return config.get(new, default)
