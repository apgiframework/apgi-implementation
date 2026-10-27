"""Version adapter (S1-IMP-10.3, S1-IMP-20).

Retires duplicated spellings through adapters instead of silent aliases. Each
legacy name is recorded in :data:`DEPRECATIONS` with its canonical replacement,
the release that deprecated it and the release that will remove it; using one
emits a :class:`DeprecationWarning` that names all three.

:data:`DEPRECATIONS` is the single source of truth: ``DEPENDENCY-MATRIX.md``
must list every entry, and ``tools/check_dependencies.py`` fails the build when
the table and the matrix disagree — a duplicate without a deprecation entry is
rejected, as the register requires.

The module also provides a small migration chain for versioned JSON documents
(run records, gate tables) so old files keep loading after a schema change.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

__all__ = [
    "SPEC_VERSION",
    "Deprecation",
    "DEPRECATIONS",
    "ConflictingAliasError",
    "adapt_config",
    "adapt_output_keys",
    "register_migration",
    "migrate",
    "deprecated_names",
]

#: Specification / package release the canonical names belong to.
SPEC_VERSION = "1.1.0"


@dataclass(frozen=True)
class Deprecation:
    """One deprecated spelling.

    Attributes:
        old: Legacy name.
        new: Canonical name.
        kind: ``"config"`` or ``"output"``.
        deprecated_in: Release that deprecated ``old``.
        removal_in: Release that removes it.
        owner: Repository/module that owns the canonical definition.
    """

    old: str
    new: str
    kind: str
    deprecated_in: str
    removal_in: str
    owner: str = "apgi-implementation"


DEPRECATIONS: tuple[Deprecation, ...] = (
    Deprecation("beta", "beta_da", "config", "1.1.0", "2.0.0"),
    Deprecation("ignite_tau", "tau_sigma", "config", "1.1.0", "2.0.0"),
    Deprecation("strict_mode", "strict", "config", "1.1.0", "2.0.0"),
    Deprecation("ai_F_values", "ai_efe_values", "output", "1.1.0", "2.0.0"),
)

_BY_KIND: dict[str, dict[str, Deprecation]] = {"config": {}, "output": {}}
for _d in DEPRECATIONS:
    _BY_KIND[_d.kind][_d.old] = _d


class ConflictingAliasError(ValueError):
    """Both a legacy name and its canonical replacement are set to different values."""


def deprecated_names(kind: str | None = None) -> list[str]:
    """Legacy names, optionally restricted to ``"config"`` or ``"output"``."""
    return sorted(d.old for d in DEPRECATIONS if kind is None or d.kind == kind)


def _warn(d: Deprecation, stacklevel: int) -> None:
    warnings.warn(
        f"{d.kind} name {d.old!r} is deprecated since {d.deprecated_in} and will be removed in "
        f"{d.removal_in}; use {d.new!r}.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


def _adapt(
    mapping: Mapping[str, Any], kind: str, warn: bool, stacklevel: int
) -> tuple[dict[str, Any], list[Deprecation]]:
    out = dict(mapping)
    applied: list[Deprecation] = []
    for old, d in _BY_KIND[kind].items():
        if old not in out:
            continue
        applied.append(d)
        if d.new in out and out[d.new] != out[old]:
            raise ConflictingAliasError(
                f"{old!r}={out[old]!r} conflicts with {d.new!r}={out[d.new]!r}"
            )
        # Both spellings present and equal is how config.CONFIG ships its defaults;
        # that is not the caller relying on the legacy name, so it stays quiet.
        if warn and d.new not in out:
            _warn(d, stacklevel)
        out.setdefault(d.new, out[old])
        del out[old]
    return out, applied


def adapt_config(
    config: Mapping[str, Any], *, warn: bool = True
) -> tuple[dict[str, Any], list[Deprecation]]:
    """Rewrite legacy config keys to their canonical names.

    Returns:
        ``(adapted_config, applied_deprecations)``; the input is not mutated.

    Raises:
        ConflictingAliasError: If an old and new spelling are both present with
            different values — silently preferring one would change the run.
    """
    return _adapt(config, "config", warn, 3)


def adapt_output_keys(
    outputs: Mapping[str, Any], *, warn: bool = True
) -> tuple[dict[str, Any], list[Deprecation]]:
    """Rewrite legacy output keys to their canonical names."""
    return _adapt(outputs, "output", warn, 3)


_MIGRATIONS: dict[tuple[str, str], Callable[[dict[str, Any]], dict[str, Any]]] = {}


def register_migration(
    from_version: str, to_version: str, fn: Callable[[dict[str, Any]], dict[str, Any]]
) -> None:
    """Register ``fn`` upgrading a document from ``from_version`` to ``to_version``."""
    key = (from_version, to_version)
    if key in _MIGRATIONS:
        raise ValueError(f"migration {from_version} -> {to_version} already registered")
    _MIGRATIONS[key] = fn


def migrate(doc: Mapping[str, Any], from_version: str, to_version: str) -> dict[str, Any]:
    """Apply registered migrations in a chain from ``from_version`` to ``to_version``.

    Raises:
        LookupError: If no chain of registered migrations connects the versions.
    """
    current = dict(doc)
    version = from_version
    seen = {version}
    while version != to_version:
        step = next(((a, b) for (a, b) in _MIGRATIONS if a == version and b not in seen), None)
        if step is None:
            raise LookupError(f"no migration path from {from_version} to {to_version}")
        current = _MIGRATIONS[step](current)
        version = step[1]
        seen.add(version)
    return current
