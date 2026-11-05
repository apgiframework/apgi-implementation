"""Run manifests — provenance records for reproducible APGI runs.

A figure or number is only reproducible if you know exactly what produced it.
This module captures that context — the seed, the resolved configuration and
its hash, the git commit, the interpreter and the versions of every numerical
dependency — and writes it next to the artifact it describes.

For a repository that is the cited Data/Code Availability artifact of a paper
series, this is the difference between "here is a figure" and "here is a
figure and the exact command that regenerates it".

Examples
--------
>>> from core.manifest import build_manifest
>>> m = build_manifest({"seed": 42, "lam": 0.1})
>>> m["seed"]
42
>>> len(m["config_sha256"]) == 64
True
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "build_manifest",
    "write_manifest",
    "config_hash",
    "git_revision",
    "dependency_versions",
]

#: Packages whose version can change numerical results.
_TRACKED_PACKAGES = ("numpy", "scipy", "pydantic", "structlog", "matplotlib")


def _json_default(obj: Any) -> str:
    """Fallback encoder for values that are not natively JSON-serialisable."""
    return repr(obj)


def config_hash(config: dict[str, Any]) -> str:
    """Return a stable SHA-256 over a configuration dictionary.

    Keys are sorted so the hash depends on content, not insertion order.
    Non-serialisable values (e.g. a live ``Generator``) fall back to ``repr``,
    which keeps hashing total rather than raising mid-run.

    Args:
        config: Configuration dictionary.

    Returns:
        64-character hex digest.
    """
    blob = json.dumps(config, sort_keys=True, default=_json_default)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def git_revision(repo_root: Path | None = None) -> dict[str, Any]:
    """Return the current git commit and working-tree cleanliness.

    A dirty working tree means the code that ran is not the code at that
    commit, so the flag matters as much as the SHA.

    Args:
        repo_root: Repository directory; defaults to this file's repository.

    Returns:
        Dict with ``commit``, ``dirty`` and ``branch``. Values are ``None``
        when git is unavailable or the directory is not a repository — an
        installed wheel has no git metadata, which is expected, not an error.
    """
    root = repo_root or Path(__file__).resolve().parent.parent

    def _run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = _run("rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "dirty": None, "branch": None}
    status = _run("status", "--porcelain")
    return {
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
        "branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
    }


def dependency_versions() -> dict[str, str | None]:
    """Return installed versions of packages that can affect results.

    Returns:
        Mapping of package name to version string, or ``None`` if not installed.
    """
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, str | None] = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = None
    return versions


def build_manifest(
    config: dict[str, Any],
    *,
    extra: dict[str, Any] | None = None,
    validation_warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble a provenance manifest for a run.

    Args:
        config: The RESOLVED configuration actually used (after presets,
            clamps and any auto-corrections), not the caller's original dict —
            recording what was requested rather than what ran would defeat the
            purpose.
        extra: Additional run-specific fields (step count, output paths, …).
        validation_warnings: Spec departures recorded during the run, e.g.
            canonical-range advisories or a ``strict=False`` override. Their
            presence means the run is not the canonical parameterisation.

    Returns:
        JSON-serialisable manifest dictionary.
    """
    manifest: dict[str, Any] = {
        "apgi_version": _package_version(),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seed": config.get("seed"),
        "strict": config.get("strict", config.get("strict_mode")),
        "config_sha256": config_hash(config),
        "config": json.loads(json.dumps(config, sort_keys=True, default=_json_default)),
        "git": git_revision(),
        "python": {
            "version": sys.version.split()[0],
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "dependencies": dependency_versions(),
        "validation_warnings": list(validation_warnings or []),
        # ``spec_conformant`` means only "no configuration-range departures were
        # recorded". It is NOT a statement that the run followed a registered
        # analysis plan (AUD-REP-004): that requires a verified protocol lock
        # (runtime.protocol_lock) recorded in a run manifest. Until then every
        # run is labelled exploratory.
        "spec_conformant": not validation_warnings,
        "conformance_scope": "configuration range checks only",
        "registered_plan_verified": False,
        "analysis_label": "exploratory",
    }
    if extra:
        manifest.update(extra)
    return manifest


def _package_version() -> str | None:
    """Return the installed apgi version, or ``None`` when running from source."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("apgi")
    except PackageNotFoundError:
        return None


def write_manifest(
    path: str | Path,
    config: dict[str, Any],
    *,
    extra: dict[str, Any] | None = None,
    validation_warnings: list[str] | None = None,
) -> Path:
    """Build a manifest and write it as JSON.

    Args:
        path: Destination file. Parent directories are created as needed.
        config: Resolved configuration (see :func:`build_manifest`).
        extra: Additional run-specific fields.
        validation_warnings: Recorded spec departures.

    Returns:
        The path written.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(config, extra=extra, validation_warnings=validation_warnings)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
