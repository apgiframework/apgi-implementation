"""Run manifest with lattice stamps (S1-IMP-16).

Every scientific run emits a manifest carrying the full lattice of stamps that
locate it: ``run_id``, ``study``, protocol id/version/hash, model id/version,
commits, dataset hash, seed, ``data_origin``, ``mode``, environment lock, gate
table, feature version and platform release.

A manifest also says whether the run may be called reproducible. It may not
until the objects it points at are accessible: a missing seed, an unresolved
commit, a dirty working tree, an absent dataset hash or protocol hash, or a gate
table that cannot be found and hashed each list a *blocker*, and a run with any
blocker has ``reproducible = False``. :func:`verify_manifest` re-checks the
referenced files against their recorded hashes.
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.manifest import dependency_versions, git_revision

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "FEATURE_VERSION",
    "DATA_ORIGINS",
    "MODES",
    "REQUIRED_FIELDS",
    "sha256_file",
    "environment_lock",
    "build_run_manifest",
    "validate_run_manifest",
    "verify_manifest",
    "write_run_manifest",
]

MANIFEST_SCHEMA_VERSION = "1"
#: Version of the engine's feature definitions (what summaries/features mean).
FEATURE_VERSION = "1.0.0"
DATA_ORIGINS = ("synthetic", "simulated", "pilot", "confirmatory", "archival")
MODES = ("desk", "exploratory", "pilot", "confirmatory")

#: Top-level keys every manifest must carry (values may be ``None`` only where
#: a blocker records why).
REQUIRED_FIELDS: tuple[str, ...] = (
    "schema_version",
    "run_id",
    "study",
    "protocol",
    "model",
    "commits",
    "dataset_hash",
    "seed",
    "data_origin",
    "mode",
    "env_lock",
    "gate_table",
    "feature_version",
    "platform_release",
    "reproducible",
    "reproducibility_blockers",
)


def sha256_file(path: str | Path) -> str:
    """SHA-256 hex digest of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def environment_lock() -> dict[str, Any]:
    """Interpreter and tracked dependency versions, with a digest over both."""
    body = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "dependencies": dependency_versions(),
    }
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return {**body, "sha256": digest}


def _package_release() -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("apgi")
    except PackageNotFoundError:
        return None


def build_run_manifest(
    *,
    study: str,
    protocol_id: str,
    protocol_version: str,
    protocol_hash: str | None,
    model_id: str,
    model_version: str,
    seed: int | None,
    data_origin: str,
    mode: str,
    dataset_hash: str | None,
    framework_commit: str | None = None,
    spec_commit: str | None = None,
    gate_table_path: str | Path | None = None,
    feature_version: str = FEATURE_VERSION,
    platform_release: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a run manifest and decide whether the run may be called reproducible.

    Args:
        study: Study identifier.
        protocol_id: Protocol identifier.
        protocol_version: Protocol version.
        protocol_hash: Content hash of the protocol document (``None`` → blocker).
        model_id: Registered model id.
        model_version: Model version.
        seed: Seed (``None`` → blocker).
        data_origin: One of :data:`DATA_ORIGINS`.
        mode: One of :data:`MODES`.
        dataset_hash: Digest of the input data (``None`` → blocker).
        framework_commit: Commit of ``apgi-framework`` the equations came from.
        spec_commit: Commit of the specification repository.
        gate_table_path: Path to ``gates/gate-table.yaml``; hashed if it exists.
        feature_version: Feature-definition version.
        platform_release: APGI platform release; defaults to the installed package version.
        extra: Additional fields merged in under ``extra``.

    Raises:
        ValueError: On an unknown ``data_origin`` or ``mode``.
    """
    if data_origin not in DATA_ORIGINS:
        raise ValueError(f"data_origin {data_origin!r} not in {DATA_ORIGINS}")
    if mode not in MODES:
        raise ValueError(f"mode {mode!r} not in {MODES}")

    git = git_revision()
    gate: dict[str, Any] = {"path": None, "sha256": None}
    if gate_table_path is not None:
        gate["path"] = str(gate_table_path)
        if Path(gate_table_path).is_file():
            gate["sha256"] = sha256_file(gate_table_path)

    blockers: list[str] = []
    if seed is None:
        blockers.append("seed is not set")
    if git["commit"] is None:
        blockers.append("implementation commit cannot be resolved")
    elif git["dirty"]:
        blockers.append("working tree has uncommitted changes")
    if not dataset_hash:
        blockers.append("dataset hash is missing")
    if not protocol_hash:
        blockers.append("protocol hash is missing")
    if gate["sha256"] is None:
        blockers.append("gate table is not accessible")

    env = environment_lock()
    identity = {
        "study": study,
        "protocol": [protocol_id, protocol_version, protocol_hash],
        "model": [model_id, model_version],
        "seed": seed,
        "dataset_hash": dataset_hash,
        "commit": git["commit"],
        "env": env["sha256"],
    }
    run_id = "run-" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]

    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "study": study,
        "protocol": {"id": protocol_id, "version": protocol_version, "hash": protocol_hash},
        "model": {"id": model_id, "version": model_version},
        "commits": {
            "implementation": git["commit"],
            "framework": framework_commit,
            "spec": spec_commit,
            "dirty": git["dirty"],
            "branch": git["branch"],
        },
        "dataset_hash": dataset_hash,
        "seed": seed,
        "data_origin": data_origin,
        "mode": mode,
        "env_lock": env,
        "gate_table": gate,
        "feature_version": feature_version,
        "platform_release": platform_release or _package_release(),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "reproducible": not blockers,
        "reproducibility_blockers": blockers,
    }
    if extra:
        manifest["extra"] = dict(extra)
    return manifest


def validate_run_manifest(manifest: Mapping[str, Any]) -> list[str]:
    """Return structural problems with ``manifest`` (empty list means valid)."""
    problems = [f"missing field: {k}" for k in REQUIRED_FIELDS if k not in manifest]
    if problems:
        return problems
    if manifest["data_origin"] not in DATA_ORIGINS:
        problems.append(f"data_origin {manifest['data_origin']!r} invalid")
    if manifest["mode"] not in MODES:
        problems.append(f"mode {manifest['mode']!r} invalid")
    for key, sub in (("protocol", ("id", "version", "hash")), ("model", ("id", "version"))):
        for s in sub:
            if s not in manifest[key]:
                problems.append(f"{key}.{s} missing")
    if manifest["reproducible"] and manifest["reproducibility_blockers"]:
        problems.append("marked reproducible despite listed blockers")
    if manifest["reproducible"] and manifest["seed"] is None:
        problems.append("marked reproducible without a seed")
    return problems


def verify_manifest(manifest: Mapping[str, Any], root: str | Path = ".") -> list[str]:
    """Re-check that referenced files exist and match their recorded hashes.

    Returns:
        Problems found; a manifest claiming ``reproducible`` with any problem here
        is invalid.
    """
    problems = validate_run_manifest(manifest)
    gate = manifest.get("gate_table", {})
    path = gate.get("path")
    if path is not None:
        p = Path(root) / path if not Path(path).is_absolute() else Path(path)
        if not p.is_file():
            problems.append(f"gate table not found: {p}")
        elif gate.get("sha256") and sha256_file(p) != gate["sha256"]:
            problems.append(f"gate table hash mismatch: {p}")
    elif manifest.get("reproducible"):
        problems.append("reproducible run without a gate table path")
    return problems


def write_run_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    """Write ``manifest`` as JSON; refuses an invalid manifest."""
    problems = validate_run_manifest(manifest)
    if problems:
        raise ValueError(f"invalid run manifest: {problems}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
