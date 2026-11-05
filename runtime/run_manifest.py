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

Identity checks (AUD-DEF-02, AUD-DAT-004, AUD-REP-001, AUD-REP-004). A claim of
reproducibility depends on *verified* objects, not on caller-supplied strings:

* digests must be 64-hex SHA-256 values and commits 40-hex SHAs; a ``None``
  framework or spec commit is a blocker;
* the model must be registered and its version must equal the registry's;
* the protocol must come with a :class:`~runtime.protocol_lock.VerifiedProtocolLock`
  whose lock hash equals ``protocol_hash`` — a bare string is not verification;
* the dataset must carry a provenance record (source, version, checksum,
  license, processing history) whose checksum equals ``dataset_hash``;
* the gate table must exist *and* parse as a gate table;
* ``mode="confirmatory"`` is refused outright without a verified protocol lock
  — an unregistered analysis can never be labelled confirmatory.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.manifest import dependency_versions, git_revision

from .protocol_lock import (
    SHA256_RE,
    ProtocolLockError,
    VerifiedProtocolLock,
    verify_protocol_bundle,
)

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "FEATURE_VERSION",
    "DATA_ORIGINS",
    "MODES",
    "REQUIRED_FIELDS",
    "DATASET_PROVENANCE_FIELDS",
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


#: Fields every dataset provenance record must carry (APGI-DAT-004).
DATASET_PROVENANCE_FIELDS: tuple[str, ...] = ("source", "version", "checksum", "license", "history")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(SHA256_RE.match(value))


def _dataset_problems(dataset: Mapping[str, Any] | None, dataset_hash: str | None) -> list[str]:
    if dataset is None:
        return ["dataset provenance record is missing"]
    out = [
        f"dataset provenance lacks {k!r}"
        for k in DATASET_PROVENANCE_FIELDS
        if dataset.get(k) in (None, "", [])
    ]
    if "history" in dataset and not isinstance(dataset["history"], list):
        out.append("dataset provenance 'history' must be a list of processing steps")
    checksum = dataset.get("checksum")
    if checksum is not None and not _is_sha256(checksum):
        out.append("dataset checksum is not a SHA-256 hex digest")
    elif checksum is not None and checksum != dataset_hash:
        out.append("dataset checksum does not equal dataset_hash")
    return out


def _gate_table_problem(path: Path) -> str | None:
    """Return why ``path`` is not a usable gate table, or ``None`` if it is."""
    candidates = [path] if path.suffix == ".json" else [path.with_suffix(".json"), path]
    for cand in candidates:
        if not cand.is_file():
            continue
        try:
            doc = json.loads(cand.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (
            isinstance(doc, dict)
            and "schema_version" in doc
            and isinstance(doc.get("parameters"), list)
            and all(isinstance(r, dict) and "parameter" in r for r in doc["parameters"])
        ):
            return None
    return "gate table does not parse as a gate table (no JSON twin with schema_version/parameters)"


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
    protocol_lock: VerifiedProtocolLock | None = None,
    dataset: Mapping[str, Any] | None = None,
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
        protocol_lock: Result of :func:`runtime.protocol_lock.verify_protocol_bundle`
            for the definitions this run used (``None`` → blocker).
        dataset: Dataset provenance record with :data:`DATASET_PROVENANCE_FIELDS`
            (``None`` → blocker).

    Raises:
        ValueError: On an unknown ``data_origin`` or ``mode``, or
            ``mode="confirmatory"`` without a verified protocol lock.
    """
    if data_origin not in DATA_ORIGINS:
        raise ValueError(f"data_origin {data_origin!r} not in {DATA_ORIGINS}")
    if mode not in MODES:
        raise ValueError(f"mode {mode!r} not in {MODES}")
    if mode == "confirmatory" and protocol_lock is None:
        raise ValueError(
            "mode='confirmatory' requires a verified protocol lock (APGI-REP-004); "
            "label the run exploratory instead"
        )

    git = git_revision()
    gate: dict[str, Any] = {"path": None, "sha256": None}
    gate_problem: str | None = None
    if gate_table_path is not None:
        gate["path"] = str(gate_table_path)
        if Path(gate_table_path).is_file():
            gate["sha256"] = sha256_file(gate_table_path)
            gate_problem = _gate_table_problem(Path(gate_table_path))

    blockers: list[str] = []
    if seed is None:
        blockers.append("seed is not set")
    if git["commit"] is None:
        blockers.append("implementation commit cannot be resolved")
    elif git["dirty"]:
        blockers.append("working tree has uncommitted changes")
    if not dataset_hash:
        blockers.append("dataset hash is missing")
    elif not _is_sha256(dataset_hash):
        blockers.append("dataset hash is not a SHA-256 hex digest")
    blockers.extend(_dataset_problems(dataset, dataset_hash))
    if not protocol_hash:
        blockers.append("protocol hash is missing")
    elif not _is_sha256(protocol_hash):
        blockers.append("protocol hash is not a SHA-256 hex digest")
    if protocol_lock is None:
        blockers.append("protocol lock not verified")
    else:
        if protocol_hash != protocol_lock.lock_hash:
            blockers.append("protocol hash does not equal the verified lock hash")
        if protocol_version != protocol_lock.protocol_version:
            blockers.append("protocol version does not match the verified lock")
    blockers.extend(_model_problems(model_id, model_version))
    for label, commit in (("framework", framework_commit), ("spec", spec_commit)):
        if commit is None:
            blockers.append(f"{label} commit is not recorded")
        elif not _COMMIT_RE.match(commit):
            blockers.append(f"{label} commit is not a 40-hex git SHA")
    if gate["sha256"] is None:
        blockers.append("gate table is not accessible")
    elif gate_problem is not None:
        blockers.append(gate_problem)

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
        "protocol": {
            "id": protocol_id,
            "version": protocol_version,
            "hash": protocol_hash,
            "lock": None if protocol_lock is None else protocol_lock.as_manifest(),
        },
        "model": {"id": model_id, "version": model_version},
        "commits": {
            "implementation": git["commit"],
            "framework": framework_commit,
            "spec": spec_commit,
            "dirty": git["dirty"],
            "branch": git["branch"],
        },
        "dataset_hash": dataset_hash,
        "dataset": None if dataset is None else json.loads(json.dumps(dict(dataset), default=str)),
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


def _model_problems(model_id: str, model_version: str) -> list[str]:
    from model_registry.variants import get_variant

    try:
        registered = get_variant(model_id).version
    except (KeyError, ValueError):
        return [f"model {model_id!r} is not registered"]
    if registered != model_version:
        return [f"model version {model_version!r} differs from registered {registered!r}"]
    return []


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

    For a manifest that claims ``reproducible``, every identity is re-checked
    (formats, registry membership, dataset provenance, gate-table content) and
    the recorded protocol lock is re-verified against the bundle on disk, so a
    later tamper with any referenced object is detected.

    Returns:
        Problems found; a manifest claiming ``reproducible`` with any problem here
        is invalid.
    """
    problems = validate_run_manifest(manifest)
    if problems:
        return problems
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
    if manifest.get("reproducible"):
        problems.extend(_reverify_identities(manifest, root))
    return problems


def _reverify_identities(manifest: Mapping[str, Any], root: str | Path) -> list[str]:
    out: list[str] = []
    proto = manifest["protocol"]
    if not _is_sha256(proto.get("hash")):
        out.append("protocol hash is not a SHA-256 hex digest")
    if not _is_sha256(manifest.get("dataset_hash")):
        out.append("dataset hash is not a SHA-256 hex digest")
    out.extend(_dataset_problems(manifest.get("dataset"), manifest.get("dataset_hash")))
    out.extend(_model_problems(manifest["model"]["id"], manifest["model"]["version"]))
    commits = manifest.get("commits", {})
    for label in ("implementation", "framework", "spec"):
        c = commits.get(label)
        if not (isinstance(c, str) and _COMMIT_RE.match(c)):
            out.append(f"{label} commit missing or not a 40-hex git SHA")
    gate_path = manifest.get("gate_table", {}).get("path")
    if gate_path is not None:
        gp = Path(root) / gate_path if not Path(gate_path).is_absolute() else Path(gate_path)
        if gp.is_file():
            why = _gate_table_problem(gp)
            if why:
                out.append(why)
    lock = proto.get("lock")
    if not lock:
        out.append("reproducible run without a verified protocol lock")
    else:
        try:
            verified = verify_protocol_bundle(
                lock["lock_path"], lock["bundle_dir"], expected_lock_hash=lock["lock_hash"]
            )
        except (ProtocolLockError, KeyError, TypeError) as exc:
            out.append(f"protocol lock re-verification failed: {exc}")
        else:
            if verified.lock_hash != proto.get("hash"):
                out.append("protocol hash does not equal the verified lock hash")
    return out


def write_run_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    """Write ``manifest`` as JSON; refuses an invalid manifest."""
    problems = validate_run_manifest(manifest)
    if problems:
        raise ValueError(f"invalid run manifest: {problems}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
