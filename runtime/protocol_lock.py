"""Protocol lock verification (APGI-REP-001, AUD-REP-001).

Every component that loads Protocol v1.2 definitions must verify each
definition file against the protocol lock manifest *before use* and refuse to
run — producing no output — when any hash does not match.

Lock manifest format (JSON)::

    {
      "lock_version": "1",
      "protocol_version": "1.2",
      "bundle": "<bundle identifier / source revision>",
      "files": {"protocol_0.json": "<sha256>", ..., "protocol_7.json": "<sha256>"},
      "timestamp_proof": "<OpenTimestamps proof reference or null>"
    }

The lock's own identity (:attr:`VerifiedProtocolLock.lock_hash`) is the SHA-256
of its canonical JSON. A caller may pin that hash (``expected_lock_hash``) so a
substituted lock is refused as well as a substituted definition.

Definitions are parsed from the *same bytes* that were hashed, so a file cannot
change between verification and use.

The approved Protocol v1.2 bundle and its lock are owned outside this
repository (S1-REG / AUD-PRO-001); this module is the enforcement mechanism and
is exercised in tests with locally generated bundles.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "LOCK_VERSION",
    "SHA256_RE",
    "ProtocolLockError",
    "VerifiedProtocolLock",
    "build_lock",
    "lock_digest",
    "verify_protocol_bundle",
]

LOCK_VERSION = "1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DEFINITION_GLOB = "protocol_*.json"


class ProtocolLockError(RuntimeError):
    """The protocol bundle does not match its lock; the run must not proceed."""


@dataclass(frozen=True)
class VerifiedProtocolLock:
    """Proof that a bundle matched its lock at load time."""

    lock_path: str
    bundle_dir: str
    lock_hash: str
    protocol_version: str
    bundle: str
    files: Mapping[str, str]
    timestamp_proof: str | None
    definitions: Mapping[str, Any] = field(repr=False, default_factory=dict)

    def as_manifest(self) -> dict[str, Any]:
        """Fields recorded in a run manifest (definitions excluded)."""
        return {
            "lock_path": self.lock_path,
            "bundle_dir": self.bundle_dir,
            "lock_hash": self.lock_hash,
            "protocol_version": self.protocol_version,
            "bundle": self.bundle,
            "files": dict(self.files),
            "timestamp_proof": self.timestamp_proof,
        }


def lock_digest(lock: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON of a lock document."""
    blob = json.dumps(lock, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_lock(
    bundle_dir: str | Path,
    *,
    protocol_version: str,
    bundle: str,
    timestamp_proof: str | None = None,
) -> dict[str, Any]:
    """Create a lock document for the definitions currently in ``bundle_dir``.

    For tooling and tests: an *approved* lock comes from the protocol owner.
    """
    root = Path(bundle_dir)
    files = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.glob(_DEFINITION_GLOB))
    }
    if not files:
        raise ProtocolLockError(f"no {_DEFINITION_GLOB} definitions in {root}")
    return {
        "lock_version": LOCK_VERSION,
        "protocol_version": protocol_version,
        "bundle": bundle,
        "files": files,
        "timestamp_proof": timestamp_proof,
    }


def _read_lock(lock_path: Path) -> dict[str, Any]:
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProtocolLockError(f"protocol lock unreadable: {lock_path} ({exc})") from exc
    if not isinstance(lock, dict) or lock.get("lock_version") != LOCK_VERSION:
        raise ProtocolLockError(f"unsupported protocol lock format in {lock_path}")
    files = lock.get("files")
    if not isinstance(files, dict) or not files:
        raise ProtocolLockError("protocol lock lists no definition files")
    for name, digest in files.items():
        if not isinstance(digest, str) or not SHA256_RE.match(digest):
            raise ProtocolLockError(f"lock entry for {name!r} is not a SHA-256 hex digest")
        if Path(name).name != name:
            raise ProtocolLockError(f"lock entry {name!r} must be a bare file name")
    for key in ("protocol_version", "bundle"):
        if not isinstance(lock.get(key), str) or not lock[key]:
            raise ProtocolLockError(f"protocol lock missing {key!r}")
    return lock


def verify_protocol_bundle(
    lock_path: str | Path,
    bundle_dir: str | Path,
    *,
    expected_lock_hash: str | None = None,
    expected_protocol_version: str | None = None,
) -> VerifiedProtocolLock:
    """Verify every definition in ``bundle_dir`` against the lock; fail closed.

    Raises:
        ProtocolLockError: If the lock is unreadable or not the pinned one; a
            listed file is missing or its bytes do not hash to the locked value;
            an unlisted ``protocol_*.json`` is present; a definition is not
            valid JSON; or the protocol version differs from the expected one.
    """
    lpath, root = Path(lock_path), Path(bundle_dir)
    lock = _read_lock(lpath)
    lhash = lock_digest(lock)
    if expected_lock_hash is not None and lhash != expected_lock_hash:
        raise ProtocolLockError(
            f"protocol lock hash {lhash} is not the pinned {expected_lock_hash}"
        )
    if (
        expected_protocol_version is not None
        and lock["protocol_version"] != expected_protocol_version
    ):
        raise ProtocolLockError(
            f"lock is for protocol {lock['protocol_version']!r}, "
            f"expected {expected_protocol_version!r}"
        )
    present = {p.name for p in root.glob(_DEFINITION_GLOB)}
    unlisted = sorted(present - set(lock["files"]))
    if unlisted:
        raise ProtocolLockError(f"definition files not covered by the lock: {unlisted}")
    definitions: dict[str, Any] = {}
    for name, expected in sorted(lock["files"].items()):
        path = root / name
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise ProtocolLockError(f"locked definition missing: {path}") from exc
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise ProtocolLockError(
                f"definition {name} hash {actual} does not match locked {expected}"
            )
        try:
            definitions[name] = json.loads(data.decode("utf-8"))
        except ValueError as exc:
            raise ProtocolLockError(f"definition {name} is not valid JSON") from exc
    return VerifiedProtocolLock(
        lock_path=str(lpath),
        bundle_dir=str(root),
        lock_hash=lhash,
        protocol_version=lock["protocol_version"],
        bundle=lock["bundle"],
        files=dict(lock["files"]),
        timestamp_proof=lock.get("timestamp_proof"),
        definitions=definitions,
    )
