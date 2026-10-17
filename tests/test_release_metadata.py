"""The release version must agree across packaging, citation and CLI metadata."""

import re
from pathlib import Path

from main import APGI_VERSION

ROOT = Path(__file__).resolve().parent.parent


def _match(path: str, pattern: str) -> str:
    text = (ROOT / path).read_text(encoding="utf-8")
    found = re.search(pattern, text, flags=re.MULTILINE)
    assert found, f"no version found in {path}"
    return found.group(1)


def test_release_version_is_consistent():
    versions = {
        "pyproject.toml": _match("pyproject.toml", r'^version = "([^"]+)"'),
        "CITATION.cff": _match("CITATION.cff", r'^version: "([^"]+)"'),
        "README.md": _match("README.md", r"^\s*version=\{([^}]+)\}"),
        "CHANGELOG.md": _match("CHANGELOG.md", r"^## \[(\d+\.\d+\.\d+)\]"),
        "main.py": APGI_VERSION,
    }
    assert len(set(versions.values())) == 1, versions
