"""Dependency policy (S1-IMP-19) and deprecation drift (S1-IMP-20)."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import check_dependencies as cd  # noqa: E402


def _errors(findings):
    return [f for f in findings if f.severity == "error"]


def test_repository_satisfies_its_own_policy():
    findings = cd.run_checks(ROOT)
    assert _errors(findings) == [], "\n".join(map(str, findings))


def test_matrix_lists_every_deprecation_and_nothing_else():
    from adapters import DEPRECATIONS

    rows = cd.parse_matrix_deprecations((ROOT / "DEPENDENCY-MATRIX.md").read_text(encoding="utf-8"))
    assert {r["legacy name"] for r in rows} == {d.old for d in DEPRECATIONS}


def _project(tmp_path: Path, files: dict[str, str]) -> Path:
    for rel, body in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(body), encoding="utf-8")
    return tmp_path


class TestCycles:
    def test_import_time_cycle_is_an_error(self, tmp_path):
        root = _project(tmp_path, {"a.py": "import b\n", "b.py": "import a\n"})
        errs = _errors(cd.check_cycles(root))
        assert len(errs) == 1 and "a <-> b" in errs[0].message

    def test_lazy_cycle_is_only_a_warning(self, tmp_path):
        root = _project(tmp_path, {"a.py": "import b\n", "b.py": "def f():\n    import a\n"})
        findings = cd.check_cycles(root)
        assert _errors(findings) == [] and [f.check for f in findings] == ["lazy-cycle"]

    def test_type_checking_imports_ignored(self, tmp_path):
        root = _project(
            tmp_path,
            {
                "a.py": "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import b\n",
                "b.py": "import a\n",
            },
        )
        assert cd.check_cycles(root) == []

    def test_package_relative_imports_resolved(self, tmp_path):
        root = _project(
            tmp_path,
            {
                "pkg/__init__.py": "",
                "pkg/x.py": "from . import y\n",
                "pkg/y.py": "from .x import something\n",
            },
        )
        assert any("pkg.x" in f.message for f in _errors(cd.check_cycles(root)))

    def test_acyclic_graph_is_clean(self, tmp_path):
        root = _project(
            tmp_path, {"a.py": "import b\nimport c\n", "b.py": "import c\n", "c.py": "x = 1\n"}
        )
        assert cd.check_cycles(root) == []


class TestForbiddenAndCopies:
    def test_downstream_import_is_an_error(self, tmp_path):
        root = _project(
            tmp_path, {"a.py": "import apgi_system.score\n", "b.py": "from apgi_system import x\n"}
        )
        assert len(_errors(cd.check_forbidden_imports(root))) == 2

    def test_copied_model_file_detected_even_with_different_comments(self, tmp_path):
        body = "\n".join(f"def f{i}(x):\n    return x + {i}\n" for i in range(12))
        root = _project(
            tmp_path,
            {
                "models/m.py": "# original\n" + body,
                "other/m_copy.py": "# a copy, reformatted comment\n\n" + body,
            },
        )
        errs = _errors(cd.check_copied_models(root))
        assert len(errs) == 1 and "duplicates" in errs[0].message

    def test_short_files_and_inits_not_flagged(self, tmp_path):
        root = _project(
            tmp_path,
            {
                "a/__init__.py": "x = 1\n" * 30,
                "b/__init__.py": "x = 1\n" * 30,
                "c.py": "y = 1\n",
                "d.py": "y = 1\n",
            },
        )
        assert cd.check_copied_models(root) == []

    def test_vendored_tree_detected(self, tmp_path):
        root = _project(tmp_path, {"apgi_framework/__init__.py": ""})
        assert any(f.check == "vendored" for f in cd.check_copied_models(root))


class TestPins:
    def test_unbounded_dependency_flagged(self, tmp_path):
        root = _project(
            tmp_path,
            {
                "pyproject.toml": """
                    [build-system]
                    requires = ["setuptools>=77.0"]
                    [project]
                    name = "x"
                    dependencies = ["numpy>=1.24,<2.0", "scipy>=1.10"]
                    [project.optional-dependencies]
                    dev = ["ruff==0.16.0", "pytest>=7"]
                """,
                "requirements.txt": "numpy>=1.24,<2.0\nmatplotlib>=3.7  # comment\n",
            },
        )
        msgs = [f.message for f in _errors(cd.check_pins(root))]
        assert any("scipy>=1.10" in m for m in msgs)
        assert any("pytest>=7" in m for m in msgs)
        assert any("matplotlib>=3.7" in m for m in msgs)
        assert not any("setuptools" in m or "ruff" in m or "numpy" in m for m in msgs)

    def test_pin_forms(self):
        assert cd._pinned(">=1,<2") and cd._pinned("==1.0") and cd._pinned("~=1.4")
        assert not cd._pinned(">=1") and not cd._pinned("")


class TestDeprecationDrift:
    def _matrix(self, rows: str) -> str:
        return (
            "# m\n\n## Deprecations\n\n"
            "| Legacy name | Canonical name | Kind | Canonical owner | Adapter | Deprecated in | Removal release |\n"
            "| --- | --- | --- | --- | --- | --- | --- |\n" + rows + "\n"
        )

    def _setup(self, tmp_path, rows):
        root = _project(tmp_path, {"DEPENDENCY-MATRIX.md": self._matrix(rows)})
        (root / "adapters").mkdir()
        (root / "adapters" / "version_adapter.py").write_text("")
        (root / "adapters" / "__init__.py").write_text("")
        return root

    def test_missing_matrix(self, tmp_path):
        assert [f.message for f in cd.check_deprecations(tmp_path)] == [
            "DEPENDENCY-MATRIX.md is missing"
        ]

    def test_parse_matrix(self):
        rows = cd.parse_matrix_deprecations(
            self._matrix("| `a` | `b` | config | o | `adapters.version_adapter` | 1.1.0 | 2.0.0 |")
        )
        assert rows == [
            {
                "legacy name": "a",
                "canonical name": "b",
                "kind": "config",
                "canonical owner": "o",
                "adapter": "adapters.version_adapter",
                "deprecated in": "1.1.0",
                "removal release": "2.0.0",
            }
        ]
        assert cd.parse_matrix_deprecations("# no section here") == []

    def test_drift_is_reported_against_the_real_adapter_table(self, tmp_path):
        from adapters import DEPRECATIONS

        d0 = DEPRECATIONS[0]
        rows = "\n".join(
            f"| `{d.old}` | `{d.new if d is not d0 else 'WRONG'}` | {d.kind} | {d.owner} | `adapters.version_adapter` | {d.deprecated_in} | {d.removal_in} |"
            for d in DEPRECATIONS[:-1]
        )
        rows += "\n| `ghost` | `x` | config | o | `adapters.version_adapter` | 1.1.0 | 2.0.0 |"
        root = self._setup(tmp_path, rows)
        msgs = [f.message for f in cd.check_deprecations(root)]
        assert any(f"{d0.old!r}: matrix canonical name='WRONG'" in m for m in msgs)
        assert any(f"{DEPRECATIONS[-1].old!r} is deprecated in code but absent" in m for m in msgs)
        assert any(
            "'ghost' is in DEPENDENCY-MATRIX.md but has no DEPRECATIONS entry" in m for m in msgs
        )

    def test_missing_adapter_module_reported(self, tmp_path):
        from adapters import DEPRECATIONS

        rows = "\n".join(
            f"| `{d.old}` | `{d.new}` | {d.kind} | {d.owner} | `adapters.nowhere` | {d.deprecated_in} | {d.removal_in} |"
            for d in DEPRECATIONS
        )
        root = self._setup(tmp_path, rows)
        assert any(
            "adapter module 'adapters.nowhere' not found" in f.message
            for f in cd.check_deprecations(root)
        )


def test_main_exit_codes(tmp_path, capsys):
    assert cd.main(["--root", str(ROOT)]) == 0
    assert "0 error(s)" in capsys.readouterr().out
    bad = _project(
        tmp_path, {"a.py": "import b\n", "b.py": "import a\n", "DEPENDENCY-MATRIX.md": "# x\n"}
    )
    assert cd.main(["--root", str(bad)]) == 1
