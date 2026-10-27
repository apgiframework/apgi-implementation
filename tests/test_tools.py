"""Command-line tools: run_k7 (S1-K7-01) and the committed gate table (S1-K7-15)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import run_k7  # noqa: E402

from estimation.composite import COMPOSITE_NAME  # noqa: E402
from estimation.gate_table import allow_list  # noqa: E402
from estimation.recovery_simulation import COVERAGE_RANGE, R_MIN  # noqa: E402
from models.parameters import REDUCED_PARAMETER_NAMES  # noqa: E402

GATES = ROOT / "gates"


def test_run_k7_end_to_end_writes_every_artifact(tmp_path):
    rc = run_k7.main(
        [
            "--out",
            str(tmp_path),
            "--subjects",
            "8",
            "--points",
            "10",
            "--starts",
            "3",
            "--estimator",
            "laplace",
            "--seed",
            "5",
        ]
    )
    assert rc == 0
    names = {p.name for p in tmp_path.iterdir()}
    assert {
        "k7a-results.json",
        "k7c-results.json",
        "gate-table.yaml",
        "gate-table.json",
        "k7-manifest.json",
    } <= names
    k7a = json.loads((tmp_path / "k7a-results.json").read_text())
    assert set(k7a) == {"behavior", "behavior+ignition", "behavior+ignition+amplitude"}
    table = json.loads((tmp_path / "gate-table.json").read_text())
    assert table["design"]["channels"] == ["behavior", "ignition", "amplitude"]
    manifest = json.loads((tmp_path / "k7-manifest.json").read_text())
    assert manifest["k7a_sha256"] == table["meta"]["k7a_sha256"]


def test_run_k7_is_seed_reproducible(tmp_path):
    args = [
        "--subjects",
        "6",
        "--points",
        "8",
        "--starts",
        "2",
        "--estimator",
        "laplace",
        "--seed",
        "9",
    ]
    run_k7.main(["--out", str(tmp_path / "a"), *args])
    run_k7.main(["--out", str(tmp_path / "b"), *args])
    for f in ("k7a-results.json", "k7c-results.json", "gate-table.json"):
        a, b = (
            json.loads((tmp_path / "a" / f).read_text()),
            json.loads((tmp_path / "b" / f).read_text()),
        )
        if f == "gate-table.json":
            a["meta"].pop("git_commit", None)
            b["meta"].pop("git_commit", None)
        assert a == b, f


@pytest.mark.skipif(not (GATES / "gate-table.json").is_file(), reason="gate table not generated")
class TestCommittedGateTable:
    @pytest.fixture(scope="class")
    def table(self):
        return json.loads((GATES / "gate-table.json").read_text())

    def test_every_parameter_carries_all_three_gates_and_a_flag(self, table):
        rows = {r["parameter"]: r for r in table["parameters"]}
        assert set(rows) == set(REDUCED_PARAMETER_NAMES) | {COMPOSITE_NAME}
        for r in rows.values():
            assert r["k7a"] in ("CONFIRMED", "FALSIFIED", "INDETERMINATE")
            assert r["k7c"] in ("CONFIRMED", "FALSIFIED", "INDETERMINATE")
            assert r["k7b"] == "PENDING_DATA"
            assert isinstance(r["exposable"], bool)

    def test_exposable_rows_meet_the_registered_criteria(self, table):
        for r in table["parameters"]:
            if r["exposable"]:
                assert r["k7a"] == r["k7c"] == "CONFIRMED"
                assert r["recovery_r"] >= R_MIN
                assert COVERAGE_RANGE[0] <= r["coverage95"] <= COVERAGE_RANGE[1]

    def test_allow_list_is_consistent(self, table):
        assert table["allow_list"] == allow_list(table)

    def test_yaml_and_json_agree(self, table):
        yaml = pytest.importorskip("yaml")
        assert yaml.safe_load((GATES / "gate-table.yaml").read_text(encoding="utf-8")) == table

    def test_results_files_hash_to_the_recorded_digests(self, table):
        import hashlib

        def digest(obj):
            return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()

        k7a = json.loads((GATES / "k7a-results.json").read_text())
        assert digest(k7a) == table["meta"]["k7a_sha256"]
        k7c = json.loads((GATES / "k7c-results.json").read_text())
        assert digest(k7c) == table["meta"]["k7c_sha256"]

    def test_individual_alpha_and_pi_are_not_exposable(self, table):
        """The α–Π degeneracy is the register's headline expectation; the data must agree."""
        rows = {r["parameter"]: r for r in table["parameters"]}
        assert not rows["alpha"]["exposable"] and not rows["pi"]["exposable"]
        assert rows[COMPOSITE_NAME]["exposable"]
