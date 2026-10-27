"""Protocol compatibility (S1-IMP-13.2).

Checks that the artefacts exchanged with protocols and downstream consumers — canonical
inputs and outputs, versioned documents, run manifests, checkpoints, gate tables — keep
their contracts, and that retiring a legacy name never changes a result.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from adapters import (
    DEPRECATIONS,
    OUTPUT_SCHEMA,
    ConflictingAliasError,
    adapt_config,
    adapt_output_keys,
    list_of_steps_to_run,
    migrate,
    register_migration,
    to_canonical_inputs,
    to_canonical_outputs,
)
from adapters import version_adapter as va
from config import CONFIG
from model_registry import create_model
from pipeline import APGIPipeline
from runtime.run_manifest import (
    DATA_ORIGINS,
    MODES,
    REQUIRED_FIELDS,
    build_run_manifest,
    validate_run_manifest,
    verify_manifest,
    write_run_manifest,
)

pytestmark = pytest.mark.conformance


class TestCanonicalInputs:
    def test_accepts_every_documented_shape(self):
        x = [0.1, 0.2, 0.3]
        shapes = [
            x,
            np.array(x),
            np.array([[0.1], [0.2], [0.3]]),
            np.array([[0.1, 1.0], [0.2, 1.0], [0.3, 1.0]]),
            {"x_e": x},
            {"stimulus": x, "interoceptive": [0, 0, 1], "dt": 0.5},
        ]
        for s in shapes:
            ci = to_canonical_inputs(s)
            np.testing.assert_allclose(ci.x_e, x)
            assert ci.n_steps == 3

    def test_two_column_array_splits_streams(self):
        ci = to_canonical_inputs(np.array([[1.0, 5.0], [2.0, 6.0]]))
        np.testing.assert_array_equal(ci.x_i, [5.0, 6.0])

    def test_dataframe_like(self):
        pd = pytest.importorskip("pandas")
        ci = to_canonical_inputs(pd.DataFrame({"x_e": [1.0, 2.0], "x_i": [0.0, 1.0]}))
        np.testing.assert_array_equal(ci.x_i, [0.0, 1.0])

    def test_idempotent_on_canonical(self):
        ci = to_canonical_inputs({"x_e": [1, 2], "dt": 0.25})
        again = to_canonical_inputs(ci)
        np.testing.assert_array_equal(again.x_e, ci.x_e)
        assert again.dt == 0.25
        assert to_canonical_inputs(ci, dt=0.5).dt == 0.5

    @pytest.mark.parametrize(
        "bad,match",
        [
            ([], "empty"),
            ([1.0, float("nan")], "non-finite"),
            ([1.0, float("inf")], "non-finite"),
            (np.zeros((2, 3)), "1 or 2 columns"),
            (np.zeros((2, 2, 2)), "1-D"),
            ({"x_e": [1, 2], "x_i": [1]}, "does not match"),
            (["a", "b"], "convert"),
        ],
    )
    def test_invalid_inputs_fail_loudly(self, bad, match):
        with pytest.raises(ValueError, match=match):
            to_canonical_inputs(bad)

    def test_missing_exteroceptive_field(self):
        with pytest.raises(KeyError, match="exteroceptive"):
            to_canonical_inputs({"foo": [1]})

    @pytest.mark.parametrize("dt", [0.0, -1.0, float("nan")])
    def test_bad_dt(self, dt):
        with pytest.raises(ValueError, match="dt"):
            to_canonical_inputs([1.0], dt=dt)

    def test_nan_allowed_only_when_requested(self):
        assert np.isnan(to_canonical_inputs([1.0, float("nan")], allow_nan=True).x_e[1])
        with pytest.raises(ValueError):
            to_canonical_inputs([1.0, float("inf")], allow_nan=True)

    def test_as_dict_is_json(self):
        json.dumps(to_canonical_inputs({"x_e": [1, 2], "x_i": [0, 0], "dt": 1}).as_dict())


class TestCanonicalOutputs:
    def _run(self, model_id="apgi_reduced"):
        return create_model(model_id, seed=1).run(np.sin(np.arange(30) * 0.2))

    @pytest.mark.parametrize("model_id", ["apgi_reduced", "apgi_reference", "apgi_null"])
    def test_every_model_maps_to_schema(self, model_id):
        out = to_canonical_outputs(self._run(model_id), model_id=model_id, model_version="1")
        assert out.n_steps == 30
        assert {"S", "theta", "B", "p_ignite"} <= set(out.series)
        assert len(out.to_records()) == 30
        assert json.loads(out.to_json())["schema_version"] == "1"

    def test_aliases_and_units(self):
        run = self._run()
        legacy = {
            "signal": run["S"],
            "threshold": run["theta"],
            "ignition": run["B"],
            "p_ign": run["p_ignite"],
        }
        out = to_canonical_outputs(legacy)
        np.testing.assert_array_equal(out.series["S"], run["S"])
        assert out.as_dict()["units"]["B"] == OUTPUT_SCHEMA["B"][0]

    def test_required_fields_enforced(self):
        run = self._run()
        del run["theta"]
        with pytest.raises(ValueError, match="missing required"):
            to_canonical_outputs(run)

    def test_invariants_enforced(self):
        run = self._run()
        with pytest.raises(ValueError, match="binary"):
            to_canonical_outputs({**run, "B": run["B"] + 0.5})
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            to_canonical_outputs({**run, "p_ignite": run["p_ignite"] + 1.0})
        bad = run["S"].copy()
        bad[3] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            to_canonical_outputs({**run, "S": bad})
        with pytest.raises(ValueError, match="different lengths"):
            to_canonical_outputs({**run, "S": run["S"][:-1]})
        with pytest.raises(ValueError, match="conflicting"):
            to_canonical_outputs({**run, "signal": run["S"] + 1})

    def test_extras_dropped_unless_requested(self):
        run = {**self._run(), "custom": np.arange(30.0), "scalar": 1.0}
        assert "custom" not in to_canonical_outputs(run).series
        assert "custom" in to_canonical_outputs(run, keep_extra=True).series

    def test_steps_transposition(self):
        steps = [create_model("apgi_reduced", seed=1).step(0.5) for _ in range(3)]
        run = list_of_steps_to_run(steps)
        assert run["S"].shape == (3,)
        with pytest.raises(ValueError):
            list_of_steps_to_run([])


class TestVersionAdapter:
    def test_every_legacy_name_warns_once_and_maps(self):
        for d in (x for x in DEPRECATIONS if x.kind == "config"):
            with pytest.warns(DeprecationWarning, match=f"{d.old!r}.*{d.removal_in}.*{d.new!r}"):
                out, applied = adapt_config({d.old: 1.0})
            assert out == {d.new: 1.0} and applied == [d]

    def test_output_keys(self):
        with pytest.warns(DeprecationWarning, match="ai_efe_values"):
            out, _ = adapt_output_keys({"ai_F_values": [1]})
        assert out == {"ai_efe_values": [1]}

    def test_no_warning_for_canonical_or_default_config(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            adapt_config({"beta_da": 1.0, "tau_sigma": 0.5, "strict": True})
            adapt_config(CONFIG)  # ships both spellings with equal values

    def test_conflicting_spellings_refused(self):
        with pytest.raises(ConflictingAliasError):
            adapt_config({"beta": 1.0, "beta_da": 2.0}, warn=False)

    def test_input_not_mutated(self):
        cfg = {"beta": 1.0}
        adapt_config(cfg, warn=False)
        assert cfg == {"beta": 1.0}

    def test_adapting_never_changes_the_result(self):
        legacy = {
            **{k: v for k, v in CONFIG.items() if k not in ("beta_da", "tau_sigma")},
            "seed": 5,
            "beta": 0.9,
            "ignite_tau": 0.4,
        }
        with pytest.warns(DeprecationWarning):
            adapted, _ = adapt_config(legacy)
        x = np.sin(np.arange(60) * 0.2)
        runs = []
        for cfg in (legacy, adapted):
            p = APGIPipeline(dict(cfg))
            runs.append([p.step(float(v), 0.0)["S"] for v in x])
        assert runs[0] == runs[1]

    def test_migration_chain(self):
        register_migration("t-a", "t-b", lambda d: {**d, "b": 1})
        register_migration("t-b", "t-c", lambda d: {**d, "c": 2})
        try:
            assert migrate({"a": 0}, "t-a", "t-c") == {"a": 0, "b": 1, "c": 2}
            assert migrate({"a": 0}, "t-a", "t-a") == {"a": 0}
            with pytest.raises(LookupError):
                migrate({}, "t-c", "t-a")
            with pytest.raises(ValueError, match="already registered"):
                register_migration("t-a", "t-b", lambda d: d)
        finally:
            va._MIGRATIONS.pop(("t-a", "t-b"), None)
            va._MIGRATIONS.pop(("t-b", "t-c"), None)

    def test_deprecated_names_listing(self):
        assert "beta" in va.deprecated_names("config") and "ai_F_values" in va.deprecated_names(
            "output"
        )
        assert va.deprecated_names() == sorted(d.old for d in DEPRECATIONS)


def _manifest(**over):
    kw = {
        "study": "S1",
        "protocol_id": "P-001",
        "protocol_version": "1.0",
        "protocol_hash": "ab" * 32,
        "model_id": "apgi_reduced",
        "model_version": "1.0.0",
        "seed": 1,
        "data_origin": "synthetic",
        "mode": "desk",
        "dataset_hash": "cd" * 32,
    }
    kw.update(over)
    return build_run_manifest(**kw)


class TestRunManifest:
    def test_carries_every_lattice_stamp(self):
        m = _manifest()
        assert set(REQUIRED_FIELDS) <= set(m)
        assert validate_run_manifest(m) == []
        assert m["protocol"] == {"id": "P-001", "version": "1.0", "hash": "ab" * 32}
        assert m["env_lock"]["sha256"] and m["run_id"].startswith("run-")

    def test_not_reproducible_until_sources_accessible(self, tmp_path):
        m = _manifest(seed=None, dataset_hash=None, protocol_hash=None)
        assert m["reproducible"] is False
        blockers = " ".join(m["reproducibility_blockers"])
        for needle in ("seed", "dataset hash", "protocol hash", "gate table"):
            assert needle in blockers

    def test_gate_table_hash_recorded_and_verified(self, tmp_path):
        gate = tmp_path / "gate-table.yaml"
        gate.write_text("a: 1\n")
        m = _manifest(gate_table_path=gate)
        assert "gate table is not accessible" not in m["reproducibility_blockers"]
        assert verify_manifest(m) == []
        gate.write_text("a: 2\n")
        assert any("hash mismatch" in p for p in verify_manifest(m))
        gate.unlink()
        assert any("not found" in p for p in verify_manifest(m))

    def test_run_id_is_stable_for_same_inputs(self):
        assert _manifest()["run_id"] == _manifest()["run_id"]
        assert _manifest()["run_id"] != _manifest(seed=2)["run_id"]

    def test_unknown_origin_or_mode_rejected(self):
        with pytest.raises(ValueError, match="data_origin"):
            _manifest(data_origin="made-up")
        with pytest.raises(ValueError, match="mode"):
            _manifest(mode="made-up")
        assert "synthetic" in DATA_ORIGINS and "desk" in MODES

    def test_validate_flags_inconsistent_manifest(self):
        m = _manifest()
        m["reproducible"] = True
        m["reproducibility_blockers"] = ["x"]
        assert any("despite" in p for p in validate_run_manifest(m))
        m["reproducibility_blockers"] = []
        m["seed"] = None
        assert any("without a seed" in p for p in validate_run_manifest(m))
        del m["study"]
        assert validate_run_manifest(m) == ["missing field: study"]

    def test_write_refuses_invalid_and_roundtrips_valid(self, tmp_path):
        m = _manifest()
        path = write_run_manifest(tmp_path / "sub" / "m.json", m)
        assert json.loads(path.read_text())["run_id"] == m["run_id"]
        bad = dict(m)
        del bad["mode"]
        with pytest.raises(ValueError, match="invalid run manifest"):
            write_run_manifest(tmp_path / "bad.json", bad)

    def test_reproducible_without_gate_path_is_invalid_on_verify(self):
        m = _manifest()
        m["reproducible"] = True
        m["reproducibility_blockers"] = []
        assert any("gate table path" in p for p in verify_manifest(m))
