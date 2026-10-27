"""Runtime: session engine, checkpoints, batch runner (S1-IMP-11)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from runtime import (
    BatchJob,
    Checkpoint,
    CheckpointCorruptError,
    InputMismatchError,
    ReplayMismatchError,
    SessionEngine,
    load_checkpoint,
    make_jobs,
    run_batch,
    run_job,
    save_checkpoint,
)
from runtime import checkpoint as ckmod

X = np.sin(0.09 * np.arange(70)) + 0.2


class TestSessionEngine:
    def test_run_returns_canonical_outputs(self):
        out = SessionEngine("apgi_reduced", seed=1).run(X)
        assert out.n_steps == 70 and out.model_id == "apgi_reduced" and out.model_version == "1.0.0"

    def test_incremental_equals_batch(self):
        a = SessionEngine("apgi_reduced", seed=2)
        for v in X:
            a.advance(float(v))
        b = SessionEngine("apgi_reduced", seed=2)
        b.run(X)
        assert a.output_digest() == b.output_digest()

    def test_run_continues_where_it_stopped(self):
        e = SessionEngine("apgi_reduced", seed=2)
        e.run(X, max_steps=30)
        assert e.n_done == 30
        e.run(X)
        assert e.n_done == 70
        ref = SessionEngine("apgi_reduced", seed=2)
        ref.run(X)
        assert e.output_digest() == ref.output_digest()

    def test_inconsistent_inputs_refused(self):
        e = SessionEngine("apgi_reduced", seed=2)
        e.run(X, max_steps=30)
        with pytest.raises(InputMismatchError, match="x_e"):
            e.run(X + 1.0)
        with pytest.raises(InputMismatchError, match="consumed 30"):
            e.run(X[:10])

    def test_interoceptive_stream_consistency(self):
        e = SessionEngine("apgi_reduced", seed=1)
        e.advance(0.1, 0.2)
        with pytest.raises(ValueError, match="x_i omitted"):
            e.advance(0.1)
        f = SessionEngine("apgi_reduced", seed=1)
        f.advance(0.1)
        with pytest.raises(ValueError, match="mid-session"):
            f.advance(0.1, 0.2)
        g = SessionEngine("apgi_reduced", seed=1)
        g.run({"x_e": X[:10], "x_i": X[:10]}, max_steps=5)
        with pytest.raises(InputMismatchError, match="x_i"):
            g.run(X[:10])

    def test_no_outputs_before_any_step(self):
        with pytest.raises(ValueError, match="no steps"):
            SessionEngine("apgi_reduced").outputs()

    def test_unknown_model(self):
        with pytest.raises(KeyError):
            SessionEngine("nope")

    def test_checkpoint_option_validation(self, tmp_path):
        with pytest.raises(ValueError, match=">= 1"):
            SessionEngine("apgi_reduced", checkpoint_every=0, checkpoint_path=tmp_path / "c")
        with pytest.raises(ValueError, match="requires checkpoint_path"):
            SessionEngine("apgi_reduced", checkpoint_every=5)

    def test_periodic_checkpoints_allow_crash_recovery(self, tmp_path):
        path = tmp_path / "auto.json"
        e = SessionEngine("apgi_reduced", seed=3, checkpoint_path=path, checkpoint_every=10)
        e.run(X, max_steps=37)  # "crash" after 37 steps
        assert load_checkpoint(path).n_done == 30  # last periodic checkpoint
        resumed = SessionEngine.resume(path)
        resumed.run(X)
        ref = SessionEngine("apgi_reduced", seed=3)
        ref.run(X)
        assert resumed.output_digest() == ref.output_digest()


class TestCheckpoint:
    def _engine(self, model="apgi_reduced"):
        e = SessionEngine(model, seed=4)
        e.run(X, max_steps=25)
        return e

    def test_roundtrip_preserves_everything(self, tmp_path):
        ck = self._engine().checkpoint()
        path = save_checkpoint(ck, tmp_path / "deep" / "ck.json")
        assert load_checkpoint(path) == ck

    def test_snapshot_model_uses_state_replay_model_does_not(self):
        assert self._engine("apgi_reduced").checkpoint().state is not None
        assert self._engine("apgi_reference").checkpoint().state is None

    @pytest.mark.parametrize(
        "edit", ["flip_value", "truncate", "garbage", "version", "missing_key"]
    )
    def test_corruption_is_detected(self, tmp_path, edit):
        path = save_checkpoint(self._engine().checkpoint(), tmp_path / "ck.json")
        text = path.read_text()
        doc = json.loads(text)
        if edit == "flip_value":
            doc["checkpoint"]["n_done"] += 1
            path.write_text(json.dumps(doc))
        elif edit == "truncate":
            path.write_text(text[: len(text) // 2])
        elif edit == "garbage":
            path.write_text("not json")
        elif edit == "version":
            doc["checkpoint"]["version"] = "9"
            doc["sha256"] = ckmod.payload_digest(doc["checkpoint"])
            path.write_text(json.dumps(doc))
        else:
            del doc["sha256"]
            path.write_text(json.dumps(doc))
        with pytest.raises(CheckpointCorruptError):
            load_checkpoint(path)

    def test_missing_file(self, tmp_path):
        with pytest.raises(CheckpointCorruptError, match="unreadable"):
            load_checkpoint(tmp_path / "absent.json")

    def test_write_is_atomic(self, tmp_path):
        path = tmp_path / "ck.json"
        save_checkpoint(self._engine().checkpoint(), path)
        before = path.read_text()
        bad = self._engine().checkpoint()
        bad.history["S"][0] = float("nan")
        with pytest.raises(ValueError):
            save_checkpoint(bad, path)
        assert path.read_text() == before  # the good file survived
        assert [p.name for p in tmp_path.iterdir()] == ["ck.json"]  # no temp litter

    def test_replay_mismatch_is_loud(self, tmp_path):
        e = self._engine("apgi_reference")
        ck = e.checkpoint()
        ck.extra["output_digest"] = "0" * 64
        with pytest.raises(ReplayMismatchError):
            SessionEngine.resume(ck)

    def test_resume_accepts_object_or_path(self, tmp_path):
        e = self._engine()
        a = SessionEngine.resume(e.checkpoint())
        b = SessionEngine.resume(e.save(tmp_path / "c.json"))
        assert a.output_digest() == b.output_digest() == e.output_digest()

    def test_checkpoint_is_plain_json(self):
        json.dumps(self._engine().checkpoint().as_dict(), allow_nan=False)
        assert isinstance(self._engine().checkpoint(), Checkpoint)


class TestBatchRunner:
    def test_job_ids_and_seeds(self):
        jobs = make_jobs(
            "apgi_reduced", X, n_seeds=2, master_seed=1, params_list=[{}, {"beta": 0.0}]
        )
        assert [j.job_id for j in jobs] == [
            "apgi_reduced-p000-s000",
            "apgi_reduced-p000-s001",
            "apgi_reduced-p001-s000",
            "apgi_reduced-p001-s001",
        ]
        assert len({j.seed for j in jobs}) == 4

    def test_results_in_job_order_with_summaries(self):
        jobs = make_jobs("apgi_reduced", X, n_seeds=3, master_seed=1)
        res = run_batch(jobs)
        assert [r.job_id for r in res.results] == [j.job_id for j in jobs]
        assert set(res.by_id()) == {j.job_id for j in jobs}
        for r in res.results:
            assert 0.0 <= r.ignition_rate <= 1.0 and r.n_steps == 70 and len(r.output_digest) == 64

    def test_resume_skips_finished_jobs_and_detects_changed_specs(self, tmp_path):
        jobs = make_jobs("apgi_reduced", X, n_seeds=2, master_seed=1)
        first = run_batch(jobs, output_dir=tmp_path)
        assert not any(r.from_cache for r in first.results)
        again = run_batch(jobs, output_dir=tmp_path)
        assert all(r.from_cache for r in again.results) and again.digest == first.digest
        changed = make_jobs("apgi_reduced", X + 0.5, n_seeds=2, master_seed=1)
        redo = run_batch(changed, output_dir=tmp_path)
        assert not any(r.from_cache for r in redo.results)
        forced = run_batch(jobs, output_dir=tmp_path, resume=False)
        assert not any(r.from_cache for r in forced.results)

    def test_corrupt_cache_entry_is_recomputed(self, tmp_path):
        jobs = make_jobs("apgi_reduced", X, n_seeds=1, master_seed=1)
        run_batch(jobs, output_dir=tmp_path)
        (tmp_path / f"{jobs[0].job_id}.json").write_text("{broken")
        res = run_batch(jobs, output_dir=tmp_path)
        assert not res.results[0].from_cache
        (tmp_path / f"{jobs[0].job_id}.json").write_text(
            json.dumps({"spec_hash": jobs[0].spec_hash()})
        )
        assert not run_batch(jobs, output_dir=tmp_path).results[0].from_cache  # incomplete entry

    def test_keep_outputs(self):
        jobs = make_jobs("apgi_reduced", X, n_seeds=1, master_seed=1)
        r = run_batch(jobs, keep_outputs=True).results[0]
        assert r.outputs is not None and len(r.outputs["S"]) == 70
        assert run_job(jobs[0]).outputs is None

    def test_interoceptive_inputs_flow_through(self):
        jobs = make_jobs(
            "apgi_reference", {"x_e": X[:30], "x_i": 0.1 * X[:30]}, n_seeds=1, master_seed=1
        )
        assert jobs[0].x_i is not None
        assert run_batch(jobs).results[0].n_steps == 30

    def test_validation(self):
        job = BatchJob("a", "apgi_reduced", {}, 1, (0.0, 1.0))
        with pytest.raises(ValueError, match="unique"):
            run_batch([job, job])
        with pytest.raises(ValueError, match="workers"):
            run_batch([job], workers=0)
