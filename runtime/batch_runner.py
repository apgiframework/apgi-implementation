"""Batch runner (S1-IMP-11.3).

Runs many sessions — conditions × seeds — and returns results that do not depend
on how the work was scheduled. Seeds come from ``SeedSequence(master).spawn`` by
*job index*, never from worker identity or completion order, so ``workers=1`` and
``workers=8`` give bit-identical results. With an ``output_dir`` each finished
job is written to its own file and skipped on re-run (``resume=True``), which
makes an interrupted batch restartable.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from adapters.canonical_inputs import to_canonical_inputs
from engine.provenance import array_digest

from .session_engine import SessionEngine

__all__ = ["BatchJob", "JobResult", "BatchResult", "make_jobs", "run_job", "run_batch"]


@dataclass(frozen=True)
class BatchJob:
    """One session to run. Must stay picklable (plain data only)."""

    job_id: str
    model_id: str
    params: Mapping[str, Any]
    seed: int
    x_e: tuple[float, ...]
    x_i: tuple[float, ...] | None = None

    def spec_hash(self) -> str:
        """Hash of everything that determines the result."""
        blob = json.dumps(
            [
                self.model_id,
                dict(self.params),
                self.seed,
                list(self.x_e),
                None if self.x_i is None else list(self.x_i),
            ],
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()


@dataclass
class JobResult:
    """Outcome of one job."""

    job_id: str
    spec_hash: str
    output_digest: str
    n_steps: int
    ignition_rate: float
    mean_S: float
    mean_theta: float
    outputs: dict[str, list[float]] | None = field(default=None, repr=False)
    from_cache: bool = False

    def summary(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "spec_hash": self.spec_hash,
            "output_digest": self.output_digest,
            "n_steps": self.n_steps,
            "ignition_rate": self.ignition_rate,
            "mean_S": self.mean_S,
            "mean_theta": self.mean_theta,
        }


@dataclass
class BatchResult:
    """All job results in job order, plus a digest over them."""

    results: list[JobResult]

    @property
    def digest(self) -> str:
        blob = json.dumps([r.summary() for r in self.results], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def by_id(self) -> dict[str, JobResult]:
        return {r.job_id: r for r in self.results}


def make_jobs(
    model_id: str,
    inputs: Any,
    *,
    n_seeds: int,
    master_seed: int,
    params_list: Sequence[Mapping[str, Any]] | None = None,
) -> list[BatchJob]:
    """Cross ``params_list`` with ``n_seeds`` independent seeds.

    Seeds depend only on ``(master_seed, job index)``.
    """
    data = to_canonical_inputs(inputs)
    plist = list(params_list) if params_list is not None else [{}]
    children = np.random.SeedSequence(master_seed).spawn(len(plist) * n_seeds)
    jobs = []
    for pi, params in enumerate(plist):
        for si in range(n_seeds):
            idx = pi * n_seeds + si
            seed = int(children[idx].generate_state(1, dtype=np.uint32)[0])
            jobs.append(
                BatchJob(
                    job_id=f"{model_id}-p{pi:03d}-s{si:03d}",
                    model_id=model_id,
                    params=dict(params),
                    seed=seed,
                    x_e=tuple(float(v) for v in data.x_e),
                    x_i=None if data.x_i is None else tuple(float(v) for v in data.x_i),
                )
            )
    return jobs


def run_job(job: BatchJob, keep_outputs: bool = False) -> JobResult:
    """Execute one job (top-level so it pickles for process pools)."""
    engine = SessionEngine(job.model_id, job.params, job.seed)
    out = engine.run({"x_e": job.x_e, "x_i": job.x_i} if job.x_i is not None else job.x_e)
    s = out.series
    return JobResult(
        job_id=job.job_id,
        spec_hash=job.spec_hash(),
        output_digest=array_digest(s),
        n_steps=out.n_steps,
        ignition_rate=float(s["B"].mean()),
        mean_S=float(s["S"].mean()),
        mean_theta=float(s["theta"].mean()),
        outputs={k: v.tolist() for k, v in s.items()} if keep_outputs else None,
    )


def _load_cached(path: Path, job: BatchJob) -> JobResult | None:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if doc.get("spec_hash") != job.spec_hash():
        return None
    keys = (
        "job_id",
        "spec_hash",
        "output_digest",
        "n_steps",
        "ignition_rate",
        "mean_S",
        "mean_theta",
    )
    try:
        return JobResult(**{k: doc[k] for k in keys}, from_cache=True)
    except KeyError:
        return None


def run_batch(
    jobs: Sequence[BatchJob],
    *,
    workers: int = 1,
    output_dir: str | Path | None = None,
    resume: bool = True,
    keep_outputs: bool = False,
) -> BatchResult:
    """Run ``jobs`` and return results in job order.

    Args:
        jobs: Jobs to run; ``job_id`` values must be unique.
        workers: Process count (1 runs in-process).
        output_dir: If set, write ``<job_id>.json`` per finished job.
        resume: Reuse a stored result when its ``spec_hash`` matches the job.
        keep_outputs: Retain full output arrays in the results (memory-heavy).

    Raises:
        ValueError: On duplicate job ids or ``workers < 1``.
    """
    ids = [j.job_id for j in jobs]
    if len(set(ids)) != len(ids):
        raise ValueError("job_id values must be unique")
    if workers < 1:
        raise ValueError("workers must be >= 1")
    out_dir = None if output_dir is None else Path(output_dir)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, JobResult] = {}
    todo: list[BatchJob] = []
    for job in jobs:
        cached = _load_cached(out_dir / f"{job.job_id}.json", job) if (out_dir and resume) else None
        if cached is not None and not keep_outputs:
            results[job.job_id] = cached
        else:
            todo.append(job)

    if workers == 1 or len(todo) <= 1:
        fresh = [run_job(j, keep_outputs) for j in todo]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            fresh = list(pool.map(run_job, todo, [keep_outputs] * len(todo)))
    for res in fresh:
        results[res.job_id] = res
        if out_dir is not None:
            (out_dir / f"{res.job_id}.json").write_text(
                json.dumps(res.summary(), sort_keys=True) + "\n", encoding="utf-8"
            )
    return BatchResult([results[i] for i in ids])
