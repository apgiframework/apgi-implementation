"""Batch runner (S1-IMP-11.3).

Runs many sessions — conditions × seeds — and returns results that do not depend
on how the work was scheduled. Seeds come from ``SeedSequence(master).spawn`` by
*job index*, never from worker identity or completion order, so ``workers=1`` and
``workers=8`` give bit-identical results. With an ``output_dir`` each finished
job is written to its own file and skipped on re-run (``resume=True``), which
makes an interrupted batch restartable.

Cache integrity (AUD-DEF-06). A cached file is reused only when

* its ``cache_identity`` — job spec hash *plus* model version, package version,
  implementation commit and environment-lock digest — equals the current one;
* its ``sha256`` matches the canonical JSON of the stored document; and
* the summary statistics recomputed from the stored output arrays, and the
  output digest over them, equal the stored summary.

Anything else is an explicit invalidation: the job is re-run and the reason is
recorded in :attr:`BatchResult.invalidated`, never silently substituted.
"""

from __future__ import annotations

import functools
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

__all__ = [
    "CACHE_FORMAT",
    "BatchJob",
    "JobResult",
    "BatchResult",
    "cache_identity",
    "make_jobs",
    "run_job",
    "run_batch",
]

CACHE_FORMAT = "2"


@functools.lru_cache(maxsize=1)
def _producer() -> dict[str, Any]:
    """Identity of the producing build (cached: it cannot change mid-process)."""
    from core.manifest import git_revision

    from .run_manifest import environment_lock
    from .session_engine import producer_identity

    git = git_revision()
    return {
        **producer_identity(),
        "implementation_commit": git["commit"],
        "env_lock": environment_lock()["sha256"],
    }


def cache_identity(job: BatchJob) -> str:
    """Digest binding a cached result to the job *and* the code that produced it."""
    from model_registry.variants import get_variant

    blob = json.dumps(
        {
            "cache_format": CACHE_FORMAT,
            "spec_hash": job.spec_hash(),
            "model_version": get_variant(job.model_id).version,
            "producer": _producer(),
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


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
    #: ``{job_id: reason}`` for cached files that were found but rejected.
    invalidated: dict[str, str] = field(default_factory=dict)

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


def _summarise(job: BatchJob, series: Mapping[str, np.ndarray], keep: bool) -> JobResult:
    s = {k: np.asarray(v, dtype=float) for k, v in series.items()}
    return JobResult(
        job_id=job.job_id,
        spec_hash=job.spec_hash(),
        output_digest=array_digest(s),
        n_steps=int(s["B"].shape[0]),
        ignition_rate=float(s["B"].mean()),
        mean_S=float(s["S"].mean()),
        mean_theta=float(s["theta"].mean()),
        outputs={k: v.tolist() for k, v in s.items()} if keep else None,
    )


def run_job(job: BatchJob, keep_outputs: bool = False) -> JobResult:
    """Execute one job (top-level so it pickles for process pools)."""
    engine = SessionEngine(job.model_id, job.params, job.seed)
    out = engine.run({"x_e": job.x_e, "x_i": job.x_i} if job.x_i is not None else job.x_e)
    return _summarise(job, out.series, keep_outputs)


def _doc_digest(body: Mapping[str, Any]) -> str:
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def _write_cached(path: Path, job: BatchJob, res: JobResult, series: Mapping[str, Any]) -> None:
    body = {
        "cache_format": CACHE_FORMAT,
        "cache_identity": cache_identity(job),
        "summary": res.summary(),
        "outputs": {k: [float(v) for v in vals] for k, vals in series.items()},
    }
    doc = {"sha256": _doc_digest(body), "body": body}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _load_cached(path: Path, job: BatchJob) -> tuple[JobResult | None, str | None]:
    """Return ``(result, None)`` for a verified cache hit, else ``(None, reason)``.

    ``reason`` is ``None`` when no cache file exists (a plain miss).
    """
    if not path.exists():
        return None, None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        body = doc["body"]
        stored_sha = doc["sha256"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return None, f"unreadable cache ({exc.__class__.__name__})"
    try:
        if _doc_digest(body) != stored_sha:
            return None, "cache document digest mismatch (modified or truncated)"
    except ValueError:
        return None, "cache document contains non-finite values"
    if body.get("cache_format") != CACHE_FORMAT:
        return None, f"unsupported cache format {body.get('cache_format')!r}"
    if body.get("cache_identity") != cache_identity(job):
        return None, "stale cache: job spec, model, software or environment changed"
    try:
        recomputed = _summarise(job, body["outputs"], keep=False)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        return None, f"cached outputs unusable ({exc.__class__.__name__})"
    if recomputed.summary() != body.get("summary"):
        return None, "cached summary does not match its stored outputs"
    recomputed.from_cache = True
    return recomputed, None


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
        resume: Reuse a stored result only after full verification (see module
            docstring); rejected files are re-run and listed in ``invalidated``.
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
    invalidated: dict[str, str] = {}
    todo: list[BatchJob] = []
    for job in jobs:
        cached = None
        if out_dir is not None and resume:
            cached, reason = _load_cached(out_dir / f"{job.job_id}.json", job)
            if reason is not None:
                invalidated[job.job_id] = reason
        if cached is not None and not keep_outputs:
            results[job.job_id] = cached
        else:
            todo.append(job)

    # Outputs are always retained internally so the cache can store them; they
    # are dropped from the returned results unless ``keep_outputs``.
    need = out_dir is not None or keep_outputs
    if workers == 1 or len(todo) <= 1:
        fresh = [run_job(j, need) for j in todo]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            fresh = list(pool.map(run_job, todo, [need] * len(todo)))
    by_id = {j.job_id: j for j in todo}
    for res in fresh:
        if out_dir is not None and res.outputs is not None:
            _write_cached(out_dir / f"{res.job_id}.json", by_id[res.job_id], res, res.outputs)
        if not keep_outputs:
            res.outputs = None
        results[res.job_id] = res
    return BatchResult([results[i] for i in ids], invalidated)
