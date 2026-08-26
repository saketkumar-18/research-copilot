"""In-memory + disk job store. Jobs persist to data/jobs/<id>.json so a
restart doesn't lose finished reports (Render free tier restarts often)."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Optional

from .models import Job

DATA_DIR = Path(os.environ.get("JOBS_DIR", "data/jobs"))
_lock = threading.Lock()
_jobs: dict[str, Job] = {}
_loaded = False


def _load_from_disk() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    if not DATA_DIR.exists():
        return
    for p in sorted(DATA_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:50]:
        try:
            job = Job.model_validate(json.loads(p.read_text(encoding="utf-8")))
            # Never clobber a live in-memory job (one a worker thread is
            # actively mutating) with a stale disk copy.
            if job.id not in _jobs:
                _jobs[job.id] = job
        except Exception:
            continue


def save_job(job: Job) -> None:
    with _lock:
        _jobs[job.id] = job
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            (DATA_DIR / f"{job.id}.json").write_text(
                job.model_dump_json(indent=None), encoding="utf-8"
            )
        except Exception:
            pass  # disk persistence is best-effort


def get_job(job_id: str) -> Optional[Job]:
    with _lock:
        _load_from_disk()
        return _jobs.get(job_id)


def list_jobs(limit: int = 20) -> list[Job]:
    with _lock:
        _load_from_disk()
        jobs = sorted(_jobs.values(), key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]
