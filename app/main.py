"""FastAPI application: research jobs API + static frontend."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import settings
from .llm import LLMClient
from .models import Job, JobStatus
from .orchestrator import run_pipeline
from .store import get_job, list_jobs, save_job

app = FastAPI(title="Multi-Agent Research Copilot", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent.parent / "web"


class JobCreate(BaseModel):
    topic: str = Field(min_length=3, max_length=500)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": settings.llm_model,
        "search_backend": settings.search_backend,
        "llm_configured": bool(settings.llm_api_key),
        "max_revision_rounds": settings.max_revision_rounds,
    }


@app.post("/api/jobs", status_code=202)
def create_job(body: JobCreate) -> dict[str, Any]:
    if not settings.llm_api_key:
        raise HTTPException(503, "LLM_API_KEY not configured on this deployment")
    job = Job(topic=body.topic.strip())
    save_job(job)

    def worker() -> None:
        try:
            run_pipeline(job, LLMClient())
        except Exception as e:  # defensive: run_pipeline already catches
            job.status = JobStatus.FAILED
            job.error = str(e)
        save_job(job)

    threading.Thread(target=worker, daemon=True).start()
    return {"id": job.id, "status": job.status.value}


@app.get("/api/jobs")
def jobs_index(limit: int = 20) -> list[dict[str, Any]]:
    out = []
    for j in list_jobs(limit):
        out.append({
            "id": j.id,
            "topic": j.topic,
            "status": j.status.value,
            "created_at": j.created_at,
            "score": j.evaluation.composite_score if j.evaluation else None,
            "grade": j.evaluation.grade if j.evaluation else None,
        })
    return out


@app.get("/api/jobs/{job_id}")
def job_detail(job_id: str) -> JSONResponse:
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return JSONResponse(job.public_dict())


@app.get("/api/jobs/{job_id}/events")
def job_events(job_id: str, after: int = 0) -> dict[str, Any]:
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    events = job.events[after:]
    return {
        "events": [e.model_dump() for e in events],
        "next_cursor": after + len(events),
        "status": job.status.value,
    }


@app.get("/api/jobs/{job_id}/report")
def job_report(job_id: str) -> dict[str, Any]:
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if not job.final_report:
        raise HTTPException(409, f"report not ready (status={job.status.value})")
    return {
        "topic": job.topic,
        "title": job.plan.title if job.plan else job.topic,
        "markdown": job.final_report,
        "sources": [s.model_dump() for s in job.sources],
    }


@app.get("/api/jobs/{job_id}/evaluation")
def job_evaluation(job_id: str) -> dict[str, Any]:
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if not job.evaluation:
        raise HTTPException(409, f"evaluation not ready (status={job.status.value})")
    return job.evaluation.model_dump()


# static frontend
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")
