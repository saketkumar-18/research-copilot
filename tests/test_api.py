"""API tests using FastAPI TestClient with a stubbed pipeline."""
from __future__ import annotations

import os
from unittest.mock import patch

os.environ.setdefault("JOBS_DIR", "data/test_jobs")

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import Evaluation, Job, JobStatus, Plan, Source
from app import store


@pytest.fixture(autouse=True)
def clean_store(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    with store._lock:
        store._jobs.clear()
        store._loaded = True
    yield


@pytest.fixture
def client():
    return TestClient(app)


def _finished_job(topic="topic X") -> Job:
    job = Job(topic=topic)
    job.status = JobStatus.DONE
    job.plan = Plan(title="T", sections=["A"], search_queries=["q"], key_questions=["k"])
    job.sources = [Source(id=1, url="https://a.com", title="A", fetched=True)]
    job.final_report = "# T\n\n## A\nClaim [1].\n\n## References\n[1] A — https://a.com"
    job.evaluation = Evaluation(composite_score=82.5, grade="B")
    job.emit("orchestrator", "done", "finished")
    return job


class TestAPI:
    def test_health(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "model" in body

    def test_create_job_requires_key(self, client, monkeypatch):
        from app import main as main_mod
        monkeypatch.setattr(main_mod.settings, "llm_api_key", "")
        r = client.post("/api/jobs", json={"topic": "anything"})
        assert r.status_code == 503

    def test_create_and_poll_job(self, client, monkeypatch):
        from app import main as main_mod
        monkeypatch.setattr(main_mod.settings, "llm_api_key", "test-key")

        def fake_pipeline(job, llm=None):
            job.status = JobStatus.DONE
            job.final_report = "# R\n\n## References\n"
            job.evaluation = Evaluation(composite_score=75.0, grade="B")
            return job

        with patch.object(main_mod, "run_pipeline", side_effect=fake_pipeline):
            r = client.post("/api/jobs", json={"topic": "test topic"})
            assert r.status_code == 202
            job_id = r.json()["id"]

        import time
        for _ in range(40):
            d = client.get(f"/api/jobs/{job_id}").json()
            if d["status"] in ("done", "failed"):
                break
            time.sleep(0.1)
        assert d["status"] == "done"

        # events endpoint
        ev = client.get(f"/api/jobs/{job_id}/events").json()
        assert ev["status"] == "done"

        # report + evaluation
        rep = client.get(f"/api/jobs/{job_id}/report").json()
        assert rep["markdown"].startswith("# R")
        evl = client.get(f"/api/jobs/{job_id}/evaluation").json()
        assert evl["composite_score"] == 75.0

    def test_job_not_found(self, client):
        assert client.get("/api/jobs/nope").status_code == 404
        assert client.get("/api/jobs/nope/report").status_code == 404

    def test_report_not_ready(self, client):
        job = Job(topic="pending")
        store.save_job(job)
        r = client.get(f"/api/jobs/{job.id}/report")
        assert r.status_code == 409

    def test_jobs_index(self, client):
        store.save_job(_finished_job("one"))
        store.save_job(_finished_job("two"))
        rows = client.get("/api/jobs").json()
        assert len(rows) == 2
        assert all(r["grade"] == "B" for r in rows)

    def test_topic_validation(self, client):
        r = client.post("/api/jobs", json={"topic": "ab"})
        assert r.status_code == 422

    def test_index_html_served(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "Research Copilot" in r.text
