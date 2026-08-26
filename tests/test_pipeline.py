"""End-to-end pipeline tests with mocked LLM + mocked search."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from app.models import Job, JobStatus, Source
from app.orchestrator import run_pipeline
from tests.conftest import FakeLLM


def _mock_gather(queries, max_pages=None):
    return [
        Source(id=1, url="https://example.com/1", title="Source One",
               snippet="about topic X", content="Full text one.", fetched=True, query=queries[0]),
        Source(id=2, url="https://example.com/2", title="Source Two",
               snippet="more topic X", content="Full text two.", fetched=True, query=queries[0]),
    ]


class TestPipeline:
    @patch("app.agents.gather_sources", side_effect=_mock_gather)
    def test_full_run_approve_first_pass(self, _mock):
        job = Job(topic="topic X")
        llm = FakeLLM(critic_verdicts=["approve"])
        run_pipeline(job, llm)

        assert job.status == JobStatus.DONE, job.error
        assert job.plan is not None and len(job.plan.sections) == 4
        assert len(job.sources) == 2
        assert job.final_report.startswith("# Test Survey")
        assert "## References" in job.final_report
        assert job.revision_rounds == 0
        assert job.evaluation is not None
        assert 0 <= job.evaluation.composite_score <= 100
        assert job.evaluation.grade in "ABCDF"
        # timeline events from every agent
        agents_seen = {e.agent for e in job.events}
        assert {"planner", "searcher", "writer", "critic", "evaluator"} <= agents_seen
        assert job.timings["total_s"] >= 0

    @patch("app.agents.gather_sources", side_effect=_mock_gather)
    def test_revision_loop_then_approve(self, _mock):
        job = Job(topic="topic X")
        llm = FakeLLM(critic_verdicts=["revise", "approve"])
        run_pipeline(job, llm)

        assert job.status == JobStatus.DONE, job.error
        assert job.revision_rounds == 1
        # writer ran twice (initial + 1 revision), critic twice
        writer_starts = [e for e in job.events if e.agent == "writer" and e.kind == "start"]
        assert len(writer_starts) == 2

    @patch("app.agents.gather_sources", side_effect=_mock_gather)
    def test_revision_capped_at_max_rounds(self, _mock):
        job = Job(topic="topic X")
        llm = FakeLLM(critic_verdicts=["revise", "revise", "revise", "revise"])
        run_pipeline(job, llm)

        from app.config import settings
        assert job.status == JobStatus.DONE
        assert job.revision_rounds == settings.max_revision_rounds

    @patch("app.agents.gather_sources", return_value=[])
    def test_no_sources_fails_job(self, _mock):
        job = Job(topic="topic X")
        run_pipeline(job, FakeLLM())
        assert job.status == JobStatus.FAILED
        assert "no sources" in job.error.lower()

    @patch("app.agents.gather_sources", side_effect=_mock_gather)
    def test_keeps_best_draft_when_revision_regresses(self, _mock):
        """If a revision scores LOWER than the previous draft, the orchestrator
        must finalize the best draft, not the last one."""
        from tests.conftest import FakeLLM

        class RegressingLLM(FakeLLM):
            def __init__(self):
                super().__init__(critic_verdicts=["revise", "revise", "revise"])
                self._writer_calls = 0

            def chat(self, system, user, temperature=0.4, max_tokens=None, json_mode=False):
                if "Write the complete survey report" in user:
                    self._writer_calls += 1
                    if self._writer_calls == 1:
                        return "FIRST DRAFT MARKER\n\n## References\n[1] A — https://a.com"
                    return "REGRESSED SHORT DRAFT\n\n## References\n[1] A — https://a.com"
                return super().chat(system, user, temperature, max_tokens, json_mode)

        llm = RegressingLLM()
        # FakeLLM's critic scores every "revise" verdict 5.0, so revisions
        # never beat the first draft's 5.0 → first draft must be kept.
        job = Job(topic="topic X")
        run_pipeline(job, llm)
        assert job.status == JobStatus.DONE
        assert job.final_report.startswith("FIRST DRAFT MARKER")

    @patch("app.agents.gather_sources", side_effect=_mock_gather)
    def test_writer_appends_references_if_missing(self, _mock):
        """Writer output without a References section gets one appended."""
        from app.agents import _ensure_references
        sources = _mock_gather(["q"])
        draft = "# T\n\n## Intro\nText [1]."
        out = _ensure_references(draft, sources)
        assert "## References" in out
        assert "[1] Source One" in out
