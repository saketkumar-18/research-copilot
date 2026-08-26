"""Orchestrator: runs the planner → searcher → writer → critic loop.

The critic can send the draft back to the writer for up to
`max_revision_rounds` revisions before the report is finalized and evaluated.
"""
from __future__ import annotations

import time
from typing import Optional

from .agents import run_critic, run_planner, run_searcher, run_writer
from .config import settings
from .evaluation import evaluate_report
from .llm import LLMClient, LLMError
from .models import AgentName, Job, JobStatus
from .store import save_job


def run_pipeline(job: Job, llm: Optional[LLMClient] = None) -> Job:
    """Execute the full multi-agent pipeline synchronously on `job`."""
    llm = llm or LLMClient()
    job.status = JobStatus.RUNNING
    job.emit("orchestrator", "start", f"Pipeline started for topic: {job.topic}")
    save_job(job)
    t0 = time.time()

    try:
        # 1. PLAN
        t = time.time()
        job.plan = run_planner(job, llm)
        job.timings["planner_s"] = round(time.time() - t, 1)
        save_job(job)

        # 2. SEARCH
        t = time.time()
        job.sources = run_searcher(job, job.plan)
        job.timings["searcher_s"] = round(time.time() - t, 1)
        save_job(job)
        if not job.sources:
            raise LLMError("Searcher found no sources — cannot write a cited report")

        # 3. WRITE → 4. CRITIC → revise loop.
        # Keep the BEST draft by critic score (revisions can regress, e.g.
        # truncated output), never blindly the last one.
        t = time.time()
        draft = run_writer(job, llm, job.plan, job.sources)
        job.draft = draft

        critique = run_critic(job, llm, job.plan, job.sources, draft)
        job.critique = critique
        best_draft, best_score = draft, critique.score
        rounds = 0
        while critique.verdict == "revise" and rounds < settings.max_revision_rounds:
            rounds += 1
            job.revision_rounds = rounds
            job.emit(
                "orchestrator", "message",
                f"Critic requested revision — round {rounds}/{settings.max_revision_rounds}",
            )
            draft = run_writer(job, llm, job.plan, job.sources,
                               critique=critique, previous_draft=best_draft)
            job.draft = draft
            critique = run_critic(job, llm, job.plan, job.sources, draft)
            job.critique = critique
            if critique.score > best_score:
                best_draft, best_score = draft, critique.score
                job.emit("orchestrator", "message",
                         f"Revision improved the draft (critic score {best_score:.1f}/10)")
            else:
                job.emit("orchestrator", "message",
                         f"Revision did not improve (score {critique.score:.1f} ≤ best "
                         f"{best_score:.1f}) — keeping previous best draft")
        job.timings["writer_critic_s"] = round(time.time() - t, 1)

        job.final_report = best_draft
        save_job(job)

        # 5. EVALUATE
        t = time.time()
        job.evaluation = evaluate_report(job, job.final_report, job.plan, job.sources, llm)
        job.timings["evaluator_s"] = round(time.time() - t, 1)

        job.timings["total_s"] = round(time.time() - t0, 1)
        job.status = JobStatus.DONE
        job.finished_at = time.time()
        job.emit(
            "orchestrator", "done",
            f"Pipeline complete in {job.timings['total_s']}s — "
            f"score {job.evaluation.composite_score}/100 (grade {job.evaluation.grade})",
        )
    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = f"{type(e).__name__}: {e}"
        job.finished_at = time.time()
        job.emit("orchestrator", "error", job.error)
    return job
