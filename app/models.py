"""Core data models shared across agents, orchestrator, API and evaluation."""
from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class AgentName(str, Enum):
    PLANNER = "planner"
    SEARCHER = "searcher"
    WRITER = "writer"
    CRITIC = "critic"
    EVALUATOR = "evaluator"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class Source(BaseModel):
    """A web source gathered by the searcher."""
    id: int
    url: str
    title: str
    snippet: str = ""
    content: str = ""          # fetched page text (truncated)
    fetched: bool = False
    fetch_error: str = ""
    query: str = ""            # the search query that surfaced it


class Plan(BaseModel):
    """Structured research plan produced by the planner."""
    title: str
    summary: str = ""
    sections: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)
    key_questions: list[str] = Field(default_factory=list)


class Critique(BaseModel):
    """Critic's verdict on a draft."""
    verdict: str = "revise"            # "approve" | "revise"
    score: float = 0.0                 # 0-10
    strengths: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    missing_points: list[str] = Field(default_factory=list)
    citation_problems: list[str] = Field(default_factory=list)


class Evaluation(BaseModel):
    """Quality evaluation of the final report (capstone angle)."""
    heuristic: dict[str, Any] = Field(default_factory=dict)
    judge: dict[str, Any] = Field(default_factory=dict)
    composite_score: float = 0.0       # 0-100
    grade: str = ""                    # A/B/C/D/F
    notes: list[str] = Field(default_factory=list)


class AgentEvent(BaseModel):
    """One entry in the live agent timeline."""
    ts: float = Field(default_factory=time.time)
    agent: str
    kind: str                          # start | message | artifact | done | error
    text: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class Job(BaseModel):
    """A research job: topic in, cited survey report out."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    topic: str
    status: JobStatus = JobStatus.QUEUED
    created_at: float = Field(default_factory=time.time)
    finished_at: Optional[float] = None
    error: str = ""

    plan: Optional[Plan] = None
    sources: list[Source] = Field(default_factory=list)
    draft: str = ""
    final_report: str = ""
    writer_source_ids: list[int] = Field(default_factory=list)
    critique: Optional[Critique] = None
    revision_rounds: int = 0
    evaluation: Optional[Evaluation] = None

    events: list[AgentEvent] = Field(default_factory=list)
    timings: dict[str, float] = Field(default_factory=dict)

    def emit(self, agent: str, kind: str, text: str = "", **data: Any) -> None:
        self.events.append(AgentEvent(agent=agent, kind=kind, text=text, data=data))

    def public_dict(self) -> dict[str, Any]:
        d = self.model_dump()
        # keep payloads light for polling clients
        for s in d.get("sources", []):
            s["content"] = s["content"][:600] + ("…" if len(s["content"]) > 600 else "")
        return d
