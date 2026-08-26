"""The four research agents: planner, searcher, writer, critic.

Each agent is a small class with a single responsibility and emits timeline
events onto the Job so the UI can show the multi-agent workflow live.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

from .config import settings
from .llm import LLMClient, LLMError
from .models import AgentName, Critique, Job, Plan, Source
from .search import gather_sources

EmitFn = Callable[[str, str, str], None]  # (agent, kind, text)


# --------------------------------------------------------------------- planner
PLANNER_SYSTEM = """You are the PLANNER agent in a multi-agent research team.
Given a research topic, produce a focused research plan for a survey report.
Be concrete and specific — generic plans produce generic reports."""


def run_planner(job: Job, llm: LLMClient) -> Plan:
    job.emit(AgentName.PLANNER, "start", f"Planning research for: {job.topic}")
    data = llm.chat_json(
        PLANNER_SYSTEM,
        f"Topic: {job.topic}\n\n"
        "Create a research plan as JSON with keys:\n"
        '- "title": report title (string)\n'
        '- "summary": 2-3 sentence scope statement (string)\n'
        '- "sections": 4-6 section headings for the survey (list of strings)\n'
        '- "search_queries": 4-6 specific web search queries to gather sources (list of strings)\n'
        '- "key_questions": 3-5 questions the report must answer (list of strings)',
    )
    plan = Plan(
        title=str(data.get("title") or job.topic),
        summary=str(data.get("summary") or ""),
        sections=[str(s) for s in data.get("sections", [])][:8],
        search_queries=[str(s) for s in data.get("search_queries", [])][:8],
        key_questions=[str(s) for s in data.get("key_questions", [])][:6],
    )
    if not plan.sections:
        raise LLMError("Planner returned no sections")
    if not plan.search_queries:
        plan.search_queries = [job.topic]
    job.emit(
        AgentName.PLANNER, "done",
        f"Plan ready: {len(plan.sections)} sections, {len(plan.search_queries)} queries",
        plan=plan.model_dump(),
    )
    return plan


# -------------------------------------------------------------------- searcher
def run_searcher(job: Job, plan: Plan) -> list[Source]:
    job.emit(
        AgentName.SEARCHER, "start",
        f"Searching {len(plan.search_queries)} queries",
        queries=plan.search_queries,
    )
    sources = gather_sources(plan.search_queries)
    fetched = sum(1 for s in sources if s.fetched)
    job.emit(
        AgentName.SEARCHER, "done",
        f"Collected {len(sources)} sources ({fetched} with full-text content)",
        count=len(sources), fetched=fetched,
    )
    return sources


# ---------------------------------------------------------------------- writer
WRITER_SYSTEM = """You are the WRITER agent in a multi-agent research team.
You write rigorous, well-organized survey reports from provided sources.

CITATION RULES (critical):
- Every factual claim must carry an inline citation like [1], [2], [5].
- Citation numbers refer to the numbered source list you are given.
- NEVER invent sources, URLs, or citation numbers not in the list.
- End the report with a "## References" section listing each cited source as:
  [n] Title — URL

STYLE: neutral academic tone, markdown headings (##), concise paragraphs,
no filler, no first person. Use tables or lists where they add clarity."""


def _select_writer_sources(sources: list[Source], max_sources: int = 12) -> list[Source]:
    """Pick the most valuable sources for the writer: full-text first, then
    top-ranked snippets. Keeps the writer prompt small enough for the model
    to process within the timeout."""
    fetched = [s for s in sources if s.fetched][:8]
    fillers = [s for s in sources if not s.fetched][: max_sources - len(fetched)]
    return fetched + fillers


def _sources_block(sources: list[Source]) -> str:
    blocks = []
    for s in sources:
        body = (s.content[:2500] if s.fetched else s.snippet) or s.snippet
        blocks.append(f"[{s.id}] {s.title}\nURL: {s.url}\nCONTENT:\n{body}")
    return "\n\n---\n\n".join(blocks)


def run_writer(
    job: Job,
    llm: LLMClient,
    plan: Plan,
    sources: list[Source],
    critique: Optional[Critique] = None,
    previous_draft: str = "",
) -> str:
    round_label = "REVISED draft" if critique else "first draft"
    job.emit(AgentName.WRITER, "start", f"Writing {round_label} of the survey report")
    writer_sources = _select_writer_sources(sources)
    job.writer_source_ids = [s.id for s in writer_sources]

    revision_instructions = ""
    if critique:
        revision_instructions = (
            "\n\nThis is a REVISION. A critic reviewed your previous draft and found issues.\n"
            f"Issues to fix:\n- " + "\n- ".join(critique.issues or ["none listed"]) +
            ("\nMissing points to add:\n- " + "\n- ".join(critique.missing_points)
             if critique.missing_points else "") +
            ("\nCitation problems to fix:\n- " + "\n- ".join(critique.citation_problems)
             if critique.citation_problems else "") +
            f"\n\nPrevious draft (for reference — produce the COMPLETE revised report, not a partial edit):\n{previous_draft[:4000]}\n"
        )

    user = (
        f"Report title: {plan.title}\n"
        f"Scope: {plan.summary}\n"
        f"Sections to cover: {plan.sections}\n"
        f"Key questions to answer: {plan.key_questions}\n"
        f"Target length: ~{settings.max_report_words_target} words.\n\n"
        f"AVAILABLE SOURCES (cite as [id]):\n\n{_sources_block(writer_sources)}"
        f"{revision_instructions}\n\n"
        "Write the complete survey report in markdown now."
    )
    draft = llm.chat(WRITER_SYSTEM, user, temperature=0.5)
    draft = _ensure_references(draft, sources)
    job.emit(
        AgentName.WRITER, "done",
        f"{round_label.capitalize()} written ({len(draft.split())} words)",
        words=len(draft.split()),
    )
    return draft


def _ensure_references(draft: str, sources: list[Source]) -> str:
    """Guarantee the report ends with a References section listing sources."""
    if re.search(r"^##\s*references", draft, re.IGNORECASE | re.MULTILINE):
        return draft
    lines = ["", "## References", ""]
    for s in sources:
        lines.append(f"[{s.id}] {s.title} — {s.url}")
    return draft.rstrip() + "\n" + "\n".join(lines)


# ---------------------------------------------------------------------- critic
CRITIC_SYSTEM = """You are the CRITIC agent in a multi-agent research team.
You review survey drafts harshly but fairly. Your job is to catch:
- claims with no inline citation or citing a source id that doesn't exist
- sections that are thin, vague, or off-topic
- missing coverage of the plan's key questions
- structure problems (no intro, no conclusion, wall-of-text)
Respond ONLY with the requested JSON."""


def run_critic(
    job: Job,
    llm: LLMClient,
    plan: Plan,
    sources: list[Source],
    draft: str,
) -> Critique:
    job.emit(AgentName.CRITIC, "start", "Reviewing draft for quality and citation integrity")
    valid_ids = [s.id for s in sources]
    data = llm.chat_json(
        CRITIC_SYSTEM,
        f"PLAN:\nTitle: {plan.title}\nSections: {plan.sections}\n"
        f"Key questions: {plan.key_questions}\n\n"
        f"VALID SOURCE IDS: {valid_ids}\n\n"
        f"DRAFT:\n{draft[:9000]}\n\n"
        "Evaluate as JSON with keys:\n"
        '- "verdict": "approve" or "revise"\n'
        '- "score": 0-10 float\n'
        '- "strengths": list of strings\n'
        '- "issues": list of concrete problems (empty list if none)\n'
        '- "missing_points": list of strings (plan points not covered)\n'
        '- "citation_problems": list of strings (uncited claims / bad ids)\n'
        'Approve only if score >= 7 and there are no citation problems.',
    )
    critique = Critique(
        verdict=str(data.get("verdict", "revise")).lower().strip(),
        score=float(data.get("score", 0) or 0),
        strengths=[str(x) for x in data.get("strengths", [])],
        issues=[str(x) for x in data.get("issues", [])],
        missing_points=[str(x) for x in data.get("missing_points", [])],
        citation_problems=[str(x) for x in data.get("citation_problems", [])],
    )
    if critique.verdict not in ("approve", "revise"):
        critique.verdict = "revise"
    job.emit(
        AgentName.CRITIC, "done",
        f"Verdict: {critique.verdict.upper()} (score {critique.score:.1f}/10, "
        f"{len(critique.issues)} issues)",
        critique=critique.model_dump(),
    )
    return critique
