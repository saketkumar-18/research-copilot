"""Output-quality evaluation engine — the capstone's evaluation angle.

Two layers:
1. Heuristic metrics (deterministic, cheap, always run):
   citation coverage, citation validity, structure, length, source utilization.
2. LLM judge (rubric-scored, runs when an LLM is available):
   accuracy, coverage, coherence, citation quality, style.

Composite score = weighted blend → letter grade.
"""
from __future__ import annotations

import re
from typing import Optional

from .config import settings
from .llm import LLMClient, LLMError
from .models import Evaluation, Job, Plan, Source

CITE_RE = re.compile(r"\[(\d{1,3})\]")
HEADING_RE = re.compile(r"^#{1,3}\s+\S", re.MULTILINE)


# ------------------------------------------------------------------ heuristics
def heuristic_metrics(report: str, sources: list[Source], plan: Optional[Plan],
                      job: Optional["Job"] = None) -> dict:
    refs_match = re.search(r"^##\s*references", report, re.IGNORECASE | re.MULTILINE)
    body = report[: refs_match.start()] if refs_match else report
    refs_section = report[refs_match.start():] if refs_match else ""

    words = len(body.split())
    headings = HEADING_RE.findall(body)
    inline_cites = CITE_RE.findall(body)
    cited_ids = {int(c) for c in inline_cites}
    valid_ids = {s.id for s in sources}

    # citation validity: every inline cite must map to a real source
    bad_cites = sorted(cited_ids - valid_ids)
    cite_density = len(inline_cites) / max(words, 1) * 100  # cites per 100 words

    # sentences lacking any citation (rough proxy for unsupported claims)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", body) if len(s.split()) > 8]
    uncited = [s for s in sentences if not CITE_RE.search(s)]
    citation_coverage = 1 - (len(uncited) / max(len(sentences), 1))

    # source utilization: how many of the sources GIVEN to the writer got
    # cited (penalizing for the long tail of un-fetched search hits would be
    # unfair — the writer never saw them)
    available_ids = set(job.writer_source_ids) if (job and job.writer_source_ids) else valid_ids
    utilization = len(cited_ids & available_ids) / max(len(available_ids), 1)

    # structure: has intro-ish heading, >=3 sections, references section
    has_refs = bool(refs_match)
    structure_score = min(1.0, len(headings) / 5) * (1.0 if has_refs else 0.4)

    # length adequacy vs target
    target = settings.max_report_words_target
    length_score = max(0.0, 1 - abs(words - target) / target) if target else 1.0

    # plan coverage: fraction of planned section titles that appear (fuzzy)
    plan_coverage = None
    if plan and plan.sections:
        hits = 0
        body_l = body.lower()
        for sec in plan.sections:
            key = re.sub(r"[^a-z0-9 ]", "", sec.lower()).strip()
            words_key = [w for w in key.split() if len(w) > 3]
            if words_key and sum(1 for w in words_key if w in body_l) >= max(1, len(words_key) // 2):
                hits += 1
        plan_coverage = hits / len(plan.sections)

    def scale(x: float) -> float:
        return round(100 * max(0.0, min(1.0, x)), 1)

    scores = {
        "citation_coverage": scale(citation_coverage),
        "citation_validity": scale(0.0 if bad_cites else 1.0),
        "source_utilization": scale(utilization),
        "structure": scale(structure_score),
        "length_adequacy": scale(length_score),
    }
    if plan_coverage is not None:
        scores["plan_coverage"] = scale(plan_coverage)

    return {
        "scores": scores,
        "stats": {
            "words": words,
            "headings": len(headings),
            "inline_citations": len(inline_cites),
            "unique_sources_cited": len(cited_ids & valid_ids),
            "sources_gathered": len(valid_ids),
            "invalid_citation_ids": bad_cites,
            "cite_density_per_100_words": round(cite_density, 2),
            "has_references_section": has_refs,
        },
        "mean": round(sum(scores.values()) / len(scores), 1) if scores else 0.0,
    }


# ------------------------------------------------------------------ LLM judge
JUDGE_SYSTEM = """You are an impartial EVALUATION JUDGE for AI-generated survey reports.
Score strictly against the rubric. Do not reward length alone; penalize
uncited claims, hallucinated references, and vague filler. Respond ONLY with JSON."""

JUDGE_RUBRIC = """Score each dimension 0-10 (integers):
- "accuracy": claims consistent with the provided sources, no fabrication
- "coverage": addresses the plan's sections and key questions
- "coherence": logical flow, clear structure, readable
- "citation_quality": claims are cited, citations match sources, references complete
- "style": academic tone, concise, no filler
Also return:
- "summary": 2-3 sentence overall assessment
- "top_improvements": up to 3 concrete improvements (list of strings)"""


def llm_judge(
    llm: LLMClient,
    report: str,
    plan: Optional[Plan],
    sources: list[Source],
) -> dict:
    src_lines = "\n".join(f"[{s.id}] {s.title} — {s.url}" for s in sources[:30])
    plan_block = (
        f"Sections: {plan.sections}\nKey questions: {plan.key_questions}\n"
        if plan else "No plan available.\n"
    )
    data = llm.chat_json(
        JUDGE_SYSTEM,
        f"PLAN:\n{plan_block}\n"
        f"SOURCE LIST (the only valid references):\n{src_lines}\n\n"
        f"REPORT TO JUDGE:\n{report[:9000]}\n\n{JUDGE_RUBRIC}\n\n"
        'Respond as JSON: {"accuracy": int, "coverage": int, "coherence": int, '
        '"citation_quality": int, "style": int, "summary": str, "top_improvements": [str]}',
        max_tokens=1500,
    )
    dims = {}
    for k in ("accuracy", "coverage", "coherence", "citation_quality", "style"):
        try:
            dims[k] = max(0, min(10, int(float(data.get(k, 0)))))
        except (TypeError, ValueError):
            dims[k] = 0
    return {
        "dimensions": dims,
        "mean_10": round(sum(dims.values()) / len(dims), 2),
        "summary": str(data.get("summary", "")),
        "top_improvements": [str(x) for x in data.get("top_improvements", [])][:3],
    }


# ------------------------------------------------------------------ composite
def grade_for(score: float) -> str:
    if score >= 85:
        return "A"
    if score >= 70:
        return "B"
    if score >= 55:
        return "C"
    if score >= 40:
        return "D"
    return "F"


def evaluate_report(
    job: Job,
    report: str,
    plan: Optional[Plan],
    sources: list[Source],
    llm: Optional[LLMClient] = None,
) -> Evaluation:
    from .models import AgentName

    job.emit(AgentName.EVALUATOR, "start", "Evaluating report quality (heuristics + judge)")
    heur = heuristic_metrics(report, sources, plan, job)

    judge: dict = {}
    judge_err = ""
    if llm is not None:
        try:
            judge = llm_judge(llm, report, plan, sources)
        except LLMError as e:
            judge_err = str(e)[:200]

    if judge:
        composite = (1 - settings.judge_weight) * heur["mean"] + \
            settings.judge_weight * judge["mean_10"] * 10
    else:
        composite = heur["mean"]
    composite = round(composite, 1)

    notes = []
    if judge_err:
        notes.append(f"LLM judge unavailable ({judge_err}); composite from heuristics only.")
    stats = heur["stats"]
    if stats["invalid_citation_ids"]:
        notes.append(f"Invalid citation ids detected: {stats['invalid_citation_ids']}")
    if not stats["has_references_section"]:
        notes.append("Missing References section.")
    if judge.get("summary"):
        notes.append(f"Judge: {judge['summary']}")

    evaluation = Evaluation(
        heuristic=heur,
        judge=judge,
        composite_score=composite,
        grade=grade_for(composite),
        notes=notes,
    )
    job.emit(
        AgentName.EVALUATOR, "done",
        f"Quality score: {composite}/100 (grade {evaluation.grade})",
        evaluation=evaluation.model_dump(),
    )
    return evaluation
