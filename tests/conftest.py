"""Shared test fixtures: a FakeLLM that scripts the whole pipeline."""
from __future__ import annotations

import json
import re

import pytest

from app.llm import LLMClient


class FakeLLM(LLMClient):
    """Deterministic LLM stand-in. Routes by keywords in the user prompt."""

    def __init__(self, critic_verdicts=None):
        super().__init__(base_url="http://fake", api_key="***", model="fake-model")
        self.calls: list[str] = []
        self.critic_verdicts = list(critic_verdicts or ["approve"])
        self._critic_calls = 0

    def chat(self, system, user, temperature=0.4, max_tokens=None, json_mode=False):
        self.calls.append(user[:80])
        if "research plan as JSON" in user:
            return json.dumps({
                "title": "Test Survey: Topic X",
                "summary": "A survey of topic X.",
                "sections": ["Introduction", "Methods", "Applications", "Conclusion"],
                "search_queries": ["topic X survey", "topic X methods"],
                "key_questions": ["What is topic X?", "How is it used?"],
            })
        if "Evaluate as JSON" in user:
            self._critic_calls += 1
            verdict = self.critic_verdicts[min(self._critic_calls - 1, len(self.critic_verdicts) - 1)]
            return json.dumps({
                "verdict": verdict,
                "score": 8.0 if verdict == "approve" else 5.0,
                "strengths": ["clear structure"],
                "issues": [] if verdict == "approve" else ["section Methods is thin"],
                "missing_points": [] if verdict == "approve" else ["applications"],
                "citation_problems": [],
            })
        if "Score each dimension" in user:
            return json.dumps({
                "accuracy": 8, "coverage": 8, "coherence": 9,
                "citation_quality": 7, "style": 8,
                "summary": "Solid survey with minor citation gaps.",
                "top_improvements": ["cite more sources in Methods"],
            })
        # writer
        return (
            "# Test Survey: Topic X\n\n## Introduction\nTopic X is important [1].\n\n"
            "## Methods\nMethods for topic X are described in [2].\n\n"
            "## Applications\nTopic X is used widely [1].\n\n"
            "## Conclusion\nTopic X matters [2].\n\n"
            "## References\n[1] Source One — https://example.com/1\n"
            "[2] Source Two — https://example.com/2"
        )

    def chat_json(self, system, user, temperature=0.2, max_tokens=None):
        from app.llm import extract_json
        return extract_json(self.chat(system, user, temperature, max_tokens))


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def fake_sources():
    from app.models import Source
    return [
        Source(id=1, url="https://example.com/1", title="Source One",
               snippet="about topic X", content="Full text about topic X methods.", fetched=True),
        Source(id=2, url="https://example.com/2", title="Source Two",
               snippet="more topic X", content="More full text about applications.", fetched=True),
    ]


@pytest.fixture
def fake_plan():
    from app.models import Plan
    return Plan(
        title="Test Survey: Topic X",
        summary="A survey of topic X.",
        sections=["Introduction", "Methods", "Applications", "Conclusion"],
        search_queries=["topic X survey"],
        key_questions=["What is topic X?", "How is it used?"],
    )
