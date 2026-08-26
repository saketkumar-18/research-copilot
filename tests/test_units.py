"""Unit tests: JSON extraction, HTML-to-text, heuristic metrics."""
from __future__ import annotations

import pytest

from app.llm import extract_json, LLMError
from app.search import html_to_text
from app.evaluation import heuristic_metrics, grade_for
from app.models import Plan


class TestExtractJson:
    def test_plain(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_fenced(self):
        assert extract_json('```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}

    def test_embedded_prose(self):
        out = extract_json('Here you go: {"verdict": "approve", "score": 8} hope that helps')
        assert out["verdict"] == "approve"

    def test_nested(self):
        out = extract_json('{"outer": {"inner": {"x": 1}}}')
        assert out["outer"]["inner"]["x"] == 1

    def test_no_json_raises(self):
        with pytest.raises(LLMError):
            extract_json("no json here at all")


class TestHtmlToText:
    def test_basic(self):
        html = "<html><head><title>t</title><style>x{}</style></head><body><h1>Hi</h1><p>Hello world.</p><script>var a=1;</script></body></html>"
        text = html_to_text(html)
        assert "Hello world." in text
        assert "var a=1" not in text
        assert "x{}" not in text

    def test_blocks_become_newlines(self):
        text = html_to_text("<p>one</p><p>two</p>")
        assert "one" in text and "two" in text
        assert text.index("one") < text.index("two")


class TestHeuristics:
    def _sources(self):
        from app.models import Source
        return [
            Source(id=1, url="https://a.com", title="A", fetched=True),
            Source(id=2, url="https://b.com", title="B", fetched=True),
        ]

    def test_good_report_scores_high(self):
        report = (
            "# Title\n\n## Introduction\nThis is a claim about the topic [1]. "
            "Another supported claim here [2].\n\n## Methods\nThe methods are "
            "well described in the literature [1].\n\n## Applications\nApplications "
            "are numerous and varied [2].\n\n## Conclusion\nWe conclude the topic "
            "matters [1].\n\n## References\n[1] A — https://a.com\n[2] B — https://b.com"
        )
        m = heuristic_metrics(report, self._sources(), None)
        assert m["scores"]["citation_validity"] == 100.0
        assert m["stats"]["has_references_section"] is True
        assert m["stats"]["unique_sources_cited"] == 2
        assert m["mean"] > 50

    def test_invalid_citation_detected(self):
        report = "## Intro\nA claim [9].\n\n## References\n"
        m = heuristic_metrics(report, self._sources(), None)
        assert 9 in m["stats"]["invalid_citation_ids"]
        assert m["scores"]["citation_validity"] == 0.0

    def test_missing_references_penalized(self):
        report = "## Intro\nA claim [1]."
        m = heuristic_metrics(report, self._sources(), None)
        assert m["stats"]["has_references_section"] is False
        assert m["scores"]["structure"] < 50

    def test_plan_coverage(self):
        plan = Plan(title="T", sections=["Introduction", "Deep Learning Methods"])
        report = "## Introduction\nIntro text [1].\n## Deep Learning Methods\nDL methods [1].\n## References\n[1] A — https://a.com"
        m = heuristic_metrics(report, self._sources(), plan)
        assert m["scores"]["plan_coverage"] == 100.0


class TestGrades:
    @pytest.mark.parametrize("score,grade", [
        (90, "A"), (85, "A"), (70, "B"), (69.9, "C"), (55, "C"), (40, "D"), (39, "F"), (0, "F"),
    ])
    def test_grade_boundaries(self, score, grade):
        assert grade_for(score) == grade
