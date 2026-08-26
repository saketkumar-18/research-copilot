"""Web search + page fetching.

Backend: DuckDuckGo via `ddgs` (no API key needed). Page text is extracted
with a stdlib HTML parser — no heavy deps, works on Render's free tier.
"""
from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Optional

import httpx

from .config import settings
from .models import Source

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

_SKIP_TAGS = {"script", "style", "noscript", "svg", "head", "nav", "footer", "iframe", "form"}
_BLOCK_TAGS = {"p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6",
               "section", "article", "table", "tr", "blockquote", "pre"}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data.strip():
            self.parts.append(data)


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(html)
    except Exception:
        pass
    text = "".join(parser.parts)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def web_search(query: str, max_results: Optional[int] = None) -> list[dict]:
    """Return [{title, href, body}] from DuckDuckGo."""
    n = max_results or settings.search_results_per_query
    try:
        from ddgs import DDGS
        results = DDGS().text(query, max_results=n)
    except Exception:
        return []
    out = []
    for r in results or []:
        href = r.get("href") or r.get("url") or ""
        if href:
            out.append({"title": r.get("title", ""), "href": href, "body": r.get("body", "")})
    return out


def fetch_page(url: str) -> tuple[str, str]:
    """Fetch a URL and extract readable text. Returns (text, error)."""
    try:
        with httpx.Client(
            timeout=settings.fetch_timeout_s,
            follow_redirects=True,
            headers={"User-Agent": _UA},
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()
            ctype = resp.headers.get("content-type", "")
            if "html" not in ctype and "text" not in ctype:
                return "", f"unsupported content-type: {ctype[:40]}"
            text = html_to_text(resp.text)
            if len(text) < 200:
                return "", "page too short after extraction (likely JS-only or blocked)"
            return text[: settings.fetch_max_chars], ""
    except Exception as e:
        return "", f"{type(e).__name__}: {str(e)[:120]}"


def gather_sources(queries: list[str], max_pages: Optional[int] = None) -> list[Source]:
    """Search all queries, dedupe URLs, fetch top pages for content."""
    budget = max_pages or settings.max_fetch_pages
    seen: set[str] = set()
    sources: list[Source] = []
    sid = 1
    for q in queries:
        for r in web_search(q):
            url = r["href"].split("#")[0].rstrip("/")
            if url in seen:
                continue
            seen.add(url)
            sources.append(
                Source(id=sid, url=url, title=r["title"], snippet=r["body"], query=q)
            )
            sid += 1
    # fetch content for the top `budget` sources (search rank order)
    for src in sources[:budget]:
        text, err = fetch_page(src.url)
        src.content = text
        src.fetched = bool(text)
        src.fetch_error = err
    return sources
