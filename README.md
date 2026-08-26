# 🔬 Multi-Agent Research Copilot

A production-ready **multi-agent system** that takes a research topic and produces a
**cited survey report**, then grades its own output quality.

**Capstone angle:** agent orchestration (planner → searcher → writer → critic loop)
plus a two-layer **evaluation engine** for output quality.

## Architecture

```
                 ┌────────────┐
   topic ──────► │  PLANNER   │  research plan: sections, queries, key questions
                 └─────┬──────┘
                       ▼
                 ┌────────────┐
                 │  SEARCHER  │  DuckDuckGo search + full-text page fetch
                 └─────┬──────┘
                       ▼
                 ┌────────────┐   revise (≤ N rounds, keep BEST draft)
           ┌────►│   WRITER   │◄──────────────┐
           │     └─────┬──────┘               │
           │           ▼                      │
           │     ┌────────────┐        verdict=revise
           │     │   CRITIC   │───────────────┘
           │     └─────┬──────┘
           │           ▼ verdict=approve / rounds exhausted
           │     ┌────────────┐
           └─────│ EVALUATOR  │  heuristics + LLM judge → score 0-100 + grade
                 └────────────┘
```

### Agents
| Agent | Role | Output |
|---|---|---|
| **Planner** | Decomposes the topic into a research plan | title, sections, search queries, key questions (JSON) |
| **Searcher** | Gathers evidence | deduped web sources, top pages fetched to full text |
| **Writer** | Writes the survey with inline `[n]` citations | markdown report + References section |
| **Critic** | Reviews for citation integrity, coverage, structure | verdict (approve/revise) + issues list |
| **Evaluator** | Grades the final report | composite quality score + letter grade |

### Orchestration details that matter
- **Revision loop**: critic can reject a draft up to `MAX_REVISION_ROUNDS` times;
  the writer receives the critique (issues, missing points, citation problems)
  and rewrites.
- **Best-draft selection**: revisions can *regress* (truncated output, drift) —
  the orchestrator finalizes the draft with the highest critic score, not the last one.
- **Truncation guard**: if the LLM returns `finish_reason=length`, the client
  retries with a larger token budget instead of accepting a cut-off report.
- **Citation safety**: the writer may only cite ids from the source list it was
  given; a References section is guaranteed (appended if the model omits it).
- **Checkpointing**: job state is persisted to disk after every stage, so a
  restart never loses a finished report.

### Evaluation engine (the capstone's evaluation angle)
1. **Heuristic metrics** (deterministic, always run):
   - citation coverage (share of sentences carrying a citation)
   - citation validity (every `[n]` maps to a real source)
   - source utilization (sources given to writer vs. actually cited)
   - structure (headings, References section)
   - length adequacy vs. target
   - plan coverage (fuzzy match of planned sections in the report)
2. **LLM judge** (rubric-scored): accuracy, coverage, coherence, citation
   quality, style — each 0-10, plus summary + top improvements.
3. **Composite** = 50% heuristics + 50% judge → 0-100 score → letter grade A-F.

## Quickstart

```bash
uv venv .venv && source .venv/Scripts/activate   # or .venv/bin/activate
uv pip install -r requirements.txt
export LLM_API_KEY=***        # any OpenAI-compatible key
export LLM_BASE_URL=https://api.tokenrouter.com/v1   # or https://openrouter.ai/api/v1
export LLM_MODEL=qwen/qwen3.8-max-free               # or any chat model

python -m uvicorn app.main:app --port 10000
# open http://localhost:10000
```

### CLI (no browser)
```bash
python -m cli "Impact of graph neural networks on drug discovery"
```

## API

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | model config + status |
| POST | `/api/jobs` | start a research job `{topic}` → `{id}` (202) |
| GET | `/api/jobs` | recent jobs with scores |
| GET | `/api/jobs/{id}` | full job state (plan, sources, report, critique, evaluation, timeline) |
| GET | `/api/jobs/{id}/events?after=N` | poll the live agent timeline (cursor-based) |
| GET | `/api/jobs/{id}/report` | final markdown report + sources |
| GET | `/api/jobs/{id}/evaluation` | quality evaluation breakdown |

## Configuration (env vars)

| Var | Default | Meaning |
|---|---|---|
| `LLM_API_KEY` | — | **required** for the pipeline |
| `LLM_BASE_URL` | tokenrouter | any OpenAI-compatible endpoint |
| `LLM_MODEL` | `qwen/qwen3.8-max-free` | chat model id |
| `LLM_TIMEOUT_S` | 300 | per-call timeout (reasoning models are slow) |
| `LLM_MAX_TOKENS` | 8000 | generation budget |
| `MAX_REVISION_ROUNDS` | 2 | critic→writer revision cap |
| `REPORT_WORDS_TARGET` | 1400 | target report length |
| `SEARCH_RESULTS_PER_QUERY` | 5 | DDG results per query |
| `MAX_FETCH_PAGES` | 8 | pages fetched to full text |
| `JUDGE_WEIGHT` | 0.5 | judge share of composite score |
| `JOBS_DIR` | `data/jobs` | job persistence directory |

## Tests

```bash
python -m pytest tests/ -q     # 33 tests: units, mocked e2e pipeline, API
```

Covers: JSON extraction, HTML→text, heuristic metrics, grade boundaries,
full pipeline (approve / revise-then-approve / revision cap / regression
guard / no-sources failure), and every API endpoint.

## Deployment

Render (free tier) via `render.yaml` — set the `LLM_API_KEY` secret, then:

```bash
render up   # or create the web service from this repo
```

The app is a single Python service: FastAPI serves the API **and** the static
frontend. Jobs persist to disk under `JOBS_DIR`.

## Project layout

```
app/
  config.py        env-driven settings
  models.py        Job, Plan, Source, Critique, Evaluation, AgentEvent
  llm.py           OpenAI-compatible client: retry, truncation guard, JSON extraction
  search.py        DuckDuckGo search + HTML→text page fetching
  agents.py        planner / searcher / writer / critic
  evaluation.py    heuristic metrics + LLM judge + composite grading
  orchestrator.py  the agent loop with best-draft selection
  store.py         in-memory + disk job persistence
  main.py          FastAPI app
web/index.html     single-page UI: live timeline, report, evaluation dashboard
tests/             33 tests (units + mocked pipeline + API)
cli.py             command-line runner
```
