"""Command-line runner: research a topic without the web UI.

Usage:
    python -m cli "Impact of graph neural networks on drug discovery"
"""
from __future__ import annotations

import sys
import time

from app.config import settings
from app.llm import LLMClient
from app.models import Job
from app.orchestrator import run_pipeline


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    topic = " ".join(sys.argv[1:]).strip()
    settings.validate_for_llm()

    job = Job(topic=topic)
    print(f"▶ Researching: {topic}\n")

    # stream timeline events while the pipeline runs in the main thread:
    # run_pipeline is synchronous, so we hook emits via a wrapper instead.
    last = 0

    def show() -> None:
        nonlocal last
        for e in job.events[last:]:
            print(f"  [{e.agent:13}] {e.text or e.kind}")
        last = len(job.events)

    import threading
    t = threading.Thread(target=lambda: run_pipeline(job, LLMClient()))
    t.start()
    while t.is_alive():
        show()
        time.sleep(1)
    show()

    print(f"\nStatus: {job.status.value}")
    if job.error:
        print(f"Error: {job.error}")
        return 1
    if job.evaluation:
        ev = job.evaluation
        print(f"Quality: {ev.composite_score}/100 (grade {ev.grade})")
    print("\n" + "=" * 70 + "\n")
    print(job.final_report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
