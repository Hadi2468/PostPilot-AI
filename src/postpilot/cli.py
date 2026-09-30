"""Command-line entry point: `postpilot --input data/sample_meeting.json`."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from postpilot.config import get_settings
from postpilot.graph import build_default_graph
from postpilot.observability import log_tracing_status
from postpilot.schemas import ReviewDecision
from postpilot.service import PostPilotService, RunResult

logger = logging.getLogger("postpilot")

DIVIDER = "=" * 60


def _print_run(run: RunResult) -> None:
    print(f"\n{DIVIDER}\nPOST  (status={run.status}, score={run.score:.2f})\n{DIVIDER}\n")
    print(run.post or "(no post)")
    print(f"\n{DIVIDER}\nSCORE HISTORY\n{DIVIDER}")
    for h in run.history:
        print(f"  iteration {h['iteration']}: {h['score']:.2f}")


def _read_multiline(prompt: str) -> str:
    print(f"{prompt} (finish with a line containing only '.')")
    lines = []
    for line in sys.stdin:
        if line.rstrip("\n") == ".":
            break
        lines.append(line.rstrip("\n"))
    return "\n".join(lines)


def _ask_decision() -> ReviewDecision:
    while True:
        prompt = "\n[a]pprove / [e]dit / [r]evise with feedback / [x] reject > "
        choice = input(prompt).strip().lower()
        try:
            if choice == "a":
                return ReviewDecision(action="approve")
            if choice == "e":
                return ReviewDecision(action="edit", post=_read_multiline("Paste the edited post"))
            if choice == "r":
                return ReviewDecision(action="revise", feedback=input("Feedback for the writer > "))
            if choice == "x":
                return ReviewDecision(action="reject")
        except ValueError as e:
            print(f"Invalid decision: {e}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Turn a meeting summary into a LinkedIn post.")
    parser.add_argument("--input", type=Path, required=True, help="Meeting summary JSON file")
    parser.add_argument("--no-review", action="store_true", help="Skip the human approval step")
    parser.add_argument("--output", type=Path, help="Write the final run result as JSON")
    parser.add_argument("--save-graph", type=Path, help="Save the graph diagram as PNG")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    settings = get_settings()  # loads .env first, so tracing env vars are visible
    log_tracing_status()
    if args.no_review:
        settings = settings.model_copy(update={"human_review": False})

    graph = build_default_graph(settings)
    if args.save_graph:
        try:
            args.save_graph.write_bytes(graph.get_graph().draw_mermaid_png())
            logger.info("Graph saved to %s", args.save_graph)
        except Exception:
            # draw_mermaid_png calls a remote rendering API; fall back to Mermaid text.
            mermaid = graph.get_graph().draw_mermaid()
            logger.warning("PNG render failed; Mermaid source:\n%s", mermaid)

    meeting_data = json.loads(args.input.read_text(encoding="utf-8"))
    service = PostPilotService(graph, settings)

    run = service.start(meeting_data)
    while run.status == "awaiting_review":
        _print_run(run)
        run = service.resume(run.thread_id, _ask_decision())

    _print_run(run)
    if args.output:
        args.output.write_text(run.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Result written to %s", args.output)


if __name__ == "__main__":
    main()
