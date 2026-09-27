"""
CLI entry point. Run from the project root:

    python main.py "Write a REST API in Spring Boot based on the slides"
    python main.py "Explain backpropagation" --domain ai_theory
    python main.py "Explain what I need to do" --mode theory
    python main.py "..." --k 10 --show-context
    python main.py "..." --save
    python main.py                                   # interactive mode
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
import time
from pathlib import Path

from src.agents.retriever import format_context
from src.graph.workflow import build_graph
from src.utils.vectorstore import get_knowledge_base

RUNS_DIR = Path(__file__).resolve().parent / "runs"


class Transcript:
    """Mirrors everything printed to the console into a buffer, so --save writes exactly what you saw."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.lines: list[str] = []

    def echo(self, text: str = "") -> None:
        print(text)
        if self.enabled:
            self.lines.append(text)

    def save(self, query: str) -> Path:
        RUNS_DIR.mkdir(exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = "".join(c if c.isalnum() else "_" for c in query[:40]).strip("_") or "query"
        path = RUNS_DIR / f"{stamp}_{slug}.md"
        path.write_text("\n".join(self.lines), encoding="utf-8")
        return path


def _report(t: Transcript, node: str, delta: dict, elapsed: float) -> None:
    """Live progress with per-node timing, so a slow run is easy to attribute to a specific agent/model."""
    if node == "router":
        t.echo(f"  [router]    ({elapsed:.1f}s) domain={delta['domain'] or 'all'} "
               f"intent={delta['intent']} scope={delta['scope']} "
               f"queries={delta['retrieval_queries'][1:] or delta['retrieval_queries']}")
    elif node == "retriever":
        srcs = {f"{d.metadata['filename']}" + (f" p.{d.metadata['page']}" if not d.metadata.get("is_code") else "")
                for d in delta["context_chunks"]}
        t.echo(f"  [retriever] ({elapsed:.1f}s) {len(delta['context_chunks'])} chunks from {len(srcs)} file(s)")
    elif node == "drafter":
        t.echo(f"  [drafter]   ({elapsed:.1f}s) draft #{delta['revision_count']} written")
    elif node == "critic":
        v = delta["verdict"]
        if v["approved"]:
            t.echo(f"  [critic]    ({elapsed:.1f}s) PASS")
        else:
            t.echo(f"  [critic]    ({elapsed:.1f}s) FAIL -> {len(v['issues'])} issue(s)")
            for issue in v["issues"]:
                t.echo(f"              - {issue}")


def run_query(graph, query: str, domain: str | None, mode: str, scope: str, k: int | None,
              show_context: bool, save: bool) -> None:
    t = Transcript(enabled=save)
    t0 = time.time()
    initial: dict = {"query": query, "revision_count": 0}
    if domain:
        initial["domain"] = domain
    if mode != "auto":
        initial["intent"] = mode
    if scope != "auto":
        initial["scope"] = scope
    if k:
        initial["k"] = k

    final: dict = dict(initial)
    last_t = time.time()
    for update in graph.stream(initial, stream_mode="updates"):
        for node, delta in update.items():
            if delta:
                final.update(delta)
                now = time.time()
                _report(t, node, delta, now - last_t)
                last_t = now

    if show_context:
        t.echo("\n--- RETRIEVED CONTEXT ---")
        t.echo(format_context(final.get("context_chunks", [])))

    t.echo("\n" + "=" * 70)
    t.echo(final["final_answer"])
    t.echo("=" * 70)
    if final.get("warning"):
        t.echo(f"\n[WARNING] {final['warning']}")
    sources = sorted({f"{d.metadata['filename']}" + (f" (p.{d.metadata['page']})" if not d.metadata.get("is_code") else "")
                      for d in final.get("context_chunks", [])})
    if sources:
        t.echo("\nSources: " + ", ".join(sources))
    t.echo(f"\nDone in {time.time() - t0:.1f}s")

    if save:
        path = t.save(query)
        print(f"\nTranscript saved: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Hierarchical Academic AI Mentor")
    parser.add_argument("query", nargs="?", help="Your question/assignment (omit for interactive mode)")
    parser.add_argument("--domain", help="Force a domain (folder name under docs/) instead of auto-routing")
    parser.add_argument("--mode", choices=["auto", "theory", "code", "mixed"], default="auto",
                         help="Force the response type instead of letting the Router decide")
    parser.add_argument("--scope", choices=["auto", "snippet", "full"], default="auto",
                         help="Force snippet (targeted piece) vs full (whole implementation) scope")
    parser.add_argument("--k", type=int, help="Override how many chunks are retrieved (default from .env)")
    parser.add_argument("--show-context", action="store_true", help="Print the exact chunks the Drafter saw")
    parser.add_argument("--save", action="store_true", help="Save the full run (trace + answer) under runs/")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

    kb = get_knowledge_base()
    if kb.count() == 0:
        print("The knowledge base is empty. Put PDFs/code in docs/ and run: python ingest.py")
        return 1
    print(f"Knowledge base: {kb.count()} chunks | domains: {', '.join(kb.available_domains())}\n")

    graph = build_graph()
    if args.query:
        run_query(graph, args.query, args.domain, args.mode, args.scope, args.k, args.show_context, args.save)
        return 0

    while True:  # interactive mode
        try:
            q = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"", "exit", "quit"}:
            break
        run_query(graph, q, args.domain, args.mode, args.scope, args.k, args.show_context, args.save)
    return 0


if __name__ == "__main__":
    sys.exit(main())
