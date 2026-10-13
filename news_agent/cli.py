"""Command line interface.

    news-agent run                 # fetch + rank + digest + render
    news-agent sources             # list the configured sources
    news-agent graph               # print the LangGraph graph (mermaid)
"""

from __future__ import annotations

import argparse
import sys
from uuid import uuid4

from .config import default_settings, load_sources
from .graph import build_graph, graph_mermaid
from .llm import build_llm


def _csv(value: str | None) -> list[str]:
    return [part.strip() for part in (value or "").split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="news-agent",
        description="LangGraph AI news agent: YC / OpenAI / Anthropic / AI press -> ranked HN-style digest.",
    )
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="run the agent (default)")
    run.add_argument("--hours", type=int, default=48, help="time window in hours (default 48)")
    run.add_argument("--top", type=int, default=40, help="max articles kept after ranking (default 40)")
    run.add_argument("--per-section", type=int, default=10, help="max articles per digest section (default 10)")
    run.add_argument("--per-source", type=int, default=25, help="max items fetched per source (default 25)")
    run.add_argument("--only", help="comma separated source ids to use")
    run.add_argument("--category", help="comma separated categories: labs,research,community,media")
    run.add_argument("--no-llm", action="store_true", help="skip the LLM, render a heuristic digest")
    run.add_argument("--provider", help="openrouter | deepseek | openai | custom")
    run.add_argument("--model", help="model name for the provider")
    run.add_argument("--sources", help="path to an alternative sources.yaml")
    run.add_argument("--out", help="output directory (default ./output)")
    run.add_argument("--thread-id", help="LangGraph thread id (enables resuming a previous run)")
    run.add_argument("--quiet", action="store_true", help="only print output paths")

    sub.add_parser("sources", help="list configured sources")

    graph = sub.add_parser("graph", help="print the agent graph")
    graph.add_argument("--format", choices=["mermaid"], default="mermaid")
    return parser


def cmd_run(args: argparse.Namespace) -> int:
    settings = default_settings(
        hours=args.hours,
        top=args.top,
        per_section=args.per_section,
        per_source=args.per_source,
        use_llm=not args.no_llm,
        provider=args.provider,
        model=args.model,
        sources_file=args.sources,
        out_dir=args.out,
        only=_csv(args.only),
        categories=_csv(args.category),
    )

    llm, reason = build_llm(settings)
    settings.use_llm = llm is not None
    if not args.quiet:
        status = f"LLM: {settings.provider}/{settings.model or 'default'}" if llm else f"LLM off ({reason})"
        print(f"news-agent · window={settings.hours}h · top={settings.top} · {status}")

    graph = build_graph()
    thread_id = args.thread_id or uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}
    if not args.quiet:
        print(f"thread-id: {thread_id} (rerun with --thread-id {thread_id} to resume)")
    try:
        final = graph.invoke({"settings": settings.to_dict()}, config=config)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    fetch_log = final.get("fetch_log") or []
    outputs = final.get("outputs") or {}
    if not args.quiet:
        print("\nfeeds:")
        for entry in sorted(fetch_log, key=lambda e: (not e.get("ok"), e["name"])):
            if entry.get("ok"):
                print(f"  ✓ {entry['name']:<28} {entry['count']:>3} items")
            else:
                print(f"  ✗ {entry['name']:<28} {entry.get('error', 'failed')}")
        digest = final.get("digest") or {}
        edited = sum(1 for s in digest.get("sections", []) if s.get("llm"))
        print(
            f"\n{len(fetch_log)} sources -> {sum(f.get('count', 0) for f in fetch_log)} raw -> "
            f"{len(final.get('articles') or [])} articles -> {len(digest.get('sections', []))} sections "
            f"({edited} LLM-edited)"
        )
        for error in final.get("errors") or []:
            print(f"  ! {error}")

    print("\noutputs:")
    for name in ("html", "markdown", "json", "rss"):
        if name in outputs:
            print(f"  {name:<9} {outputs[name]}")

    ok = sum(1 for f in fetch_log if f.get("ok"))
    return 0 if ok else 1


def cmd_sources(_args: argparse.Namespace) -> int:
    sources = load_sources()
    print(f"{len(sources)} sources in registry\n")
    print(f"{'id':<22}{'type':<6}{'category':<11}{'weight':<7}url")
    for src in sources:
        print(f"{src.id:<22}{src.type:<6}{src.category:<11}{src.weight:<7.1f}{src.url}")
    return 0


def cmd_graph(_args: argparse.Namespace) -> int:
    print(graph_mermaid())
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    known = {"run", "sources", "graph"}
    if not argv or (argv[0] not in known and argv[0] not in {"-h", "--help"}):
        argv.insert(0, "run")

    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {"run": cmd_run, "sources": cmd_sources, "graph": cmd_graph}
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())