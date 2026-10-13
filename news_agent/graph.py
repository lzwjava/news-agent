"""Graph assembly."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from . import nodes
from .render import render
from .state import AgentState


def build_graph(checkpointer=None):
    """Compile the news agent graph.

    Parallelism comes from two dynamic fan-outs built with ``Send``:
    one branch per feed, and one LLM editor per digest section.
    """
    builder = StateGraph(AgentState)

    builder.add_node("select_sources", nodes.select_sources)
    builder.add_node("fetch", nodes.fetch_node)
    builder.add_node("normalize", nodes.normalize)
    builder.add_node("rank", nodes.rank)
    builder.add_node("build_sections", nodes.build_sections)
    builder.add_node("summarize_section", nodes.summarize_section)
    builder.add_node("compile_digest", nodes.compile_digest)
    builder.add_node("render", render)

    builder.add_edge(START, "select_sources")
    builder.add_conditional_edges("select_sources", nodes.route_fetch, ["fetch"])
    builder.add_edge("fetch", "normalize")
    builder.add_edge("normalize", "rank")
    builder.add_edge("rank", "build_sections")
    builder.add_conditional_edges(
        "build_sections", nodes.route_digest, ["summarize_section", "compile_digest"]
    )
    builder.add_edge("summarize_section", "compile_digest")
    builder.add_edge("compile_digest", "render")
    builder.add_edge("render", END)

    return builder.compile(checkpointer=checkpointer or MemorySaver())


def graph_mermaid() -> str:
    return build_graph().get_graph().draw_mermaid()