"""
workflow.py — wires the agents into a LangGraph state machine.

    START -> router -> retriever -> drafter -> critic --approved--------------> finalize -> END
                                        ^          |
                                        +--rejected & revisions left

`revision_count` = drafts written so far. Total drafts allowed = 1 + MAX_REVISIONS.
"""
from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from src.agents.critic import critic_node
from src.agents.drafter import drafter_node
from src.agents.retriever import retriever_node
from src.agents.router import router_node
from src.config import settings
from src.graph.state import MentorState


def route_after_critic(state: MentorState) -> Literal["drafter", "finalize"]:
    if state["verdict"]["approved"]:
        return "finalize"
    if state["revision_count"] > settings.max_revisions:  # budget exhausted: stop looping
        return "finalize"
    return "drafter"


def finalize_node(state: MentorState) -> dict:
    """Never present an unvalidated draft as if it passed: attach an explicit warning."""
    warning = state.get("warning")
    verdict = state.get("verdict") or {}
    if not verdict.get("approved") and not warning:
        issues = "; ".join(verdict.get("issues", [])) or "unspecified"
        warning = (f"Not fully validated after {state['revision_count']} draft(s). "
                   f"Remaining issues flagged by the reviewer: {issues}")
    return {"final_answer": state["draft"], "warning": warning}


def build_graph():
    g = StateGraph(MentorState)
    g.add_node("router", router_node)
    g.add_node("retriever", retriever_node)
    g.add_node("drafter", drafter_node)
    g.add_node("critic", critic_node)
    g.add_node("finalize", finalize_node)

    g.add_edge(START, "router")
    g.add_edge("router", "retriever")
    g.add_edge("retriever", "drafter")
    g.add_edge("drafter", "critic")
    g.add_conditional_edges("critic", route_after_critic, {"drafter": "drafter", "finalize": "finalize"})
    g.add_edge("finalize", END)
    return g.compile()
