"""Router (Manager LLM): understands the request, picks the domain, rewrites it into search queries."""
from __future__ import annotations

import logging
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.graph.state import MentorState
from src.llm.managers import get_structured_manager
from src.utils.vectorstore import get_knowledge_base

log = logging.getLogger(__name__)


class RoutingDecision(BaseModel):
    domain: str = Field(description="Exactly one available domain name, or 'all'.")
    intent: Literal["code", "theory", "mixed"]
    scope: Literal["snippet", "full"] = Field(
        description="'snippet' if the student asked for a specific piece — particular method(s), "
        "a fix, one class, one feature — rather than the whole program (e.g. 'write the methods "
        "for calculating cost', 'fix the save button'). 'full' if they asked for a complete "
        "program/class/feature built from scratch with no narrower scope stated."
    )
    retrieval_queries: list[str] = Field(
        description="1 to 3 short search queries using terms likely to appear in lecture material."
    )


_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are the Router of a university teaching-assistant system.\n"
     "Given a student's request, decide:\n"
     "1. domain: which knowledge domain holds the relevant course material. Choose EXACTLY one "
     "from: {domains}. If unsure, or the request spans several domains, answer 'all'.\n"
     "2. intent: 'code' (write or fix code), 'theory' (explain concepts), or 'mixed'.\n"
     "3. scope: 'snippet' if the request names a specific piece (particular method(s), a bug fix, "
     "one class) rather than 'build the whole thing'; 'full' otherwise. Read the request literally "
     "— 'write the methods for X' is snippet scope, not an invitation to build the whole app.\n"
     "4. retrieval_queries: 1 to 3 short, self-contained search queries (keywords or phrases as "
     "they would appear in lecture slides). Decompose the request; do not just copy it."),
    ("human", "{query}"),
])


def router_node(state: MentorState) -> dict:
    query = state["query"]
    domains = get_knowledge_base().available_domains()
    forced_intent = state.get("intent")  # set only if CLI passed --mode theory/code/mixed
    forced_scope = state.get("scope")    # set only if CLI passed --scope snippet/full

    try:
        decision: RoutingDecision = (_PROMPT | get_structured_manager(RoutingDecision)).invoke(
            {"domains": ", ".join(domains) or "none", "query": query}
        )
        # Always keep the original query too: cheap insurance for recall.
        extra = [q.strip() for q in decision.retrieval_queries if q.strip()][:3]
        queries = [query] + extra
        domain, intent, scope = decision.domain.strip().lower(), decision.intent, decision.scope
    except Exception as exc:  # noqa: BLE001 - routing failure must degrade gracefully, not crash
        log.warning("Router failed (%s: %s). Falling back to plain search.", type(exc).__name__, exc)
        queries, domain, intent, scope = [query], "all", forced_intent or "mixed", forced_scope or "full"

    if state.get("domain"):            # CLI --domain override wins
        domain = state["domain"].lower()
    if forced_intent:                  # CLI --mode override wins over the Router's own guess
        intent = forced_intent
    if forced_scope:                   # CLI --scope override wins over the Router's own guess
        scope = forced_scope
    final_domain = domain if domain in domains else None   # unknown/'all' => no filter
    return {"domain": final_domain, "intent": intent, "scope": scope, "retrieval_queries": queries}
