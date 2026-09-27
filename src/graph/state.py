"""Shared state passed between LangGraph nodes. Each node returns a *partial* dict that LangGraph merges in."""
from __future__ import annotations

from typing import Optional, TypedDict

from langchain_core.documents import Document


class MentorState(TypedDict, total=False):
    query: str                      # the user's assignment / question
    domain: Optional[str]           # chosen knowledge domain (None = search everything); CLI --domain overrides Router
    intent: str                     # "code" | "theory" | "mixed"; CLI --mode overrides Router if not "auto"
    scope: str                      # "snippet" | "full"; CLI --scope overrides Router if not "auto"
    k: int                          # CLI --k override for retrieval breadth (falls back to settings.retrieval_k)
    retrieval_queries: list[str]    # search queries produced by the Router
    context_chunks: list[Document]  # what the Retriever found
    draft: str                      # latest Drafter output
    verdict: dict                   # Critic output: {approved, issues, feedback}
    revision_count: int             # number of drafts written so far
    final_answer: str
    warning: Optional[str]          # set when the answer is not fully validated
