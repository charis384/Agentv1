"""Retriever: deliberately NOT an LLM. Semantic search is a vector-DB operation; an LLM here would only add latency."""
from __future__ import annotations

from langchain_core.documents import Document

from src.config import settings
from src.graph.state import MentorState
from src.utils.vectorstore import get_knowledge_base


def retriever_node(state: MentorState) -> dict:
    kb = get_knowledge_base()
    queries, domain = state["retrieval_queries"], state.get("domain")
    k = state.get("k") or settings.retrieval_k  # CLI --k override, else the configured default

    docs = kb.search(queries, domain, k)
    if not docs and domain:  # domain filter too strict / mis-routed -> widen to the whole DB
        docs = kb.search(queries, None, k)
    return {"context_chunks": docs}


def format_context(docs: list[Document]) -> str:
    """Numbered, source-tagged blocks so the Drafter can cite [n] and the Critic can verify."""
    if not docs:
        return "(no relevant material was found in the course documents)"
    blocks = []
    for i, d in enumerate(docs, 1):
        m = d.metadata
        if m.get("is_code"):
            # Fenced as real source, not prose: the Drafter must copy identifiers
            # verbatim from here rather than inventing plausible-looking ones.
            lang = m.get("filename", "").rsplit(".", 1)[-1] if "." in m.get("filename", "") else ""
            header = f"[{i}] EXISTING SOURCE FILE: {m.get('filename', '?')}"
            blocks.append(f"{header}\n```{lang}\n{d.page_content}\n```")
        else:
            blocks.append(f"[{i}] {m.get('filename', '?')} (page {m.get('page', '?')})\n{d.page_content}")
    return "\n\n".join(blocks)
