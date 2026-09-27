"""Critic (Manager LLM): validates the draft against the request and the context. Approves or forces a rewrite."""
from __future__ import annotations

import logging

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.agents.retriever import format_context
from src.graph.state import MentorState
from src.llm.managers import get_structured_manager

log = logging.getLogger(__name__)


class Verdict(BaseModel):
    approved: bool = Field(description="True only if the draft has no substantive problems.")
    issues: list[str] = Field(description="Concrete problems found. Empty if approved.")
    feedback: str = Field(description="Actionable instructions for the rewrite. Empty if approved.")


_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict but fair reviewer of a teaching assistant's answer.\n"
     "You may be given CONVERSATION HISTORY from earlier in this session. Use it ONLY to understand "
     "what a reference in the request means (e.g. confirm 'that' correctly refers to the B2 method "
     "discussed earlier) and whether the draft is a sensible continuation of that thread. Do NOT "
     "treat anything stated in a past turn as verified fact on its own — COURSE CONTEXT is still "
     "the only source of truth for grounding and correctness checks.\n"
     "REQUEST TYPE and SCOPE tell you what the student actually wants; judge completeness against "
     "THAT, not against what you'd personally include.\n"
     "- If REQUEST TYPE is 'theory', a full runnable implementation is OUT OF SCOPE — do not flag "
     "its absence, and do not reject an explanation for lacking code.\n"
     "- If SCOPE is 'snippet', the student asked for a SPECIFIC piece (named method(s)/class/fix). "
     "Judge the draft ONLY on whether that specific piece is correct and complete. Do NOT reject "
     "for missing surrounding integration, GUI wiring, persistence, or other classes/methods the "
     "student did not ask for — that is out of scope by design, not an omission.\n"
     "- If SCOPE is 'full' (and REQUEST TYPE is 'code' or 'mixed'), a complete, runnable "
     "implementation of the whole requested feature IS required.\n"
     "Check, in order:\n"
     "1. COMPLETENESS: does the draft satisfy every requirement in the student's request, at the "
     "depth REQUEST TYPE and SCOPE call for (no more, no less)?\n"
     "2. GROUNDING: are technical claims supported by the course context, or otherwise "
     "well-established and correct? Flag invented APIs/annotations/facts and any contradiction "
     "with the context.\n"
     "3. CORRECTNESS: syntax errors, logic bugs, missing imports/dependencies in code; "
     "factual errors in explanations.\n"
     "4. CITATIONS: [n] references must point to context blocks that actually support the claim.\n"
     "5. CONTINUITY (only if CONVERSATION HISTORY is present): if the request builds on a prior "
     "turn, does the draft correctly build on what was actually said before, rather than drifting "
     "or contradicting it?\n"
     "Approve if there are no substantive issues; do NOT reject over style, minor wording, or "
     "scope the student didn't ask for. If rejecting, give concrete, actionable feedback."),
    ("human",
     "REQUEST TYPE: {intent}\nSCOPE: {scope}\n\n"
     "CONVERSATION HISTORY:\n{history}\n\n"
     "STUDENT REQUEST:\n{query}\n\nCOURSE CONTEXT:\n{context}\n\nDRAFT TO REVIEW:\n{draft}"),
])


def critic_node(state: MentorState) -> dict:
    try:
        verdict: Verdict = (_PROMPT | get_structured_manager(Verdict)).invoke({
            "intent": state.get("intent", "mixed"),
            "scope": state.get("scope", "full"),
            "history": state.get("conversation_history") or "(none — this is the first turn)",
            "query": state["query"],
            "context": format_context(state.get("context_chunks", [])),
            "draft": state["draft"],
        })
        return {"verdict": verdict.model_dump()}
    except Exception as exc:  # noqa: BLE001 - e.g. rate limit; never loop forever or lose the draft
        log.warning("Critic failed: %s: %s", type(exc).__name__, exc)
        return {
            "verdict": {"approved": True, "issues": [], "feedback": ""},
            "warning": f"Critic unavailable ({type(exc).__name__}); this answer was NOT validated.",
        }
