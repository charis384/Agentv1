"""Drafter (Worker LLM): writes the answer/code from retrieved context; on retries, fixes the Critic's issues."""
from __future__ import annotations

from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.agents.retriever import format_context
from src.graph.state import MentorState
from src.llm.workers import get_worker_llm

_SYSTEM = (
    "You are a precise university teaching assistant.\n"
    "Rules:\n"
    "- Base your answer on the COURSE CONTEXT. Cite the sources you use as [1], [2], ... "
    "matching the context numbers.\n"
    "- Never invent APIs, annotations, or facts. If the context lacks something needed, say so "
    "explicitly instead of guessing.\n"
    "- If the COURSE CONTEXT includes an 'EXISTING SOURCE FILE' block, you are extending that "
    "real code. Use its exact class/field/method names and existing structure. Do NOT invent "
    "field names, GUI component names, or classes that are not shown in that source. If a needed "
    "detail (e.g. a GUI field's exact name) is not visible in the context, say so explicitly and "
    "ask for it rather than guessing a plausible-sounding name.\n"
    "- Address EVERY requirement of the student's request, and nothing beyond what was asked.\n\n"
    "REQUEST TYPE FOR THIS TURN: {mode_instruction}"
)

# WHY this exists: without an explicit, enforced instruction, the model defaults
# to "assignment => write all the code," even for a request scoped to one method.
# intent controls WHAT KIND of content; scope controls HOW MUCH of the surrounding
# program must come with it. Keyed as (intent, scope).
_MODE_INSTRUCTIONS = {
    ("theory", "snippet"): (
        "THEORY/EXPLANATION, NARROWLY SCOPED. Answer only the specific point asked, briefly. "
        "Do not write a complete implementation or cover requirements the student didn't ask about."
    ),
    ("theory", "full"): (
        "THEORY/EXPLANATION ONLY. The student wants to understand what to do, not a finished "
        "solution. Give a clear explanation and, if useful, a numbered action plan of the steps "
        "involved. Do NOT write a complete, runnable implementation. You may include a very short "
        "(a few lines) illustrative snippet only if a concept is otherwise hard to convey, but do "
        "not produce full classes or complete methods."
    ),
    ("code", "snippet"): (
        "CODE, NARROWLY SCOPED. The student asked for a SPECIFIC piece — write ONLY that: the "
        "named method(s)/class/fix, nothing else. Do NOT add surrounding classes, GUI code, "
        "persistence/serialization logic, or unrelated methods the student did not ask for, even "
        "if they would be needed to run the whole program. If the piece you write depends on "
        "something not shown in the context (e.g. a field name), state that assumption in one line "
        "rather than inventing or building the surrounding structure to justify it. Keep the "
        "explanation to 1-2 sentences."
    ),
    ("code", "full"): (
        "CODE, FULL IMPLEMENTATION. Provide complete, runnable code with file names and short "
        "comments, then a brief explanation of the key design decisions."
    ),
    ("mixed", "snippet"): (
        "MIXED, NARROWLY SCOPED. Briefly explain the specific point asked, then give only the "
        "code for that specific piece — not a full program."
    ),
    ("mixed", "full"): (
        "MIXED, FULL IMPLEMENTATION. Briefly explain the approach, then provide complete, runnable "
        "code for it."
    ),
}
_DEFAULT_MODE = _MODE_INSTRUCTIONS[("mixed", "full")]

_HUMAN = (
    "STUDENT REQUEST:\n{query}\n\n"
    "COURSE CONTEXT:\n{context}\n"
    "{revision_block}"
)


@lru_cache(maxsize=1)
def _chain():
    prompt = ChatPromptTemplate.from_messages([("system", _SYSTEM), ("human", _HUMAN)])
    return prompt | get_worker_llm() | StrOutputParser()


def _revision_block(state: MentorState) -> str:
    verdict = state.get("verdict")
    if not verdict or verdict.get("approved") or not state.get("draft"):
        return ""
    issues = "\n".join(f"- {i}" for i in verdict.get("issues", []))
    return (
        "\n--- YOUR PREVIOUS DRAFT (rejected by the reviewer) ---\n"
        f"{state['draft']}\n"
        "--- REVIEWER FEEDBACK: fix ALL of it and output the complete corrected answer ---\n"
        f"{issues}\n{verdict.get('feedback', '')}\n"
    )


def drafter_node(state: MentorState) -> dict:
    key = (state.get("intent", "mixed"), state.get("scope", "full"))
    mode = _MODE_INSTRUCTIONS.get(key, _DEFAULT_MODE)
    draft = _chain().invoke({
        "query": state["query"],
        "context": format_context(state.get("context_chunks", [])),
        "revision_block": _revision_block(state),
        "mode_instruction": mode,
    })
    return {"draft": draft, "revision_count": state.get("revision_count", 0) + 1}
