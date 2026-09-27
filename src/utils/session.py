"""
session.py — lightweight persistent conversation memory.

WHY JSON on disk, not a database: a session is a handful of turns for one
student working through one assignment. JSON is human-readable (you can open
a session file directly and read exactly what was asked/answered), needs no
extra service, and is trivial to back up, inspect, or delete.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from src.config import settings

SESSIONS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "sessions"


class SessionStore:
    def __init__(self, name: str, persist: bool = True):
        self.name = name
        self.persist = persist  # False => in-memory only for this process; nothing read/written on disk
        self.path = SESSIONS_DIR / f"{name}.json"
        self.turns: list[dict] = self._load() if persist else []

    def _load(self) -> list[dict]:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return []  # corrupted file: start fresh rather than crash the whole run
        return []

    def clear(self) -> None:
        self.turns = []
        if self.persist:
            self._save()

    def add_turn(self, *, query: str, intent: str, scope: str, domain: str | None,
                 answer: str, warning: str | None, sources: list[str]) -> None:
        self.turns.append({
            "ts": dt.datetime.now().isoformat(timespec="seconds"),
            "query": query,
            "intent": intent,
            "scope": scope,
            "domain": domain,
            "answer": answer,
            "warning": warning,
            "sources": sources,
        })
        if self.persist:
            self._save()

    def _save(self) -> None:
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.turns, indent=2, ensure_ascii=False), encoding="utf-8")

    def conversation_history_text(self) -> str:
        """Formats the last settings.history_max_turns turns for prompt injection. Empty on a fresh session."""
        recent = self.turns[-settings.history_max_turns:]
        if not recent:
            return ""
        blocks = []
        for t in recent:
            answer = t["answer"]
            if len(answer) > settings.history_answer_chars:
                answer = answer[:settings.history_answer_chars] + " ... [truncated]"
            blocks.append(f"Student previously asked: {t['query']}\nYou previously answered: {answer}")
        return "\n\n".join(blocks)

    @staticmethod
    def list_sessions() -> list[tuple[str, int, str]]:
        """Returns (name, turn_count, last_updated_timestamp) for every session on disk."""
        if not SESSIONS_DIR.exists():
            return []
        out = []
        for path in sorted(SESSIONS_DIR.glob("*.json")):
            try:
                turns = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                turns = []
            last = turns[-1]["ts"] if turns else "-"
            out.append((path.stem, len(turns), last))
        return out
