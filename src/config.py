"""
config.py — the single place where settings and API keys are read.

WHY: agents and LLM factories never call os.getenv themselves. That makes
switching providers (ollama -> groq, gemini -> deepseek) a .env edit, and it
gives us one place to validate that required keys exist.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from ingest import IngestConfig  # reuse ingestion settings so DB name/embeddings can never drift

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


def _get(name: str, default: str = "") -> str:
    return os.getenv(name, default).split("#")[0].strip()  # tolerate inline "# comments"


# Vector DB location + embedding model come from the ingestion config (single source of truth).
INGEST_CFG = IngestConfig()
CHROMA_DIR = ROOT_DIR / INGEST_CFG.persist_dir


@dataclass(frozen=True)
class Settings:
    # Worker tier
    worker_provider: str = _get("WORKER_PROVIDER", "ollama").lower()
    ollama_model: str = _get("OLLAMA_MODEL", "llama3.1:8b")
    ollama_base_url: str = _get("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_num_ctx: int = int(_get("OLLAMA_NUM_CTX", "8192"))
    groq_model: str = _get("GROQ_MODEL", "llama-3.1-8b-instant")
    groq_api_key: str = _get("GROQ_API_KEY")

    # Manager tier
    manager_provider: str = _get("MANAGER_PROVIDER", "deepseek").lower()
    gemini_model: str = _get("GEMINI_MODEL", "gemini-3.1-flash-lite")
    google_api_key: str = _get("GOOGLE_API_KEY")
    deepseek_model: str = _get("DEEPSEEK_MODEL", "deepseek-chat")
    deepseek_api_key: str = _get("DEEPSEEK_API_KEY")
    deepseek_base_url: str = _get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    # Pipeline
    retrieval_k: int = int(_get("RETRIEVAL_K", "6"))
    max_revisions: int = int(_get("MAX_REVISIONS", "2"))
    worker_temperature: float = 0.2   # a little creativity for drafting
    manager_temperature: float = 0.0  # deterministic routing/validation

    # Conversation history (see src/utils/session.py). Now read by Router,
    # Drafter, AND Critic each turn, so raising this multiplies token cost
    # across three LLM calls, not one — tune with that in mind.
    history_max_turns: int = int(_get("HISTORY_MAX_TURNS", "8"))
    history_answer_chars: int = int(_get("HISTORY_ANSWER_CHARS", "400"))


settings = Settings()


def require(value: str, env_name: str) -> str:
    """Fail early with a clear message instead of a cryptic 401 deep in a library."""
    if not value:
        raise RuntimeError(f"Missing {env_name}. Set it in your .env file (see .env.example).")
    return value
