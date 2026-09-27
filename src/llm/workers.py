"""Worker tier: cheap/local models that do the high-volume token work (drafting)."""
from __future__ import annotations

from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel

from src.config import require, settings


@lru_cache(maxsize=1)
def get_worker_llm() -> BaseChatModel:
    provider = settings.worker_provider
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=settings.worker_temperature,
            # WHY: Ollama's default context window is small (2-4k tokens). RAG context
            # + prompt would be silently truncated and quality would collapse.
            num_ctx=settings.ollama_num_ctx,
        )
    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=settings.groq_model,
            api_key=require(settings.groq_api_key, "GROQ_API_KEY"),
            temperature=settings.worker_temperature,
        )
    raise ValueError(f"Unknown WORKER_PROVIDER '{provider}' (use 'ollama' or 'groq').")
