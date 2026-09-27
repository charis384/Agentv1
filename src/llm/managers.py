"""Manager tier: smart models used sparingly, for short structured decisions (routing, validation)."""
from __future__ import annotations

from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

from src.config import require, settings


@lru_cache(maxsize=1)
def get_manager_llm() -> BaseChatModel:
    provider = settings.manager_provider
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=require(settings.google_api_key, "GOOGLE_API_KEY"),
            temperature=settings.manager_temperature,
        )
    if provider == "deepseek":
        # DeepSeek exposes an OpenAI-compatible API, so we reuse ChatOpenAI with a custom base_url.
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.deepseek_model,
            api_key=require(settings.deepseek_api_key, "DEEPSEEK_API_KEY"),
            base_url=settings.deepseek_base_url,
            temperature=settings.manager_temperature,
        )
    raise ValueError(f"Unknown MANAGER_PROVIDER '{provider}' (use 'gemini' or 'deepseek').")


@lru_cache(maxsize=8)
def get_structured_manager(schema: type[BaseModel]):
    """
    Manager LLM that returns a validated Pydantic object instead of free text.

    WHY: routing/validation decisions must be machine-readable (a bool, a list),
    not prose we would have to regex. Structured output also keeps responses tiny = cheap.
    """
    llm = get_manager_llm()
    if settings.manager_provider == "deepseek":
        # DeepSeek supports tool/function calling but not OpenAI's json_schema mode.
        return llm.with_structured_output(schema, method="function_calling")
    return llm.with_structured_output(schema)
