"""LLM client factory.

Groq exposes an OpenAI-compatible API, so we use LangChain's ChatOpenAI with a
different base URL, the same way the course's Praxa app used ChatOpenAI for
OpenRouter. The OpenAI client retries HTTP 429 (rate limit) responses with
exponential backoff, which matters on Groq's free plan.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI


def make_chat_llm(settings, model: str | None = None, max_tokens: int | None = None) -> ChatOpenAI:
    model = model or settings.llm_model
    extra = {}
    if "gpt-oss" in model:
        extra["reasoning_effort"] = "low"  # reasoning model: keep hidden reasoning short
    elif "qwen" in model:
        # Qwen3 models "think" first; ask Groq to keep the reasoning out of the reply.
        extra["extra_body"] = {"reasoning_format": "hidden"}
    return ChatOpenAI(
        model=model,
        base_url=settings.llm_base_url,
        api_key=settings.groq_api_key or "not-configured",
        temperature=0,
        max_tokens=max_tokens or settings.max_tokens,
        max_retries=settings.llm_max_retries,
        timeout=settings.llm_timeout,
        use_responses_api=False,  # Groq: plain Chat Completions endpoint
        **extra,
    )
