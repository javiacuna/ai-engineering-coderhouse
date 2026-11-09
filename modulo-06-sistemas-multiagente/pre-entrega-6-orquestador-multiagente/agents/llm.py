"""LLM factory and helpers shared by the agents.

Model sizing (unit 6): the supervisor coordinates and validates, so it gets the most capable model;
the specialists do narrow tasks with tools, so they get a light, cheap one.
Each role gets a list of models: the primary and, optionally, a fallback used if the primary fails.
"""

import json
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any, Literal, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

logger = logging.getLogger("orchestrator.llm")

LLM_TIMEOUT = 60
LLM_MAX_RETRIES = 2
T = TypeVar("T")

Role = Literal["supervisor", "worker"]


def chat_models(role: Role) -> list[BaseChatModel]:
    """Models for a role, in order of preference. LLM_PROVIDER: gemini (default), openai or anthropic."""
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        name = os.getenv("OPENAI_SUPERVISOR_MODEL", "gpt-4o") if role == "supervisor" else os.getenv("OPENAI_WORKER_MODEL", "gpt-4o-mini")
        return [ChatOpenAI(model=name, temperature=0, timeout=LLM_TIMEOUT, max_retries=LLM_MAX_RETRIES)]
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        name = os.getenv("ANTHROPIC_SUPERVISOR_MODEL", "claude-sonnet-4-5") if role == "supervisor" else os.getenv("ANTHROPIC_WORKER_MODEL", "claude-haiku-4-5")
        return [ChatAnthropic(model=name, max_tokens=2048, timeout=LLM_TIMEOUT, max_retries=LLM_MAX_RETRIES)]

    from langchain_google_genai import ChatGoogleGenerativeAI
    if role == "supervisor":
        names = [os.getenv("GEMINI_MODEL", "gemini-3.8-flash")]
    else:
        names = [os.getenv("GEMINI_WORKER_MODEL", "gemini-flash-lite-latest")]
    fallback = os.getenv("GEMINI_FALLBACK_MODEL")
    if fallback and fallback not in names:
        names.append(fallback)
    return [ChatGoogleGenerativeAI(model=n, temperature=0, timeout=LLM_TIMEOUT, max_retries=LLM_MAX_RETRIES)
            for n in names]


async def with_fallback(models: list[Any], call: Callable[[Any], Awaitable[T]], what: str) -> T:
    """Runs `call` with the first model; if it raises, tries the next one. Re-raises the last error."""
    last_error: Exception | None = None
    for model in models:
        try:
            return await call(model)
        except Exception as e:   # provider errors: 503 overloaded, timeout, quota...
            last_error = e
            logger.warning("%s failed with %s: %s", what, getattr(model, "model", type(model).__name__), str(e)[:200])
    raise last_error or RuntimeError(f"{what}: no models configured")


def tool_results(messages: list[BaseMessage]) -> list[tuple[str, dict[str, Any], Any]]:
    """(tool name, arguments, parsed result) for every tool call in a ReAct transcript."""
    calls = {c["id"]: c for m in messages if isinstance(m, AIMessage) for c in m.tool_calls}
    results = []
    for message in messages:
        if isinstance(message, ToolMessage):
            call = calls.get(message.tool_call_id, {})
            content = message.content
            try:
                content = json.loads(content) if isinstance(content, str) else content
            except json.JSONDecodeError:
                pass
            results.append((message.name or call.get("name", "?"), call.get("args", {}), content))
    return results


def final_text(messages: list[BaseMessage]) -> str:
    """Text of the last AI message (the agent's final answer)."""
    last = next((m for m in reversed(messages) if isinstance(m, AIMessage) and not m.tool_calls), None)
    return last.text.strip() if last is not None else ""
