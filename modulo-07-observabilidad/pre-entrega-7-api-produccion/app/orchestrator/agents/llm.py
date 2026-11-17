"""LLM factory and helpers shared by the agents.

Model sizing (unit 6): the supervisor coordinates and validates, so it gets the most capable model;
the specialists do narrow tasks with tools, so they get a light, cheap one.
Each role gets a list of models: the primary and, optionally, a fallback used if the primary fails.
"""

import asyncio
import json
import logging
import os
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any, Literal, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

logger = logging.getLogger("orchestrator.llm")

LLM_TIMEOUT = 60
LLM_MAX_RETRIES = 2
# Free tiers limit requests per minute (HTTP 429): wait and retry once before moving on
RATE_LIMIT_WAIT_SECONDS = float(os.getenv("RATE_LIMIT_WAIT_SECONDS", 30))
T = TypeVar("T")

Role = Literal["supervisor", "worker"]
_rate_limiter: Any = None


def shared_rate_limiter():
    """One limiter for every model of the process (LLM_REQUESTS_PER_MINUTE; 0 = off).

    Free tiers cap requests per minute (Gemini: ~15). With 5 jobs at once (~50 calls in a burst) the API would
    answer 429 again and again; the limiter spaces the calls instead, and the wait shows up in the LLM spans.
    Default: 12 per minute with Gemini (free tier), no limit with the other providers and the offline model.
    """
    global _rate_limiter
    default = "12" if os.getenv("LLM_PROVIDER", "gemini").lower() == "gemini" else "0"
    rpm = float(os.getenv("LLM_REQUESTS_PER_MINUTE", default) or 0)
    if rpm <= 0:
        return None
    if _rate_limiter is None:
        from langchain_core.rate_limiters import InMemoryRateLimiter

        _rate_limiter = InMemoryRateLimiter(requests_per_second=rpm / 60, check_every_n_seconds=0.1,
                                            max_bucket_size=2)
    return _rate_limiter


def chat_models(role: Role) -> list[BaseChatModel]:
    """Models for a role, in order of preference. LLM_PROVIDER: gemini (default), openai, anthropic or fake
    (offline rule-based model, no API key: for demos and tests)."""
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    if provider == "fake":
        from app.orchestrator.agents.fake_llm import fake_models
        return fake_models(role)
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        name = os.getenv("OPENAI_SUPERVISOR_MODEL", "gpt-4o") if role == "supervisor" else os.getenv("OPENAI_WORKER_MODEL", "gpt-4o-mini")
        return [ChatOpenAI(model=name, temperature=0, timeout=LLM_TIMEOUT, max_retries=LLM_MAX_RETRIES,
                           rate_limiter=shared_rate_limiter())]
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        name = os.getenv("ANTHROPIC_SUPERVISOR_MODEL", "claude-sonnet-4-5") if role == "supervisor" else os.getenv("ANTHROPIC_WORKER_MODEL", "claude-haiku-4-5")
        return [ChatAnthropic(model=name, max_tokens=2048, timeout=LLM_TIMEOUT, max_retries=LLM_MAX_RETRIES,
                              rate_limiter=shared_rate_limiter())]

    from langchain_google_genai import ChatGoogleGenerativeAI
    if role == "supervisor":
        names = [os.getenv("GEMINI_MODEL", "gemini-3.8-flash")]
    else:
        names = [os.getenv("GEMINI_WORKER_MODEL", "gemini-flash-lite-latest")]
    fallback = os.getenv("GEMINI_FALLBACK_MODEL")
    if fallback and fallback not in names:
        names.append(fallback)
    return [ChatGoogleGenerativeAI(model=n, temperature=0, timeout=LLM_TIMEOUT, max_retries=LLM_MAX_RETRIES,
                                   rate_limiter=shared_rate_limiter())
            for n in names]


def is_rate_limit(error: Exception) -> bool:
    text = f"{type(error).__name__} {error}"
    return "RateLimit" in text or "429" in text or "RESOURCE_EXHAUSTED" in text


# Models whose DAILY quota ran out, and until when (time.monotonic()). Found reading the pre-entrega 7 traces:
# with the daily quota of gemini-3.8-flash exhausted, every supervisor call waited 30 s and retried a model
# that would not answer for 8 hours: 31 failed calls and 740 s lost in one load test.
_exhausted_until: dict[str, float] = {}
DAILY_QUOTA_COOLDOWN_SECONDS = 3600   # when the error does not say how long to wait


def daily_quota_cooldown(error: Exception) -> float | None:
    """Seconds until a model's DAILY quota resets, if the error is a daily quota (not a per-minute limit)."""
    text = str(error)
    if not is_rate_limit(error) or not re.search(r"PerDay|per day", text, re.IGNORECASE):
        return None
    match = re.search(r"retryDelay'?\"?:\s*'?\"?(\d+)s", text)
    return float(match.group(1)) if match else DAILY_QUOTA_COOLDOWN_SECONDS


def is_exhausted(name: str) -> bool:
    until = _exhausted_until.get(name)
    if until is None:
        return False
    if time.monotonic() >= until:
        del _exhausted_until[name]
        return False
    return True


async def with_fallback(models: list[Any], call: Callable[[Any], Awaitable[T]], what: str) -> T:
    """Runs `call` with the first model; if it raises, tries the next one. Re-raises the last error.

    A rate limit (429) gets one retry on the same model after RATE_LIMIT_WAIT_SECONDS: switching to a
    fallback of the same provider usually hits the same per-minute quota.
    """
    last_error: Exception | None = None
    seen: set[str] = set()
    names = [getattr(m, "model", None) or str(id(m)) for m in models]
    available = [n for n in names if not is_exhausted(n)]
    for model, name in zip(models, names):
        if name in seen:   # the fallback is the same model as the primary: retrying it right away adds nothing
            continue
        seen.add(name)
        if is_exhausted(name) and available:   # daily quota spent: go straight to a model that can answer
            logger.info("%s: skipping %s (daily quota exhausted)", what, name)
            continue
        for attempt in (1, 2):
            try:
                return await call(model)
            except Exception as e:   # provider errors: 503 overloaded, timeout, quota...
                last_error = e
                logger.warning("%s failed with %s: %s", what, name, str(e)[:200])
                cooldown = daily_quota_cooldown(e)
                if cooldown is not None:   # a daily quota does not come back in 30 s: no wait, next model
                    _exhausted_until[name] = time.monotonic() + cooldown
                    logger.warning("%s: daily quota of %s exhausted for %.0f s; using the fallback", what, name, cooldown)
                    break
                if attempt == 1 and is_rate_limit(e) and RATE_LIMIT_WAIT_SECONDS > 0:
                    logger.warning("%s: rate limit, waiting %.0f s before retrying", what, RATE_LIMIT_WAIT_SECONDS)
                    await asyncio.sleep(RATE_LIMIT_WAIT_SECONDS)
                    continue
                break
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
