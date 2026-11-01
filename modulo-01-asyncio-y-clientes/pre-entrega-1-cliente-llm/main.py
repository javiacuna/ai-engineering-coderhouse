"""Validation script: one short question in normal mode and in streaming mode.

Usage:
  python main.py                       # uses LLM_PROVIDER from .env
  python main.py --provider gemini     # forces a provider (openai | anthropic | gemini)
"""

import argparse
import asyncio
import logging

from dotenv import load_dotenv
from pydantic import ValidationError

from clients import LLMClientError
from manager import AsyncLLMManager

QUESTION = "¿Qué es la entropía?"
SYSTEM = "Answer in Spanish, in at most 3 sentences."


async def main(provider: str | None) -> None:
    try:
        manager = AsyncLLMManager.from_env(provider)
    except ValidationError as e:
        print(f"Invalid configuration: {e.errors()[0]['msg']}")
        return
    print(manager, "\n")

    # 1) Normal mode: full response
    print(f"=== Normal mode · {QUESTION} ===")
    r = await manager.generate(QUESTION, system=SYSTEM)
    if r.ok:
        print(r.content)
        usage = f"{r.usage.input_tokens} in / {r.usage.output_tokens} out" if r.usage else "n/a"
        print(f"\n[{r.latency_ms:.0f} ms · tokens: {usage} · finish: {r.finish_reason}]")
    else:
        print(f"Controlled error -> type={r.error.type} | retryable={r.error.retryable} | {r.error.message}")

    # 2) Streaming mode: chunks are printed as they arrive
    print(f"\n=== Streaming mode · {QUESTION} ===")
    chunks = 0
    try:
        async for chunk in manager.stream(QUESTION, system=SYSTEM):
            print(chunk, end="", flush=True)
            chunks += 1
        print(f"\n\n[{chunks} chunks received]")
    except LLMClientError as e:
        print(f"\nControlled error -> type={e.error.type} | retryable={e.error.retryable} | {e.error.message}")


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(name)s | %(message)s")
    parser = argparse.ArgumentParser(description="Unified Async LLM Client test")
    parser.add_argument("--provider", choices=["openai", "anthropic", "gemini"], help="overrides LLM_PROVIDER")
    asyncio.run(main(parser.parse_args().provider))
