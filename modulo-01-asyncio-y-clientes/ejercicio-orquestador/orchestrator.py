"""
Module 1 exercise - Concurrent model orchestrator (Coderhouse AI Engineering)

Simulates a backend that queries several AI models at once:
  - Part A: fires 3 models at the same time with asyncio.gather
  - Part B: fires 10 calls, but a Semaphore only lets 2 run at a time
In both parts, each call has a 2-second timeout.
"""

import asyncio
import time

TIMEOUT_SECONDS = 2.0
MAX_CONCURRENT = 2

# Program start time, used to show relative timestamps in the logs
START = time.perf_counter()


def log(message: str) -> None:
    """Print a message with the seconds elapsed since the start."""
    elapsed = time.perf_counter() - START
    print(f"[{elapsed:5.2f}s] {message}")


# ---------------------------------------------------------------------------
# 1. Simulated model calls
#    Each one "waits" a different time, as if it were network latency.
# ---------------------------------------------------------------------------

async def gpt_4_call(prompt: str) -> str:
    log("gpt-4     -> start")
    await asyncio.sleep(1.0)  # await: yields control while "waiting for the network"
    log("gpt-4     -> done")
    return f"gpt-4 answered '{prompt}'"


async def claude_3_call(prompt: str) -> str:
    log("claude-3  -> start")
    await asyncio.sleep(1.5)
    log("claude-3  -> done")
    return f"claude-3 answered '{prompt}'"


async def local_llama_call(prompt: str) -> str:
    log("llama     -> start")
    await asyncio.sleep(3.0)  # slower than the timeout: fails on purpose
    log("llama     -> done")  # this line is never printed
    return f"llama answered '{prompt}'"


# ---------------------------------------------------------------------------
# 2. Timeout wrapper: if the call takes longer than 2s it is cancelled
#    and an error message is returned instead of crashing the program.
# ---------------------------------------------------------------------------

async def call_with_timeout(name: str, call) -> str:
    try:
        async with asyncio.timeout(TIMEOUT_SECONDS):
            return await call
    except TimeoutError:
        log(f"{name:<9} -> TIMEOUT: exceeded {TIMEOUT_SECONDS}s, cancelled")
        return f"ERROR: {name} did not answer in time"


# ---------------------------------------------------------------------------
# 3. Part A: the three models at the same time
# ---------------------------------------------------------------------------

async def part_a(prompt: str) -> None:
    log("=== PART A: 3 models concurrently with gather ===")
    results = await asyncio.gather(
        call_with_timeout("gpt-4", gpt_4_call(prompt)),
        call_with_timeout("claude-3", claude_3_call(prompt)),
        call_with_timeout("llama", local_llama_call(prompt)),
        return_exceptions=True,  # if something still blows up, it does not take down the others
    )
    for r in results:
        log(f"result: {r}")


# ---------------------------------------------------------------------------
# 4. Part B: 10 calls, but only 2 at a time (Semaphore)
# ---------------------------------------------------------------------------

async def limited_call(number: int, sem: asyncio.Semaphore) -> str:
    log(f"call {number:>2} -> queued")
    async with sem:  # only MAX_CONCURRENT calls get in at a time
        log(f"call {number:>2} -> ENTERS (running)")
        # call 7 is slow on purpose, to show the timeout here too
        latency = 2.5 if number == 7 else 0.5
        try:
            async with asyncio.timeout(TIMEOUT_SECONDS):
                await asyncio.sleep(latency)
        except TimeoutError:
            log(f"call {number:>2} -> TIMEOUT")
            return f"call {number}: ERROR timeout"
        log(f"call {number:>2} -> leaves")
        return f"call {number}: OK"


async def part_b() -> None:
    log(f"=== PART B: 10 calls, at most {MAX_CONCURRENT} at a time ===")
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    tasks = [limited_call(i, sem) for i in range(1, 11)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    ok = sum(1 for r in results if isinstance(r, str) and r.endswith("OK"))
    log(f"Summary: {ok}/10 OK, {10 - ok} with errors")


# ---------------------------------------------------------------------------
# 5. Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    await part_a("¿Qué es la entropía?")
    print()
    await part_b()


if __name__ == "__main__":
    asyncio.run(main())
