"""Sends one message to a thread. As a separate process, it proves the memory lives in SQLite,
not in the RAM of the script that created the conversation.

Usage:
    python ask.py --thread sesion-1 "¿Cuál era el total del primer cliente?"
    python ask.py --thread nueva "¿Cuál era el total del primer cliente?"   # new thread: no memory
"""

import argparse
import asyncio
import logging
import sys

from dotenv import load_dotenv

from agent import build_graph, build_model, run_turn, sqlite_checkpointer


async def main() -> int:
    parser = argparse.ArgumentParser(description="Ask the agent on a given thread")
    parser.add_argument("question")
    parser.add_argument("--thread", default="sesion-1", help="thread_id of the conversation")
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "google_genai", "aiosqlite"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    async with sqlite_checkpointer() as checkpointer:
        app = build_graph(build_model(), checkpointer)
        before = await app.aget_state({"configurable": {"thread_id": args.thread}})
        print(f"Thread '{args.thread}': {len(before.values.get('messages', []))} mensajes guardados antes de esta pregunta")
        trace = await run_turn(app, args.question, args.thread)
    print(f"\n💬 {trace.answer}\n({trace.tool_calls} llamadas a herramientas, {trace.messages_in_thread} mensajes en el thread)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
