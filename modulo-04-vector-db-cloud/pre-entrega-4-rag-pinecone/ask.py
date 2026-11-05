"""Asks the RAG system a question: shows the top-5 hybrid chunks and, with --answer, a grounded answer.

Usage:
    python ask.py "¿Para qué sirve asyncio.shield?"
    python ask.py "¿Para qué sirve asyncio.shield?" --mode bm25
    python ask.py "¿Para qué sirve asyncio.shield?" --answer
"""

import argparse
import asyncio
import logging
import sys

from dotenv import load_dotenv
from pinecone.errors import PineconeError

from config import Settings
from rag_system import IndexNotReadyError, RAGSystem


async def run(question: str, mode: str, answer: bool) -> None:
    rag = RAGSystem(Settings.from_env())
    if answer:
        result = await rag.answer(question)
        print(f"\n💬 {result.answer}\n")
        chunks = result.sources
    else:
        chunks = rag.describe(await rag.aretrieve(question, mode=mode))
    print(f"Top-{len(chunks)} ({'hybrid' if answer else mode}):")
    for c in chunks:
        print(f"  [{c.rank}] {c.chunk_id} | {c.section} (pág. {c.page}) | {c.category}")
        print(f"      {c.excerpt}...")


def main() -> int:
    parser = argparse.ArgumentParser(description="Query the cloud RAG system")
    parser.add_argument("question")
    parser.add_argument("--mode", choices=["hybrid", "vector", "bm25"], default="hybrid")
    parser.add_argument("--answer", action="store_true", help="also generate a grounded answer with Gemini")
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(name)s | %(message)s")
    try:
        asyncio.run(run(args.question, args.mode, args.answer))
    except (IndexNotReadyError, ValueError, PineconeError) as e:
        print(f"Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
