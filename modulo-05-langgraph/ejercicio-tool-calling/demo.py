"""Manual demo of search_knowledge_base (no LLM needed).

Usage:
    python demo.py
"""

import asyncio

from pydantic import ValidationError

import search_tool
from search_tool import search_knowledge_base


class BrokenVectorDB:
    async def similarity_search(self, query: str, k: int):
        raise ConnectionError("vector store unreachable")


def main() -> None:
    # 1. The contract the LLM sees: name and argument schema (with descriptions)
    print(search_knowledge_base.name, search_knowledge_base.args)

    # 2. Valid call
    print(asyncio.run(search_knowledge_base.ainvoke({"query": "políticas de vacaciones", "limit": 2})))

    # 3. Invalid limit: Pydantic rejects it before the function runs
    try:
        asyncio.run(search_knowledge_base.ainvoke({"query": "políticas de vacaciones", "limit": 50}))
    except ValidationError as e:
        print(f"Validation error: {e.errors()[0]['loc']} -> {e.errors()[0]['msg']}")

    # 4. Infrastructure failure: the error comes back as a message, the program keeps running
    original_db, search_tool.db = search_tool.db, BrokenVectorDB()
    try:
        print(asyncio.run(search_knowledge_base.ainvoke({"query": "políticas de vacaciones"})))
    finally:
        search_tool.db = original_db


if __name__ == "__main__":
    main()
