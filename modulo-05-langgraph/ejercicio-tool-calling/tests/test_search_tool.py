"""Tests of search_knowledge_base: success, invalid input and database failures (no LLM needed)."""

import asyncio
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import search_tool  # noqa: E402
from search_tool import search_knowledge_base  # noqa: E402


def run(args: dict):
    return asyncio.run(search_knowledge_base.ainvoke(args))


class RecordingDB:
    """Fake vector store that records the arguments it receives."""

    def __init__(self, results=None):
        self.calls = []
        self.results = ["Resultado para políticas de vacaciones"] if results is None else results

    async def similarity_search(self, query: str, k: int):
        self.calls.append((query, k))
        return self.results


class FailingDB:
    def __init__(self, error: Exception):
        self.error = error

    async def similarity_search(self, query: str, k: int):
        raise self.error


class SlowDB:
    async def similarity_search(self, query: str, k: int):
        await asyncio.sleep(1)
        return ["too late"]


# ---------------------------------------------------------------- contract
def test_tool_contract_exposes_name_description_and_schema():
    assert search_knowledge_base.name == "search_knowledge_base"
    assert "base de conocimiento" in search_knowledge_base.description
    args = search_knowledge_base.args
    assert set(args) == {"query", "limit"}
    assert args["query"]["description"] and args["limit"]["description"]
    assert args["limit"]["minimum"] == 1 and args["limit"]["maximum"] == 10 and args["limit"]["default"] == 3


# ---------------------------------------------------------------- success
def test_valid_search_returns_results(monkeypatch):
    fake = RecordingDB()
    monkeypatch.setattr(search_tool, "db", fake)
    assert run({"query": "políticas de vacaciones", "limit": 2}) == ["Resultado para políticas de vacaciones"]
    assert fake.calls == [("políticas de vacaciones", 2)]


def test_limit_defaults_to_3_and_query_is_cleaned(monkeypatch):
    fake = RecordingDB()
    monkeypatch.setattr(search_tool, "db", fake)
    run({"query": "  políticas \n de   vacaciones "})
    assert fake.calls == [("políticas de vacaciones", 3)]


def test_no_results_returns_a_message(monkeypatch):
    monkeypatch.setattr(search_tool, "db", RecordingDB(results=[]))
    assert run({"query": "algo que no existe"}).startswith("No se encontraron")


# ---------------------------------------------------------------- invalid input
@pytest.mark.parametrize("args, field", [
    ({"query": "políticas de vacaciones", "limit": 50}, "limit"),   # above the maximum
    ({"query": "políticas de vacaciones", "limit": 0}, "limit"),    # below the minimum
    ({"query": "ab"}, "query"),                                      # too short
    ({"query": "   a    "}, "query"),                                # only spaces around one character
    ({"limit": 2}, "query"),                                         # missing required field
])
def test_invalid_input_is_rejected_before_searching(monkeypatch, args, field):
    fake = RecordingDB()
    monkeypatch.setattr(search_tool, "db", fake)
    with pytest.raises(ValidationError) as error:
        run(args)
    assert error.value.errors()[0]["loc"] == (field,)
    assert fake.calls == []                     # the database was never touched


# ---------------------------------------------------------------- database failures
@pytest.mark.parametrize("exception", [ConnectionError("unreachable"), RuntimeError("index corrupted")])
def test_database_failure_returns_an_error_message(monkeypatch, exception):
    monkeypatch.setattr(search_tool, "db", FailingDB(exception))
    result = run({"query": "políticas de vacaciones"})
    assert isinstance(result, str) and result.startswith("Error:")
    assert type(exception).__name__ in result


def test_slow_database_times_out(monkeypatch):
    monkeypatch.setattr(search_tool, "db", SlowDB())
    monkeypatch.setattr(search_tool, "SEARCH_TIMEOUT_SECONDS", 0.05)
    assert run({"query": "políticas de vacaciones"}).startswith("Error: la base de conocimiento no respondió")
