"""Tests of the trace analysis helpers (no Phoenix needed)."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze_traces import idle_gaps, agent_of, is_grounded, parse_numbers, tool_content, union_seconds  # noqa: E402


def ts(seconds: float) -> pd.Timestamp:
    return pd.Timestamp("2026-11-09T20:00:00Z") + pd.Timedelta(seconds=seconds)


def test_union_does_not_double_count_parallel_calls():
    intervals = [(ts(0), ts(2)), (ts(1), ts(3)), (ts(5), ts(6))]     # two overlapping + one separate
    assert union_seconds(intervals) == 4.0
    assert union_seconds([]) == 0.0


def test_parse_numbers_in_spanish_and_english_notation():
    text = "Promedio 4.17 estrellas (4,2), precio $182.000 contra $88.667; sentimiento 0.33 y 50% positivas."
    assert parse_numbers(text) >= {4.17, 4.2, 182000.0, 88667.0, 0.33, 50.0}


def test_grounded_numbers_allow_rounding_but_flag_invented_ones():
    sources = {4.17, 182000.0, 88666.67, 0.33, 50.0}
    for value in (4.17, 4.2, 182000.0, 88667.0, 0.33, 50.0):
        assert is_grounded(value, sources), value
    assert not is_grounded(4.9, sources)            # an invented rating
    assert not is_grounded(150000.0, sources)       # an invented price


def test_tool_content_unwraps_the_serialized_tool_message():
    inner = {"product": "SoundMax Pro", "found": True, "avg_rating": 4.17}
    wrapped = json.dumps({"type": "tool", "data": {"content": json.dumps(inner), "type": "tool"}})
    assert json.loads(tool_content(wrapped)) == inner
    assert tool_content(float("nan")) == ""
    assert tool_content("texto plano") == "texto plano"


def test_agent_of_walks_up_to_the_orchestrator_node():
    parents = {"llm": "agent", "agent": "researcher-node", "researcher-node": "root"}
    names = {"llm": "ChatGoogleGenerativeAI", "agent": "agent", "researcher-node": "researcher", "root": "consulta:q1"}
    assert agent_of("llm", parents, names) == "researcher"
    assert agent_of("root", parents, names) == "otro"


def test_parse_argentine_decimal_with_thousands():
    assert 88666.67 in parse_numbers("cuesta $88.666,67 en promedio")


def test_idle_gaps_counts_only_long_waits():
    import pandas as pd
    t = lambda s: pd.Timestamp("2026-01-01") + pd.Timedelta(seconds=s)
    work = [(t(0), t(2)), (t(2.3), t(4)), (t(34), t(36))]   # 0.3 s gap (code) + 30 s gap (backoff)
    assert abs(idle_gaps(work, t(0), t(36.2)) - 30) < 1e-6
