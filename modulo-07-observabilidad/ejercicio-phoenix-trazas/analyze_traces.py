"""Reads the traces of a session from Phoenix and computes the analysis of the assignment.

For every request (trace):
- latency split into EXTERNAL calls (LLM API = network + generation; data sources) and LOCAL processing
  (tool computations + orchestration), using the union of time intervals so parallel calls are not
  counted twice;
- the slowest span, the tokens per agent and an estimated cost;
- errors recorded in the spans;
- a groundedness check: every number in the final answer must appear in the tool outputs of that trace
  (or in the user's request). A number with no source is evidence of hallucination.
Plus p50/p95 latencies for the session and the most token-hungry request.

Usage:
    python analyze_traces.py --project orquestador-m7-20261109-2010
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / "results"
AGENTS = ("supervisor", "researcher", "analyst", "synthesizer")

# USD per 1M tokens (input, output), paid tier, as listed on Oct 25 2026 (morphllm.com/gemini-api-pricing).
# The free tier costs 0: these estimate what the same traffic would cost on the paid tier.
# gemini-flash-lite-latest is priced as gemini-3.5-flash-lite (assumption: the alias points to the newest Flash-Lite).
PRICES = {"flash-lite": (0.30, 2.50), "3.8-flash": (0.75, 3.75), "gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.50, 10.00)}
DEFAULT_PRICE = PRICES["flash-lite"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_spans(project: str, endpoint: str) -> pd.DataFrame:
    from phoenix.client import Client

    df = Client(base_url=endpoint).spans.get_spans_dataframe(project_name=project, limit=100_000, timeout=60)
    if df.empty:
        raise SystemExit(f"No spans in project '{project}'")
    df = df.reset_index(drop=True)
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
    df["end_time"] = pd.to_datetime(df["end_time"], utc=True)
    df["duration_s"] = (df["end_time"] - df["start_time"]).dt.total_seconds()
    return df


def col(df: pd.DataFrame, name: str, default: Any = None) -> pd.Series:
    return df[name] if name in df.columns else pd.Series([default] * len(df), index=df.index)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def union_seconds(intervals: list[tuple[pd.Timestamp, pd.Timestamp]]) -> float:
    """Total time covered by possibly overlapping intervals (parallel tool calls are not double counted)."""
    total, current_start, current_end = 0.0, None, None
    for start, end in sorted(intervals):
        if current_end is None or start > current_end:
            if current_end is not None:
                total += (current_end - current_start).total_seconds()
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_end is not None:
        total += (current_end - current_start).total_seconds()
    return total


def idle_gaps(intervals: list[tuple[pd.Timestamp, pd.Timestamp]], start: pd.Timestamp, end: pd.Timestamp,
              threshold: float = 1.0) -> float:
    """Seconds with NO work span running (LLM, tool or data source), counting only gaps longer than
    `threshold`. Short gaps are orchestration code; long ones are waits, such as a backoff after a 429."""
    merged: list[list[pd.Timestamp]] = []
    for s_, e_ in sorted(intervals):
        if merged and s_ <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e_)
        else:
            merged.append([s_, e_])
    edges = [start] + [t for pair in merged for t in pair] + [end]
    gaps = [(edges[i + 1] - edges[i]).total_seconds() for i in range(0, len(edges) - 1, 2)]
    return sum(g for g in gaps if g > threshold)


def agent_of(span_id: str, parents: dict[str, str], names: dict[str, str]) -> str:
    """Nearest ancestor that is one of the orchestrator's agents (LangGraph node spans carry their name)."""
    current = span_id
    for _ in range(50):
        if names.get(current) in AGENTS:
            return names[current]
        current = parents.get(current)
        if current is None:
            break
    return "otro"


def tool_content(output: Any) -> str:
    """The tool result inside the serialized ToolMessage that LangChain records as output.value."""
    text = "" if output is None or (isinstance(output, float) and pd.isna(output)) else str(output)
    try:
        data = json.loads(text)
        content = data.get("data", {}).get("content", text) if isinstance(data, dict) else text
        return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    except (json.JSONDecodeError, AttributeError):
        return text


NUMBER = re.compile(r"\d[\d.,]*\d|\d")


def parse_numbers(text: str) -> set[float]:
    """Numbers in Spanish or English notation: $182.000 -> 182000, 4,2 -> 4.2, 88.667 -> 88667, 0.33 -> 0.33."""
    found = set()
    for raw in NUMBER.findall(text or ""):
        token = raw.strip(".,")
        if re.fullmatch(r"\d{1,3}(\.\d{3})+,\d+", token):     # 88.666,67 (Argentine: thousands dot, decimal comma)
            value = float(token.replace(".", "").replace(",", "."))
        elif re.fullmatch(r"\d{1,3}(\.\d{3})+", token):        # 182.000 / 1.340 (thousands with dots)
            value = float(token.replace(".", ""))
        elif re.fullmatch(r"\d{1,3}(,\d{3})+", token):         # 182,000 (thousands with commas)
            value = float(token.replace(",", ""))
        elif "," in token and "." not in token:                # 4,2 (decimal comma)
            value = float(token.replace(",", "."))
        else:
            try:
                value = float(token.replace(",", ""))
            except ValueError:
                continue
        found.add(round(value, 4))
    return found


def is_grounded(value: float, sources: set[float]) -> bool:
    """A number is grounded if a source number matches it, allowing rounding (4.17 -> 4.2, 88666.67 -> 88.667)."""
    for src in sources:
        if abs(value - src) <= 0.051 or (src and abs(value - src) / abs(src) <= 0.005):
            return True
        if src and abs(value - round(src)) < 1e-9:              # 88666.67 shown as 88.667 (thousands)
            return True
    return False


# ---------------------------------------------------------------------------
# Per-trace analysis
# ---------------------------------------------------------------------------
def analyze_trace(spans: pd.DataFrame, prices: dict[str, tuple[float, float]]) -> dict[str, Any]:
    root = spans[spans["name"].str.startswith("consulta:")].iloc[0]
    parents = dict(zip(spans["context.span_id"], col(spans, "parent_id")))
    names = dict(zip(spans["context.span_id"], spans["name"]))
    total = (root["end_time"] - root["start_time"]).total_seconds()

    llm = spans[spans["span_kind"] == "LLM"].copy()
    tools = spans[spans["span_kind"] == "TOOL"]
    retrievers = spans[spans["span_kind"] == "RETRIEVER"]
    llm["agent"] = [agent_of(s, parents, names) for s in llm["context.span_id"]]

    llm_time = union_seconds(list(zip(llm["start_time"], llm["end_time"])))
    source_time = union_seconds(list(zip(retrievers["start_time"], retrievers["end_time"])))
    tool_wall = union_seconds(list(zip(tools["start_time"], tools["end_time"])))
    external = union_seconds(list(zip(llm["start_time"], llm["end_time"])) + list(zip(retrievers["start_time"], retrievers["end_time"])))
    work_intervals = list(zip(llm["start_time"], llm["end_time"])) + list(zip(tools["start_time"], tools["end_time"]))
    busy = union_seconds(work_intervals)
    waiting = idle_gaps(work_intervals, root["start_time"], root["end_time"])

    # Tokens and cost
    prompt_tokens = col(llm, "attributes.llm.token_count.prompt", 0).fillna(0)
    completion_tokens = col(llm, "attributes.llm.token_count.completion", 0).fillna(0)
    models = col(llm, "attributes.llm.model_name", "").fillna("")
    cost = 0.0
    for model, p_tok, c_tok in zip(models, prompt_tokens, completion_tokens):
        price_in, price_out = next((v for k, v in prices.items() if k in str(model)), DEFAULT_PRICE)
        cost += p_tok / 1e6 * price_in + c_tok / 1e6 * price_out
    by_agent = {}
    for agent, group in llm.assign(p=prompt_tokens, c=completion_tokens).groupby("agent"):
        by_agent[agent] = {"llm_calls": int(len(group)), "prompt_tokens": int(group["p"].sum()),
                           "completion_tokens": int(group["c"].sum()),
                           "llm_seconds": round(float(group["duration_s"].sum()), 2)}

    # Slowest unit of work
    work = spans[spans["span_kind"].isin(["LLM", "TOOL", "RETRIEVER"])]
    slowest = work.loc[work["duration_s"].idxmax()] if not work.empty else None

    # Errors: where they originated (tool, data source or LLM) and how far up they propagated
    errors = spans[spans["status_code"] == "ERROR"]
    origin = errors[errors["span_kind"].isin(["TOOL", "RETRIEVER", "LLM"])]
    error_list, seen = [], set()
    for _, r in origin.iterrows():
        message = str(r.get("status_message") or "").strip().splitlines()[0][:120] if r.get("status_message") else ""
        if (r["name"], message) not in seen:
            seen.add((r["name"], message))
            error_list.append({"span": r["name"], "kind": r["span_kind"], "message": message,
                               "agent": agent_of(r["context.span_id"], parents, names)})
    propagated = int((errors["span_kind"].isin(["CHAIN", "AGENT"])).sum())

    # Groundedness: numbers in the answer vs numbers in the tool outputs (+ the user's request)
    answer = str(root.get("attributes.output.value") or "")
    task = str(root.get("attributes.input.value") or "")
    tool_outputs = [tool_content(v) for v in col(tools, "attributes.output.value", "")]
    source_numbers = set().union(*(parse_numbers(o) for o in tool_outputs), parse_numbers(task), set())
    answer_numbers = parse_numbers(answer)
    ungrounded = sorted(n for n in answer_numbers if not is_grounded(n, source_numbers))
    found_data = any('"found": true' in o or '"avg_rating"' in o for o in tool_outputs)
    admits_missing = bool(re.search(r"no (hay|tengo|se encontr|encontr|dispon|pude|cuento|cuenta con|fue posible|"
                                    r"tenemos|contamos)|sin datos|no existe", answer.lower()))

    return {
        "trace_id": root["context.trace_id"],
        "scenario": root["name"].removeprefix("consulta:"),
        "total_seconds": round(total, 2),
        "external_seconds": round(external, 2),                      # LLM API + data sources (network)
        "llm_seconds": round(llm_time, 2),
        "data_source_seconds": round(source_time, 2),
        "local_processing_seconds": round(max(total - external - waiting, 0), 2),   # tool computations + orchestration
        "tool_compute_seconds": round(max(tool_wall - source_time, 0), 2),
        "orchestration_seconds": round(max(total - busy - waiting, 0), 2),
        "wait_seconds": round(waiting, 2),                                           # idle gaps > 1 s (backoff)
        "llm_calls": int(len(llm)),
        "tool_calls": int(len(tools)),
        "prompt_tokens": int(prompt_tokens.sum()),
        "completion_tokens": int(completion_tokens.sum()),
        "estimated_cost_usd": round(cost, 6),
        "tokens_by_agent": by_agent,
        "slowest_span": None if slowest is None else {
            "name": slowest["name"], "kind": slowest["span_kind"], "seconds": round(float(slowest["duration_s"]), 2),
            "agent": agent_of(slowest["context.span_id"], parents, names)},
        "errors": error_list,
        "error_propagated_spans": propagated,
        "groundedness": {
            "answer_numbers": sorted(answer_numbers),
            "ungrounded_numbers": ungrounded,
            "grounded": not ungrounded and (found_data or admits_missing),
            "used_tool_data": found_data,
            "admits_missing_data": admits_missing,
        },
        "final_answer": answer,
    }


def percentile(values: list[float], q: float) -> float:
    return round(float(pd.Series(values).quantile(q)), 2) if values else 0.0


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def build_report(project: str, traces: list[dict[str, Any]], spans: pd.DataFrame) -> dict[str, Any]:
    llm_durations = spans.loc[spans["span_kind"] == "LLM", "duration_s"].tolist()
    totals = [t["total_seconds"] for t in traces]
    longest = max(traces, key=lambda t: t["prompt_tokens"] + t["completion_tokens"])
    return {
        "project": project,
        "requests": len(traces),
        "latency": {
            "request_p50": percentile(totals, 0.5), "request_p95": percentile(totals, 0.95),
            "llm_call_p50": percentile(llm_durations, 0.5), "llm_call_p95": percentile(llm_durations, 0.95),
            "share_external_pct": round(100 * sum(t["external_seconds"] for t in traces) / max(sum(totals), 1e-9), 1),
        "share_wait_pct": round(100 * sum(t["wait_seconds"] for t in traces) / max(sum(totals), 1e-9), 1),
        "request_p95_without_waits": percentile([t["total_seconds"] - t["wait_seconds"] for t in traces], 0.95),
        },
        "tokens": {
            "total_prompt": sum(t["prompt_tokens"] for t in traces),
            "total_completion": sum(t["completion_tokens"] for t in traces),
            "estimated_cost_usd": round(sum(t["estimated_cost_usd"] for t in traces), 6),
            "longest_reasoning": {"scenario": longest["scenario"], "trace_id": longest["trace_id"],
                                  "prompt_tokens": longest["prompt_tokens"],
                                  "completion_tokens": longest["completion_tokens"],
                                  "llm_calls": longest["llm_calls"],
                                  "estimated_cost_usd": longest["estimated_cost_usd"],
                                  "by_agent": longest["tokens_by_agent"]},
        },
        "traces": traces,
    }


def markdown(report: dict[str, Any]) -> str:
    lat, tok = report["latency"], report["tokens"]
    lines = [
        f"# Análisis de trazas · proyecto `{report['project']}`", "",
        f"{report['requests']} consultas · p50 {lat['request_p50']} s · p95 {lat['request_p95']} s · "
        f"llamadas al LLM: p50 {lat['llm_call_p50']} s, p95 {lat['llm_call_p95']} s · "
        f"{lat['share_external_pct']}% del tiempo en llamadas externas · {lat['share_wait_pct']}% en esperas "
        f"(p95 sin esperas: {lat['request_p95_without_waits']} s)", "",
        "## Latencia por consulta (externa vs. procesamiento local)", "",
        "| Consulta | Total | LLM (API) | Fuentes de datos | Procesamiento local | Espera | Span más lento | Llamadas LLM | Tokens (in/out) | Costo est. |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for t in report["traces"]:
        s = t["slowest_span"] or {}
        lines.append(f"| {t['scenario']} | {t['total_seconds']} s | {t['llm_seconds']} s | {t['data_source_seconds']} s | "
                     f"{t['local_processing_seconds']} s | {t['wait_seconds']} s | {s.get('name')} ({s.get('agent')}, {s.get('seconds')} s) | "
                     f"{t['llm_calls']} | {t['prompt_tokens']}/{t['completion_tokens']} | ${t['estimated_cost_usd']:.5f} |")
    lines += ["", "## Groundedness y errores", "",
              "| Consulta | ¿Grounded? | Números sin fuente | Usó datos de herramientas | Reconoce falta de datos | Errores en spans |",
              "|---|---|---|---|---|---|"]
    for t in report["traces"]:
        g = t["groundedness"]
        errs = "; ".join(f"{e['span']} ({e['agent']}): {e['message'][:70]}" for e in t["errors"]) or "—"
        if t["errors"]:
            errs += f" · propagado a {t['error_propagated_spans']} spans superiores"
        errs = errs.replace("|", "/")
        lines.append(f"| {t['scenario']} | {'sí' if g['grounded'] else '**NO**'} | {g['ungrounded_numbers'] or '—'} | "
                     f"{'sí' if g['used_tool_data'] else 'no'} | {'sí' if g['admits_missing_data'] else 'no'} | {errs} |")
    lr = tok["longest_reasoning"]
    lines += ["", f"## Razonamiento más largo: `{lr['scenario']}`", "",
              f"{lr['llm_calls']} llamadas al LLM · {lr['prompt_tokens']} tokens de entrada · "
              f"{lr['completion_tokens']} de salida · costo estimado ${lr['estimated_cost_usd']:.5f}", "",
              "| Agente | Llamadas | Tokens de entrada | Tokens de salida | Segundos en el LLM |", "|---|---|---|---|---|"]
    for agent, a in sorted(lr["by_agent"].items(), key=lambda kv: -kv[1]["prompt_tokens"]):
        lines.append(f"| {agent} | {a['llm_calls']} | {a['prompt_tokens']} | {a['completion_tokens']} | {a['llm_seconds']} |")
    lines += ["", f"Total de la sesión: {tok['total_prompt']} tokens de entrada, {tok['total_completion']} de salida, "
                  f"costo estimado ${tok['estimated_cost_usd']:.5f} (en el plan pago; el plan gratuito no cobra)."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze a Phoenix tracing session")
    parser.add_argument("--project", default=None, help="Phoenix project (default: the one in results/traffic_run.json)")
    parser.add_argument("--endpoint", default="http://localhost:6006")
    args = parser.parse_args()

    project = args.project
    if project is None:
        run_file = RESULTS_DIR / "traffic_run.json"
        if not run_file.exists():
            print("Pasá --project o corré antes run_traffic.py")
            return 1
        project = json.loads(run_file.read_text(encoding="utf-8"))["project"]

    spans = load_spans(project, args.endpoint)
    traces = [analyze_trace(group, PRICES) for _, group in spans.groupby("context.trace_id")
              if group["name"].str.startswith("consulta:").any()]
    order = {s: i for i, s in enumerate(["q1", "q2", "q3", "q4", "q5", "f1", "f2"])}
    traces.sort(key=lambda t: order.get(t["scenario"][:2], 99))
    report = build_report(project, traces, spans)

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "trace_analysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str),
                                                     encoding="utf-8")
    (RESULTS_DIR / "trace_analysis.md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report))
    print("Guardado en results/trace_analysis.md y results/trace_analysis.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
