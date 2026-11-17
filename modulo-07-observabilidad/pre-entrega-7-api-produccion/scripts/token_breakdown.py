"""Tokens, cost and latency of the LLM calls per agent, read from Phoenix.

The traces table of Phoenix shows each LLM call as "ChatGoogleGenerativeAI", without saying which agent made
it. That information IS in each span: its metadata carries `langgraph_checkpoint_ns`, the path of graph nodes
that led to the call (e.g. "run_orchestrator:…|researcher:…|agent:…"). This script groups the LLM spans of a
project by that agent and answers "which node of the graph consumes the most tokens".

Usage (after the load test, with Phoenix running):
    python scripts/token_breakdown.py                       # project from PHOENIX_PROJECT_NAME
    python scripts/token_breakdown.py --project pe7-carga --minutes 60
"""

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import find_dotenv, load_dotenv

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
AGENTS = ("supervisor", "researcher", "analyst", "synthesizer")
# USD per million tokens (input, output): the same prices Phoenix uses in its built-in price list
PRICES = {"gemini-3.8-flash": (0.75, 3.75), "gemini-flash-lite-latest": (0.30, 2.50)}


def agent_of(metadata) -> str:
    """The orchestrator agent in the node path of a span (or the outer node: assess_risk, ...)."""
    if not isinstance(metadata, dict):
        return "?"
    path = str(metadata.get("langgraph_checkpoint_ns") or "")
    nodes = [part.split(":")[0] for part in path.split("|") if part]
    return next((n for n in nodes if n in AGENTS), metadata.get("langgraph_node") or "?")


def column(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name] if name in df.columns else pd.Series([None] * len(df), index=df.index)


def main() -> int:
    load_dotenv(find_dotenv(usecwd=True))
    parser = argparse.ArgumentParser(description="LLM tokens and cost per agent, from Phoenix")
    parser.add_argument("--phoenix", default=os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006"))
    parser.add_argument("--project", default=os.getenv("PHOENIX_PROJECT_NAME", "pe7-api-agente"))
    parser.add_argument("--minutes", type=float, default=120, help="only spans from the last N minutes")
    args = parser.parse_args()

    from phoenix.client import Client

    start = datetime.now(timezone.utc) - timedelta(minutes=args.minutes)
    spans = Client(base_url=args.phoenix).spans.get_spans_dataframe(project_name=args.project, start_time=start,
                                                                    limit=100_000, timeout=120)
    llm = spans[spans["span_kind"] == "LLM"].copy()
    if llm.empty:
        print(f"No hay spans LLM en el proyecto {args.project!r} en los últimos {args.minutes:.0f} minutos.")
        return 1
    llm["agent"] = column(llm, "attributes.metadata").map(agent_of)
    llm["model"] = column(llm, "attributes.llm.model_name").fillna("?")
    llm["input"] = pd.to_numeric(column(llm, "attributes.llm.token_count.prompt"), errors="coerce").fillna(0)
    llm["output"] = pd.to_numeric(column(llm, "attributes.llm.token_count.completion"), errors="coerce").fillna(0)
    llm["seconds"] = (llm["end_time"] - llm["start_time"]).dt.total_seconds()
    llm["error"] = llm["status_code"] == "ERROR"
    llm["cost"] = [(i * PRICES.get(m, (0, 0))[0] + o * PRICES.get(m, (0, 0))[1]) / 1e6
                   for i, o, m in zip(llm["input"], llm["output"], llm["model"])]

    table = llm.groupby("agent").agg(calls=("agent", "size"), errors=("error", "sum"), input=("input", "sum"),
                                     output=("output", "sum"), cost=("cost", "sum"), seconds=("seconds", "sum"),
                                     error_seconds=("seconds", lambda s: s[llm.loc[s.index, "error"]].sum()))
    table = table.sort_values("cost", ascending=False)
    total_tokens = (table["input"] + table["output"]).sum()
    total_cost = table["cost"].sum()

    lines = [f"# Tokens y costo por agente · proyecto `{args.project}`", "",
             f"{len(llm)} llamadas al LLM · {int(total_tokens):,} tokens · costo estimado US$ {total_cost:.5f} "
             f"(precios de la tabla de Phoenix)".replace(",", "."), "",
             "| Agente | Llamadas | Fallidas (429/503) | Tokens entrada | Tokens salida | % de los tokens | Costo est. | % del costo | Segundos en el LLM | de ellos, en llamadas fallidas |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for agent, r in table.iterrows():
        share_tokens = 100 * (r["input"] + r["output"]) / max(total_tokens, 1)
        share_cost = 100 * r["cost"] / max(total_cost, 1e-12)
        lines.append(f"| {agent} | {int(r['calls'])} | {int(r['errors'])} | {int(r['input']):,} | {int(r['output']):,} | "
                     f"{share_tokens:.0f} % | US$ {r['cost']:.5f} | {share_cost:.0f} % | {r['seconds']:.1f} s | "
                     f"{r['error_seconds']:.1f} s |".replace(",", "."))
    by_model = llm.groupby("model").agg(calls=("model", "size"), errors=("error", "sum"))
    lines += ["", "Llamadas por modelo: " + ", ".join(f"`{m}` {int(r['calls'])} ({int(r['errors'])} fallidas)"
                                                       for m, r in by_model.iterrows())]
    text = "\n".join(lines) + "\n"
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"token_breakdown_{args.project}.md").write_text(text, encoding="utf-8")
    print(text + f"\nGuardado en results/token_breakdown_{args.project}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
