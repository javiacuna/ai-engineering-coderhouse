"""LLM-as-a-judge faithfulness evaluation of each traced request, written back to Phoenix as annotations.

The numeric check of analyze_traces.py only catches invented NUMBERS. A sentence such as "the iPhone 17
has not been released yet" has no numbers, but it is not in any tool output either: the model took it
from its own knowledge. A judge LLM compares every claim of the answer with the data the agents
actually retrieved (the tool outputs of that trace) and lists the unsupported ones.

- The judge is a different model from the one that wrote the answers (GEMINI_JUDGE_MODEL, default
  gemini-3.8-flash vs. gemini-flash-lite-latest), to reduce self-preference bias.
- The verdict is stored in Phoenix as a "faithfulness" annotation on the request's root span
  (annotator_kind="LLM"), next to the numeric check ("numeric_grounding", annotator_kind="CODE").
- Judges have biases too (unit 7, error 3): this is an early-warning signal, not ground truth.

Usage:
    python evaluate_faithfulness.py                 # project from results/traffic_run.json
    python evaluate_faithfulness.py --no-annotate   # do not write annotations to Phoenix
"""

import argparse
import asyncio
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, Field

from analyze_traces import RESULTS_DIR, col, load_spans, tool_content

JUDGE_PROMPT = """\
Sos un evaluador de FIDELIDAD (faithfulness) de un sistema RAG multi-agente.

Recibís la PREGUNTA del usuario, el CONTEXTO (los resultados que devolvieron las herramientas, incluidos
los errores) y la RESPUESTA final. Tu tarea es decidir si cada afirmación de la RESPUESTA se deriva del
CONTEXTO.

Reglas:
- Una afirmación es "no respaldada" si no se puede deducir del CONTEXTO, aunque sea verdadera en el mundo
  real. Usar conocimiento propio del modelo cuenta como alucinación.
- Los redondeos razonables (4,17 → 4,2) y las conclusiones que se siguen directamente de los datos
  (por ejemplo, "es el más barato" si su precio es el menor) SÍ están respaldados.
- Decir que no hay datos está respaldado si el CONTEXTO muestra que no se encontraron o que hubo un error.
  Afirmar que faltan datos que el CONTEXTO sí tiene es una afirmación no respaldada.
- Las recomendaciones generales de cortesía ("te sugerimos comparar en tiendas") no cuentan como afirmaciones.
Respondé en español."""


class FaithfulnessVerdict(BaseModel):
    verdict: Literal["fiel", "alucinación"] = Field(description="'fiel' si todas las afirmaciones están respaldadas")
    unsupported_claims: list[str] = Field(default_factory=list, description="Afirmaciones de la respuesta sin respaldo en el contexto")
    explanation: str = Field(description="Justificación breve, en una o dos oraciones")


RETRYABLE = ("429", "RESOURCE_EXHAUSTED", "500", "502", "503", "504", "DEADLINE", "UNAVAILABLE", "INTERNAL", "timed out")


def build_judges() -> list[tuple[str, Any]]:
    """Main judge plus a fallback model (used only if the main one keeps failing)."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    models = [os.getenv("GEMINI_JUDGE_MODEL", "gemini-3.8-flash"),
              os.getenv("GEMINI_JUDGE_FALLBACK", "gemini-flash-lite-latest")]
    models = list(dict.fromkeys(models))   # no duplicates
    return [(m, ChatGoogleGenerativeAI(model=m, temperature=0, timeout=120, max_retries=1)
             .with_structured_output(FaithfulnessVerdict)) for m in models]


def is_retryable(error: Exception) -> bool:
    return isinstance(error, (asyncio.TimeoutError, TimeoutError)) or any(code in str(error) for code in RETRYABLE)


def trace_inputs(spans: pd.DataFrame) -> dict[str, Any]:
    """Question, retrieved context (tool outputs and errors) and final answer of one trace."""
    root = spans[spans["name"].str.startswith("consulta:")].iloc[0]
    tools = spans[spans["span_kind"] == "TOOL"].sort_values("start_time")
    context = []
    for _, tool in tools.iterrows():
        output = tool_content(tool.get("attributes.output.value"))
        if tool["status_code"] == "ERROR":
            output = f"ERROR: {str(tool.get('status_message') or '').splitlines()[0][:200]}"
        context.append(f"- {tool['name']}({tool_content(tool.get('attributes.input.value'))[:80]}) → {output[:3000]}")
    return {
        "span_id": root["context.span_id"],
        "scenario": root["name"].removeprefix("consulta:"),
        "question": str(root.get("attributes.input.value") or ""),
        "context": "\n".join(context) or "(ninguna herramienta devolvió resultados)",
        "answer": str(root.get("attributes.output.value") or ""),
    }


async def judge_trace(judges: list[tuple[str, Any]], item: dict[str, Any]) -> tuple[str, FaithfulnessVerdict]:
    """Up to 3 attempts per model on transient errors (429 rate limit, 5xx, timeouts), then the fallback model.
    Returns (model that answered, verdict). Raises the last error if every model failed."""
    message = (f"PREGUNTA:\n{item['question']}\n\nCONTEXTO:\n{item['context']}\n\nRESPUESTA:\n{item['answer']}")
    last_error: Exception | None = None
    for model, judge in judges:
        for attempt, wait in enumerate((15, 30, None), start=1):
            try:
                return model, await judge.ainvoke([("system", JUDGE_PROMPT), ("human", message)])
            except Exception as e:
                last_error = e
                if not is_retryable(e) or wait is None:
                    break
                print(f"    {model}: {str(e).splitlines()[0][:80]} → reintento {attempt} en {wait} s")
                await asyncio.sleep(wait)
        print(f"    {model} no respondió; paso al modelo de respaldo")
    raise last_error


def annotate(client, item: dict[str, Any], verdict: FaithfulnessVerdict, numeric: dict[str, Any] | None,
             judge_model: str) -> None:
    client.spans.add_span_annotation(
        span_id=item["span_id"], annotation_name="faithfulness", annotator_kind="LLM",
        label=verdict.verdict, score=1.0 if verdict.verdict == "fiel" else 0.0,
        explanation=verdict.explanation + (f" Sin respaldo: {verdict.unsupported_claims}" if verdict.unsupported_claims else ""),
        metadata={"judge_model": judge_model}, sync=True,
    )
    if numeric is not None:
        client.spans.add_span_annotation(
            span_id=item["span_id"], annotation_name="numeric_grounding", annotator_kind="CODE",
            label="ok" if not numeric["ungrounded_numbers"] else "números sin fuente",
            score=0.0 if numeric["ungrounded_numbers"] else 1.0,
            explanation=f"Números sin fuente en las herramientas: {numeric['ungrounded_numbers'] or 'ninguno'}", sync=True,
        )


async def main() -> int:
    parser = argparse.ArgumentParser(description="LLM-as-a-judge faithfulness evaluation")
    parser.add_argument("--project", default=None)
    parser.add_argument("--endpoint", default="http://localhost:6006")
    parser.add_argument("--no-annotate", action="store_true", help="do not write annotations to Phoenix")
    parser.add_argument("--pause", type=float, default=5, help="seconds between judge calls")
    args = parser.parse_args()

    load_dotenv(find_dotenv(usecwd=True))
    warnings.filterwarnings("ignore")
    project = args.project or json.loads((RESULTS_DIR / "traffic_run.json").read_text(encoding="utf-8"))["project"]
    spans = load_spans(project, args.endpoint)
    items = [trace_inputs(group) for _, group in spans.groupby("context.trace_id")
             if group["name"].str.startswith("consulta:").any()]
    items.sort(key=lambda i: i["scenario"])

    numeric_file = RESULTS_DIR / "trace_analysis.json"
    numeric = {t["scenario"]: t["groundedness"] for t in json.loads(numeric_file.read_text(encoding="utf-8"))["traces"]} \
        if numeric_file.exists() else {}

    judges = build_judges()
    judge_model = judges[0][0]
    client = None
    if not args.no_annotate:
        from phoenix.client import Client
        client = Client(base_url=args.endpoint)

    results = []
    for n, item in enumerate(items, start=1):
        num = numeric.get(item["scenario"], {})
        numeric_check = "ok" if not num.get("ungrounded_numbers") else f"sin fuente: {num['ungrounded_numbers']}"
        try:
            used_model, verdict = await judge_trace(judges, item)
        except Exception as e:   # one failed request must not stop the whole evaluation
            error = str(e).splitlines()[0][:150]
            results.append({"scenario": item["scenario"], "verdict": "sin evaluar", "unsupported_claims": [],
                            "explanation": f"El juez no respondió: {error}", "judge_model": None,
                            "numeric_check": numeric_check})
            print(f"[{n}/{len(items)}] {item['scenario']}: SIN EVALUAR ({error})")
            continue
        if client is not None:
            annotate(client, item, verdict, num or None, used_model)
        results.append({"scenario": item["scenario"], **verdict.model_dump(), "judge_model": used_model,
                        "numeric_check": numeric_check})
        print(f"[{n}/{len(items)}] {item['scenario']}: {verdict.verdict.upper()} ({used_model})"
              + (f" — sin respaldo: {verdict.unsupported_claims}" if verdict.unsupported_claims else ""))
        if n < len(items):
            await asyncio.sleep(args.pause)

    used = sorted({r["judge_model"] for r in results if r["judge_model"]}) or [judge_model]
    note = "" if used == [judge_model] else f" (el juez principal `{judge_model}` no respondió en todas las consultas)"
    lines = [f"# Evaluación de fidelidad (LLM-as-a-judge: {', '.join(f'`{m}`' for m in used)}){note}", "",
             "| Consulta | Veredicto del juez | Juez | Afirmaciones sin respaldo | Chequeo numérico | Explicación |",
             "|---|---|---|---|---|---|"]
    for r in results:
        claims = "; ".join(r["unsupported_claims"]).replace("|", "/") or "—"
        shown = {"fiel": "fiel", "alucinación": "**ALUCINACIÓN**"}.get(r["verdict"], r["verdict"])
        lines.append(f"| {r['scenario']} | {shown} | {r['judge_model'] or '—'} | {claims} | "
                     f"{r['numeric_check']} | {r['explanation'].replace('|', '/')} |")
    (RESULTS_DIR / "faithfulness_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RESULTS_DIR / "faithfulness_eval.json").write_text(
        json.dumps({"project": project, "judge_model": judge_model, "judge_models_used": used, "results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print("\n" + "\n".join(lines))
    print("\nGuardado en results/faithfulness_eval.md" + ("" if args.no_annotate else " · anotaciones cargadas en Phoenix"))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
