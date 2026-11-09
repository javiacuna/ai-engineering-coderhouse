"""Analyst agent: processes the researcher's data (sentiment analysis and statistics).

Its tools work ONLY on the research already in the state (they are built for each turn with that data):
the analyst cannot search sources, and its numbers come from deterministic computations, not from the LLM.
"""

import logging
import re
import statistics
import unicodedata
from collections import Counter
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from agents.llm import final_text, tool_results, with_fallback
from state import (AnalysisResult, Contribution, OrchestratorState, ProductAnalysis, ProductFindings,
                   ResearchFindings, rounds_of)

logger = logging.getLogger("orchestrator.analyst")

AGENT_RECURSION_LIMIT = 16

ANALYST_PROMPT = """\
Sos el agente de ANÁLISIS de un equipo de análisis de productos. Procesás los datos que ya juntó el
investigador; no buscás información nueva.

- Para CADA producto disponible, usá analyze_sentiment y compute_statistics.
- Usá SOLO los números que devuelven las herramientas: no los estimes ni los redondees de otra forma.
- Terminá con una interpretación breve en español: qué producto tiene mejor percepción, cuál es más
  barato, los principales puntos a favor y en contra de cada uno, y cualquier contradicción que veas
  (por ejemplo, buen puntaje pero muchas quejas sobre un aspecto)."""

# ---------------------------------------------------------------------------
# Sentiment analysis (lexicon-based, Spanish, with negation)
# ---------------------------------------------------------------------------
POSITIVE = {"excelente", "increible", "buen", "buena", "bueno", "buenos", "buenas", "recomiendo", "comodos",
            "comodo", "livianos", "liviano", "potente", "resistente", "rapida", "estable", "economicos",
            "baratos", "ideales", "ideal", "correcto", "bien", "muchisimo", "genial"}
NEGATIVE = {"poco", "molesto", "mala", "malo", "mal", "rompio", "incomodos", "duelen", "debil", "caros",
            "pesado", "desconectan", "grande", "lento", "falla"}
NEGATIONS = {"no", "nunca", "ni"}

ASPECTS = {
    "sonido": {"sonido", "audio", "suenan", "suena", "graves", "volumen"},
    "batería": {"bateria"},
    "comodidad": {"comodos", "comodo", "incomodos", "duelen", "livianos", "liviano", "pesado"},
    "cancelación de ruido": {"cancelacion"},
    "conexión": {"conexion", "desconectan"},
    "precio": {"caros", "baratos", "economicos", "precio"},
    "durabilidad": {"rompio", "materiales", "resistente", "bisagra"},
}


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.findall(r"[a-zñ]+", text)


def review_polarity(text: str) -> float:
    """Score from -1 to 1. A negation flips the next two words ("no están mal" is positive)."""
    tokens = tokenize(text)
    score, hits = 0, 0
    for i, token in enumerate(tokens):
        value = 1 if token in POSITIVE else -1 if token in NEGATIVE else 0
        if value == 0 or (token == "poco" and i > 0 and tokens[i - 1] == "un"):   # "un poco caros": a modifier
            continue
        if any(t in NEGATIONS for t in tokens[max(0, i - 2):i]):
            value = -value
        score += value
        hits += 1
    return score / hits if hits else 0.0


def sentiment_report(findings: ProductFindings) -> dict[str, Any]:
    polarities = [review_polarity(r.text) for r in findings.reviews]
    pros, cons = Counter(), Counter()
    for review, polarity in zip(findings.reviews, polarities):
        tokens = set(tokenize(review.text))
        for aspect, words in ASPECTS.items():
            if tokens & words:
                (pros if polarity > 0 else cons if polarity < 0 else Counter())[aspect] += 1
    n = len(polarities)
    return {
        "product": findings.product,
        "review_count": n,
        "sentiment_score": round(sum(polarities) / n, 2) if n else None,
        "positive_pct": round(100 * sum(p > 0 for p in polarities) / n, 1) if n else None,
        "negative_pct": round(100 * sum(p < 0 for p in polarities) / n, 1) if n else None,
        "pros": [a for a, _ in pros.most_common(3)],
        "cons": [a for a, _ in cons.most_common(3)],
    }


def statistics_report(findings: ProductFindings) -> dict[str, Any]:
    ratings = [r.rating for r in findings.reviews]
    prices = [p.price for p in findings.prices]
    cheapest = min(findings.prices, key=lambda p: p.price) if findings.prices else None
    return {
        "product": findings.product,
        "review_count": len(ratings),
        "avg_rating": round(statistics.mean(ratings), 2) if ratings else None,
        "rating_distribution": {str(k): ratings.count(k) for k in range(5, 0, -1)},
        "avg_price": round(statistics.mean(prices), 2) if prices else None,
        "min_price": cheapest.price if cheapest else None,
        "best_store": cheapest.store if cheapest else None,
    }


class ProductInput(BaseModel):
    product: str = Field(..., description="Nombre exacto del producto, tal como figura en los datos disponibles.")


def make_analyst_tools(research: ResearchFindings) -> list[BaseTool]:
    """Tools bound to THIS research: the analyst can only work on the data the researcher found."""
    available = {p.product: p for p in research.products if p.found}

    def lookup(product: str) -> ProductFindings | dict[str, Any]:
        match = next((p for name, p in available.items() if name.lower() == product.strip().lower()), None)
        return match or {"error": f"No hay datos de '{product}'.", "productos_disponibles": list(available)}

    @tool(args_schema=ProductInput)
    def analyze_sentiment(product: str) -> dict[str, Any]:
        """Analiza el sentimiento de las reseñas de un producto: puntaje de -1 (muy negativo) a 1 (muy
        positivo), porcentaje de reseñas positivas y negativas, y los aspectos más elogiados (pros) y
        más criticados (contras), como batería, sonido o comodidad."""
        findings = lookup(product)
        return findings if isinstance(findings, dict) else sentiment_report(findings)

    @tool(args_schema=ProductInput)
    def compute_statistics(product: str) -> dict[str, Any]:
        """Calcula estadísticas de un producto: cantidad de reseñas, puntaje promedio (1 a 5),
        distribución de puntajes, precio promedio, precio mínimo y la tienda más barata."""
        findings = lookup(product)
        return findings if isinstance(findings, dict) else statistics_report(findings)

    return [analyze_sentiment, compute_statistics]


def build_analysis(previous: AnalysisResult | None, results: list[tuple[str, dict, Any]], summary: str) -> AnalysisResult:
    """Artifact built from the TOOL RESULTS (the numbers can't be made up), merged with a previous round."""
    products: dict[str, ProductAnalysis] = {p.product: p for p in previous.products} if previous else {}
    for name, _, result in results:
        if not isinstance(result, dict) or "error" in result or "product" not in result:
            continue
        current = products.get(result["product"]) or ProductAnalysis(product=result["product"], review_count=0)
        if name == "analyze_sentiment":
            fields = ("review_count", "sentiment_score", "positive_pct", "negative_pct", "pros", "cons")
        else:
            fields = ("review_count", "avg_rating", "avg_price", "min_price", "best_store")
        products[result["product"]] = current.model_copy(update={f: result[f] for f in fields})
    return AnalysisResult(products=list(products.values()), summary=summary or (previous.summary if previous else ""))


def make_analyst_node(models: list[Any]):
    async def analyst_node(state: OrchestratorState) -> dict[str, Any]:
        round_number = rounds_of(state, "analyst") + 1
        instruction = state["instruction"]
        research = state.get("research")
        if research is None or not any(p.found for p in research.products):
            message = "No hay datos de investigación para analizar."
            contribution = Contribution(agent="analyst", round=round_number, instruction=instruction,
                                        summary=message, error="missing research")
            return {"contributions": [contribution], "messages": [AIMessage(content=message, name="analyst")]}

        tools = make_analyst_tools(research)
        available = [p.product for p in research.products if p.found]
        # Context isolation: the instruction and the list of products; the data itself is reached through the tools
        context = f"{instruction}\n\nProductos con datos disponibles: {', '.join(available)}."

        async def run(model):
            agent = create_react_agent(model, tools, prompt=ANALYST_PROMPT)
            out = await agent.ainvoke({"messages": [HumanMessage(content=context)]},
                                      {"recursion_limit": AGENT_RECURSION_LIMIT})
            return out["messages"]

        try:
            transcript = await with_fallback(models, run, "analyst")
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)[:200]}"
            logger.error("analyst failed: %s", error)
            contribution = Contribution(agent="analyst", round=round_number, instruction=instruction,
                                        summary="El análisis falló.", error=error)
            return {"contributions": [contribution],
                    "messages": [AIMessage(content=f"[analyst] Error: {error}", name="analyst")]}

        results = tool_results(transcript)
        summary = final_text(transcript)
        analysis = build_analysis(state.get("analysis"), results, summary)
        contribution = Contribution(
            agent="analyst", round=round_number, instruction=instruction, summary=summary,
            tools_used=[f"{name}({args.get('product', '')})" for name, args, _ in results],
        )
        return {"analysis": analysis, "contributions": [contribution],
                "messages": [AIMessage(content=summary, name="analyst")]}

    return analyst_node
