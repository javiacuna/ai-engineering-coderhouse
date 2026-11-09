"""Research agent: searches external sources (simulated) for product reviews and prices.

Its tools are scoped to READING sources; it never computes or concludes. The artifact it leaves in the
state (ResearchFindings) is built from the TOOL RESULTS, not from the LLM's text, so data can't be invented.
"""

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from agents.llm import final_text, tool_results, with_fallback
from data import sources
from state import Contribution, OrchestratorState, PriceQuote, ProductFindings, ResearchFindings, Review, rounds_of

logger = logging.getLogger("orchestrator.researcher")

AGENT_RECURSION_LIMIT = 12   # steps of the agent's own ReAct loop

RESEARCH_PROMPT = """\
Sos el agente de INVESTIGACIÓN de un equipo de análisis de productos. Tu único trabajo es buscar
información en las fuentes: reseñas de clientes y precios en tiendas. No calculás promedios ni das
opiniones: eso lo hace el analista.

- Usá search_reviews y search_prices para CADA producto que te pidan.
- Si una búsqueda devuelve found=false, probá con las sugerencias que trae; si no hay, informalo.
- Terminá con un resumen breve en español: qué productos encontraste, cuántas reseñas y cuántos precios
  de cada uno, y cuáles no encontraste."""


class ProductInput(BaseModel):
    product: str = Field(..., min_length=2, description="Nombre del producto, por ejemplo 'SoundMax Pro'.")


@tool(args_schema=ProductInput)
async def search_reviews(product: str) -> dict[str, Any]:
    """Busca reseñas de clientes de un producto en la tienda, foros y blogs.
    Devuelve las reseñas (puntaje de 1 a 5, texto, fuente y fecha). Si el producto no existe,
    devuelve found=false con sugerencias de nombres parecidos."""
    official = sources.resolve_product(product)
    if official is None:
        return {"product": product, "found": False, "suggestions": sources.suggestions(product),
                "error": f"No se encontraron reseñas para '{product}'."}
    reviews = await sources.fetch_reviews(official)
    return {"product": official, "found": True, "category": sources.PRODUCTS[official]["category"],
            "review_count": len(reviews), "reviews": reviews}


@tool(args_schema=ProductInput)
async def search_prices(product: str) -> dict[str, Any]:
    """Busca el precio actual de un producto en distintas tiendas online (en pesos argentinos).
    Si el producto no existe, devuelve found=false con sugerencias de nombres parecidos."""
    official = sources.resolve_product(product)
    if official is None:
        return {"product": product, "found": False, "suggestions": sources.suggestions(product),
                "error": f"No se encontraron precios para '{product}'."}
    return {"product": official, "found": True, "prices": await sources.fetch_prices(official)}


RESEARCH_TOOLS: list[BaseTool] = [search_reviews, search_prices]


def merge_findings(previous: ResearchFindings | None, results: list[tuple[str, dict, Any]], summary: str) -> ResearchFindings:
    """Builds the artifact from the tool results, on top of a previous round (refinements add, never lose data)."""
    products: dict[str, ProductFindings] = {p.product: p for p in previous.products} if previous else {}
    for name, args, result in results:
        if not isinstance(result, dict):
            continue
        key = result.get("product") or args.get("product", "?")
        current = products.get(key) or ProductFindings(product=key, found=False)
        if result.get("found"):
            current.found = True
            current.suggestions = []
            current.category = result.get("category", current.category)
            if name == "search_reviews":
                current.reviews = [Review(**r) for r in result.get("reviews", [])]
            elif name == "search_prices":
                current.prices = [PriceQuote(**p) for p in result.get("prices", [])]
        elif not current.found:
            current.suggestions = result.get("suggestions", [])
        products[key] = current
    # A product "not found" under a name that later resolved to an official one is noise: drop it
    found_names = {p for p, f in products.items() if f.found}
    cleaned = [f for f in products.values() if f.found or not (set(f.suggestions) & found_names)]
    return ResearchFindings(products=cleaned, summary=summary or (previous.summary if previous else ""))


def make_research_node(models: list[Any]):
    async def research_node(state: OrchestratorState) -> dict[str, Any]:
        round_number = rounds_of(state, "researcher") + 1
        instruction = state["instruction"]
        # Context isolation: the researcher receives only its instruction (+ what is already found,
        # on a refinement), not the whole conversation nor the other agents' metadata
        context = instruction
        if state.get("research"):
            have = ", ".join(f"{p.product} ({len(p.reviews)} reseñas)" for p in state["research"].products if p.found)
            context += f"\n\nYa tenés datos de: {have or 'ningún producto'}."

        async def run(model):
            agent = create_react_agent(model, RESEARCH_TOOLS, prompt=RESEARCH_PROMPT)
            out = await agent.ainvoke({"messages": [HumanMessage(content=context)]},
                                      {"recursion_limit": AGENT_RECURSION_LIMIT})
            return out["messages"]

        try:
            transcript = await with_fallback(models, run, "researcher")
        except Exception as e:   # the graph goes on: the supervisor sees the error and decides
            error = f"{type(e).__name__}: {str(e)[:200]}"
            logger.error("researcher failed: %s", error)
            contribution = Contribution(agent="researcher", round=round_number, instruction=instruction,
                                        summary="La investigación falló.", error=error)
            return {"contributions": [contribution],
                    "messages": [AIMessage(content=f"[researcher] Error: {error}", name="researcher")]}

        results = tool_results(transcript)
        summary = final_text(transcript)
        research = merge_findings(state.get("research"), results, summary)
        contribution = Contribution(
            agent="researcher", round=round_number, instruction=instruction, summary=summary,
            tools_used=[f"{name}({args.get('product', '')})" for name, args, _ in results],
        )
        return {"research": research, "contributions": [contribution],
                "messages": [AIMessage(content=summary, name="researcher")]}

    return research_node
