"""Synthesis agent: writes the final answer for the user from the validated artifacts.

It receives the user's task and the STRUCTURED results (not the agents' conversations), and has no tools:
it cannot search or compute, only explain what the specialists already established.
"""

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.llm import with_fallback
from state import Contribution, OrchestratorState, rounds_of

logger = logging.getLogger("orchestrator.synthesizer")

SYNTHESIS_PROMPT = """\
Sos el agente de SÍNTESIS. Redactás la respuesta final para el usuario, en español, a partir de los
resultados ya validados del equipo.

Reglas:
1. Usá SOLO los datos del bloque RESULTADOS. No inventes números ni productos.
2. Respondé exactamente lo que pidió el usuario, en el orden en que lo pidió.
3. Mostrá los montos en pesos con separador de miles (por ejemplo $182.000) y los puntajes con un decimal.
4. Si falta información de algún producto, decilo explícitamente en lugar de suponer.
5. Si hay una tensión en los datos (por ejemplo, el más barato es el peor valorado), explicala.
6. Máximo 12 líneas. Cerrá con una recomendación justificada en una oración."""


def results_block(state: OrchestratorState) -> str:
    """Compact, structured context for the synthesis (no raw reviews, no agent chatter)."""
    research, analysis = state.get("research"), state.get("analysis")
    payload = {
        "productos_no_encontrados": [p.product for p in research.products if not p.found] if research else [],
        "analisis": [p.model_dump(exclude_none=True) for p in analysis.products] if analysis else [],
        "interpretacion_del_analista": analysis.summary if analysis else "",
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def make_synthesis_node(models: list[Any]):
    async def synthesis_node(state: OrchestratorState) -> dict[str, Any]:
        round_number = rounds_of(state, "synthesizer") + 1
        instruction = state["instruction"]
        messages = [SystemMessage(content=SYNTHESIS_PROMPT),
                    HumanMessage(content=f"PEDIDO DEL USUARIO:\n{state['task']}\n\nINDICACIÓN DEL SUPERVISOR:\n"
                                         f"{instruction}\n\nRESULTADOS:\n{results_block(state)}")]
        try:
            response = await with_fallback(models, lambda m: m.ainvoke(messages), "synthesizer")
            answer, error = response.text.strip(), None
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)[:200]}"
            logger.error("synthesizer failed: %s", error)
            answer = "No se pudo generar la respuesta final. Datos disponibles:\n" + results_block(state)

        contribution = Contribution(agent="synthesizer", round=round_number, instruction=instruction,
                                    summary=answer[:300], error=error)
        return {"final_answer": answer, "contributions": [contribution],
                "messages": [AIMessage(content=answer, name="synthesizer")]}

    return synthesis_node
