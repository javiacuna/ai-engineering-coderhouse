"""Offline chat model (LLM_PROVIDER=fake): runs the whole system without an API key.

It is NOT a language model: it follows fixed rules so that the API, Redis, the HITL flow and the traces in
Phoenix can be tried (and tested) without spending quota. Every role uses the same class, and it behaves
according to the tools bound to it:

- Supervisor (bound to the RouteDecision schema): reads the rubric the supervisor node writes in its prompt and
  picks the next agent the way the deterministic plan would.
- Researcher / analyst (bound to their tools): first turn → one tool call per tool and product mentioned in
  the instruction; next turn (after the tool results) → a short summary.
- Synthesizer (no tools): writes the final answer from the RESULTADOS block, using only those numbers.

It reports token usage (≈ 4 characters per token) so Phoenix can show tokens per node, and it waits a little
before answering, like a real API.
"""

import asyncio
import json
import os
import re
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool

from app.orchestrator.data.sources import products_mentioned

ROUTE_TOOL = "RouteDecision"


def count_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def content_of(message: BaseMessage) -> str:
    return message.content if isinstance(message.content, str) else json.dumps(message.content, ensure_ascii=False)


class FakeChatModel(BaseChatModel):
    """Rule-based stand-in for a chat model. See the module docstring."""

    model_name: str = "fake-llm"
    latency_seconds: float = 0.3
    tool_names: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "fake-chat"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name}

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "FakeChatModel":
        names = [convert_to_openai_tool(t)["function"]["name"] for t in tools]
        return self.model_copy(update={"tool_names": names})

    # ------------------------------------------------------------------ behaviour per role
    def _supervisor_turn(self, prompt: str) -> AIMessage:
        rubric = dict(re.findall(r"- (\w+): ([✓✗])", prompt))
        done = lambda key: rubric.get(key) == "✓"  # noqa: E731
        task = prompt.split("PEDIDO DEL USUARIO:\n", 1)[-1].split("\n\nPASO", 1)[0].strip()
        if done("final_answer_ready"):
            args = {"next_agent": "FINISH", "instruction": "", "reason": "la respuesta final está lista"}
        elif not done("research_complete"):
            args = {"next_agent": "researcher", "reason": "falta investigación",
                    "instruction": f"Buscá reseñas y precios de los productos de este pedido: {task}"}
        elif not (done("analysis_complete") and done("analysis_consistent")):
            args = {"next_agent": "analyst", "reason": "falta el análisis",
                    "instruction": f"Analizá sentimiento y estadísticas de los productos de este pedido: {task}"}
        else:
            args = {"next_agent": "synthesizer", "reason": "la rúbrica se cumple",
                    "instruction": f"Respondé el pedido del usuario: {task}"}
        return AIMessage(content="", tool_calls=[{"name": ROUTE_TOOL, "args": args, "id": "route-1"}])

    def _tool_turn(self, messages: list[BaseMessage]) -> AIMessage:
        last = messages[-1]
        if isinstance(last, ToolMessage):
            results = [m for m in messages if isinstance(m, ToolMessage)]
            products = sorted({p for m in results for p in products_mentioned(content_of(m))})
            failed = sum('"found": false' in content_of(m).lower() for m in results)
            text = (f"Usé {len(results)} herramientas sobre {', '.join(products) or 'ningún producto del catálogo'}."
                    + (f" {failed} búsquedas no devolvieron datos." if failed else ""))
            return AIMessage(content=text)
        instruction = next((content_of(m) for m in reversed(messages) if isinstance(m, HumanMessage)), "")
        products = products_mentioned(instruction)
        if not products:
            return AIMessage(content="No encontré productos del catálogo en el pedido.")
        calls = [{"name": tool, "args": {"product": product}, "id": f"{tool}-{product}"}
                 for product in products for tool in self.tool_names]
        return AIMessage(content="", tool_calls=calls)

    def _synthesis_turn(self, prompt: str) -> AIMessage:
        try:
            data = json.loads(prompt.split("RESULTADOS:\n", 1)[1])
        except (IndexError, json.JSONDecodeError):
            return AIMessage(content="No hay resultados para responder.")
        lines = []
        for p in data.get("analisis", []):
            price = f"${p['avg_price']:,.0f}".replace(",", ".") if p.get("avg_price") else "sin precio"
            lines.append(f"- {p['product']}: puntaje {p.get('avg_rating', '?')}, sentimiento "
                         f"{p.get('sentiment_score', '?')}, precio promedio {price}"
                         + (f", más barato en {p['best_store']}" if p.get("best_store") else "") + ".")
        for name in data.get("productos_no_encontrados", []):
            lines.append(f"- {name}: no está en las fuentes, no hay datos.")
        rated = [p for p in data.get("analisis", []) if p.get("sentiment_score") is not None]
        if rated:
            best = max(rated, key=lambda p: p["sentiment_score"])
            lines.append(f"Recomendación: {best['product']}, porque tiene el mejor sentimiento en las reseñas.")
        return AIMessage(content="\n".join(lines) or "No encontré datos de los productos pedidos.")

    def _respond(self, messages: list[BaseMessage]) -> AIMessage:
        prompt = content_of(messages[-1])
        if ROUTE_TOOL in self.tool_names:
            message = self._supervisor_turn(prompt)
        elif self.tool_names:
            message = self._tool_turn(messages)
        else:
            message = self._synthesis_turn(prompt)
        input_tokens = sum(count_tokens(content_of(m)) for m in messages)
        output_tokens = count_tokens(content_of(message) + json.dumps(message.tool_calls, ensure_ascii=False))
        message.usage_metadata = {"input_tokens": input_tokens, "output_tokens": output_tokens,
                                  "total_tokens": input_tokens + output_tokens}
        message.response_metadata = {"model_name": self.model_name}
        return message

    def _result(self, message: AIMessage) -> ChatResult:
        usage = message.usage_metadata or {}
        return ChatResult(generations=[ChatGeneration(message=message)],
                          llm_output={"model_name": self.model_name,
                                      "token_usage": {"prompt_tokens": usage.get("input_tokens", 0),
                                                      "completion_tokens": usage.get("output_tokens", 0),
                                                      "total_tokens": usage.get("total_tokens", 0)}})

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None,
                  run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any) -> ChatResult:
        return self._result(self._respond(messages))

    async def _agenerate(self, messages: list[BaseMessage], stop: list[str] | None = None,
                         run_manager: AsyncCallbackManagerForLLMRun | None = None, **kwargs: Any) -> ChatResult:
        await asyncio.sleep(self.latency_seconds)
        return self._result(self._respond(messages))


def fake_models(role: str) -> list[FakeChatModel]:
    """FAKE_LLM_MODEL_NAME lets the spans carry a real model name (Phoenix then prices them like that model)."""
    name = os.getenv("FAKE_LLM_MODEL_NAME", f"fake-{role}")
    latency = float(os.getenv("FAKE_LLM_LATENCY_SECONDS", "0.3"))
    from app.orchestrator.agents.llm import shared_rate_limiter

    return [FakeChatModel(model_name=name, latency_seconds=latency, rate_limiter=shared_rate_limiter())]
