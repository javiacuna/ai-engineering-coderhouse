"""Phoenix instrumentation for the Module 6 orchestrator (pre-entrega 6).

- LangChain/LangGraph: OpenInference's LangChainInstrumentor creates a span for every graph node, agent,
  LLM call and tool call, with inputs, outputs, latency, token counts and errors.
- Data sources: the "database" calls (reviews and prices) are not LangChain objects, so they would be a
  blind spot. They are wrapped by hand in RETRIEVER spans, with the returned documents as attributes.
"""

import functools
import json
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable

from openinference.instrumentation.langchain import LangChainInstrumentor, get_current_span
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

REPO_DIR = Path(__file__).resolve().parents[2]
ORCHESTRATOR_DIR = REPO_DIR / "modulo-06-sistemas-multiagente" / "pre-entrega-6-orquestador-multiagente"
DEFAULT_ENDPOINT = "http://localhost:6006"
PROMPT_TEMPLATE_VERSION = "orquestador-v1.2"   # logged on every span: a prompt change shows up in the traces

if str(ORCHESTRATOR_DIR) not in sys.path:
    sys.path.insert(0, str(ORCHESTRATOR_DIR))   # import the orchestrator (graph, agents, data) unchanged


def setup_tracing(project_name: str, endpoint: str = DEFAULT_ENDPOINT):
    """Registers the OTel exporter to the local Phoenix and instruments LangChain + the data sources."""
    from phoenix.otel import register

    tracer_provider = register(project_name=project_name, endpoint=f"{endpoint}/v1/traces",
                               batch=True, verbose=False)
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    instrument_data_sources(tracer_provider.get_tracer("data_sources"))
    return tracer_provider


def retriever_span(tracer: trace.Tracer, name: str, fetch: Callable[[str], Awaitable[list[dict[str, Any]]]]):
    """Wraps an async source query in an OpenInference RETRIEVER span (query, documents, errors)."""

    @functools.wraps(fetch)
    async def wrapper(product: str) -> list[dict[str, Any]]:
        # The LangChain instrumentor creates its spans from callbacks and does not make them "current" in the
        # OpenTelemetry context: without this, the DB span would hang from the request root instead of from
        # the tool that called it (agent -> tool -> database)
        parent = get_current_span()
        context = trace.set_span_in_context(parent) if parent is not None else None
        with tracer.start_as_current_span(name, context=context) as span:
            span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.RETRIEVER.value)
            span.set_attribute(SpanAttributes.INPUT_VALUE, product)
            try:
                documents = await fetch(product)
            except Exception as e:   # recorded in the trace, then re-raised: the agent decides what to do
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
            for i, doc in enumerate(documents):
                prefix = f"retrieval.documents.{i}.document"
                span.set_attribute(f"{prefix}.content", doc.get("text") or json.dumps(doc, ensure_ascii=False))
                span.set_attribute(f"{prefix}.metadata", json.dumps(doc, ensure_ascii=False))
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, f"{len(documents)} documentos")
            span.set_status(Status(StatusCode.OK))
            return documents

    wrapper.__wrapped_original__ = fetch
    return wrapper


def instrument_data_sources(tracer: trace.Tracer) -> None:
    """Replaces the source functions the research tools call (sources.fetch_reviews / fetch_prices)."""
    from data import sources

    for attr, span_name in (("fetch_reviews", "db.reviews.search"), ("fetch_prices", "db.prices.search")):
        original = getattr(sources, attr)
        original = getattr(original, "__wrapped_original__", original)   # idempotent
        setattr(sources, attr, retriever_span(tracer, span_name, original))
