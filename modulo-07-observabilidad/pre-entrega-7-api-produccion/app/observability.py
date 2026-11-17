"""Observability with Arize Phoenix (OpenTelemetry + OpenInference).

What gets traced, and how:
- Every node of the graph (risk, approvals, orchestrator, delivery), every agent of the orchestrator, every
  LLM call (model, tokens in/out, latency) and every tool → LangChainInstrumentor, automatically.
- The data sources (reviews and prices) → RETRIEVER spans made by hand: they are not LangChain objects.
- Each run of a job (the start and every resume after an approval) → a root span `job.start` / `job.resume`,
  with the job_id as session_id: in Phoenix → Sessions, all the runs of a job appear together.

Phoenix computes the cost of each LLM span from its model and tokens (built-in price list), and adds it up
per trace: that is the "cost per execution" of the load test.
"""

import functools
import json
import logging
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from typing import Any

from openinference.instrumentation import using_attributes
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from app.config import Settings

logger = logging.getLogger("app.observability")

PROMPT_TEMPLATE_VERSION = "orquestador-v1.3"   # on every span: a prompt change shows up in the traces
_tracer_provider = None


def setup_tracing(settings: Settings):
    """Registers the exporter to Phoenix and instruments LangChain + the data sources. Idempotent."""
    global _tracer_provider
    if not settings.phoenix_enabled:
        logger.info("tracing disabled (PHOENIX_ENABLED=false)")
        return None
    if _tracer_provider is not None:
        return _tracer_provider
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from phoenix.otel import register

    # batch=True: spans are exported in the background by a thread, never blocking the event loop
    _tracer_provider = register(project_name=settings.phoenix_project, endpoint=f"{settings.phoenix_endpoint}/v1/traces",
                                batch=True, verbose=False, set_global_tracer_provider=True)
    LangChainInstrumentor().instrument(tracer_provider=_tracer_provider)
    instrument_data_sources(_tracer_provider.get_tracer("data_sources"))
    logger.info("tracing to Phoenix at %s, project %s", settings.phoenix_endpoint, settings.phoenix_project)
    return _tracer_provider


def shutdown_tracing() -> None:
    if _tracer_provider is not None:
        _tracer_provider.force_flush()   # send the spans still in the batch before exiting


@contextmanager
def job_span(job_id: str, action: str, task: str) -> Iterator[trace.Span]:
    """Root span of one run of a job. Metadata and tags travel to every span inside it."""
    tracer = trace.get_tracer("app.worker")
    with using_attributes(session_id=job_id, tags=["pe7", action], metadata={"job_id": job_id, "action": action},
                          prompt_template_version=PROMPT_TEMPLATE_VERSION):
        with tracer.start_as_current_span(f"job.{action}") as span:
            span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
            span.set_attribute(SpanAttributes.INPUT_VALUE, task)
            yield span


def trace_id_of(span: trace.Span) -> str | None:
    context = span.get_span_context()
    return format(context.trace_id, "032x") if context.is_valid else None


def retriever_span(tracer: trace.Tracer, name: str, fetch: Callable[[str], Awaitable[list[dict[str, Any]]]]):
    """Wraps an async source query in an OpenInference RETRIEVER span (query, documents, errors)."""
    from openinference.instrumentation.langchain import get_current_span

    @functools.wraps(fetch)
    async def wrapper(product: str) -> list[dict[str, Any]]:
        # LangChain spans are created from callbacks and are not "current" in the OpenTelemetry context:
        # get_current_span() finds the tool span, so the DB span hangs from it (agent → tool → database)
        parent = get_current_span()
        context = trace.set_span_in_context(parent) if parent is not None else None
        with tracer.start_as_current_span(name, context=context) as span:
            span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.RETRIEVER.value)
            span.set_attribute(SpanAttributes.INPUT_VALUE, product)
            try:
                documents = await fetch(product)
            except Exception as e:   # recorded in the trace and re-raised: the tool turns it into data
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
    from app.orchestrator.data import sources

    for attr, span_name in (("fetch_reviews", "db.reviews.search"), ("fetch_prices", "db.prices.search")):
        original = getattr(sources, attr)
        original = getattr(original, "__wrapped_original__", original)   # idempotent
        setattr(sources, attr, retriever_span(tracer, span_name, original))
