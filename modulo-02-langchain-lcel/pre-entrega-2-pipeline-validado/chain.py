"""LCEL pipeline: prompt | model.with_structured_output(TechnicalExtraction), with validation and retries."""

import logging
import os
from enum import Enum
from typing import Any, Optional

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from schemas import TechnicalExtraction

logger = logging.getLogger("extraction_pipeline")


# ---------------------------------------------------------------------------
# Configuration (same idea as the Module 1 client: provider chosen by env vars, keys as SecretStr)
# ---------------------------------------------------------------------------
class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


DEFAULT_MODELS = {
    Provider.OPENAI: "gpt-4o-mini",
    Provider.ANTHROPIC: "claude-haiku-4-5",
    Provider.GEMINI: "gemini-3.8-flash",
}

KEY_ENV_VARS = {
    Provider.OPENAI: ("OPENAI_API_KEY",),
    Provider.ANTHROPIC: ("ANTHROPIC_API_KEY",),
    Provider.GEMINI: ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
}


class PipelineConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: Provider
    model: str
    fallback_model: Optional[str] = None
    api_key: SecretStr
    max_tokens: int = Field(default=1024, gt=0)
    max_attempts: int = Field(default=3, ge=1, le=5, description="Total attempts, including the first one")
    retry_backoff: bool = True

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        provider = Provider(os.getenv("LLM_PROVIDER", "gemini").lower())
        api_key = next((os.getenv(v) for v in KEY_ENV_VARS[provider] if os.getenv(v)), None)
        if not api_key:
            raise ValueError(f"Missing {' or '.join(KEY_ENV_VARS[provider])} for provider '{provider.value}'")
        # With Gemini, the GEMINI_MODEL / GEMINI_FALLBACK_MODEL variables used in Module 2 are also accepted
        is_gemini = provider == Provider.GEMINI
        return cls(
            provider=provider,
            model=os.getenv("LLM_MODEL") or (is_gemini and os.getenv("GEMINI_MODEL")) or DEFAULT_MODELS[provider],
            fallback_model=os.getenv("LLM_FALLBACK_MODEL") or (is_gemini and os.getenv("GEMINI_FALLBACK_MODEL")) or None,
            api_key=api_key,
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", 1024)),
            max_attempts=int(os.getenv("LLM_MAX_ATTEMPTS", 3)),
        )


def build_chat_model(config: PipelineConfig, model: Optional[str] = None) -> BaseChatModel:
    """LangChain chat model for the configured provider."""
    name = model or config.model
    key = config.api_key.get_secret_value()
    if config.provider == Provider.OPENAI:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=name, api_key=key, temperature=0, max_tokens=config.max_tokens)
    if config.provider == Provider.ANTHROPIC:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=name, api_key=key, max_tokens=config.max_tokens)
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=name, google_api_key=key, temperature=0, max_output_tokens=config.max_tokens)


# ---------------------------------------------------------------------------
# Prompt: modular template with two inputs, the raw text and the format instructions
# ---------------------------------------------------------------------------
FORMAT_INSTRUCTIONS = """\
- tecnologias: every concrete technology mentioned (languages, frameworks, databases, queues, cloud services, tools), with its official name. Do not invent technologies that are not in the text. If the text names no concrete technology, list the generic components it does mention, in Spanish (e.g. "base de datos", "servidor"). Never use placeholders such as "No especificada" or "N/A".
- nivel_de_criticidad: "alta" if there is an outage, data loss, a security issue or blocked users; "media" if performance is degraded or there are partial errors; "baja" if it is informative with no visible impact.
- resumen_tecnico: one or two sentences in Spanish describing the architecture or the problem and its most likely cause."""

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a senior software engineer who analyzes architecture descriptions and error logs. "
     "Extract the technical information from the text and return it with this structure:\n{format_instructions}"),
    ("human", "Text to analyze:\n<text>\n{text}\n</text>"),
]).partial(format_instructions=FORMAT_INSTRUCTIONS)


# ---------------------------------------------------------------------------
# Output validation: detects truncated or malformed answers BEFORE using them
# ---------------------------------------------------------------------------
class IncompleteOutputError(Exception):
    """The model answer is truncated, malformed or does not match the schema. Triggers a retry."""


TRUNCATION_REASONS = {"length", "max_tokens"}   # OpenAI: "length" · Anthropic: "max_tokens" · Gemini: "MAX_TOKENS"


def finish_reason_of(raw: Any) -> Optional[str]:
    metadata = getattr(raw, "response_metadata", None) or {}
    reason = metadata.get("finish_reason") or metadata.get("stop_reason")
    return str(reason).lower() if reason is not None else None


def check_output(result: dict) -> TechnicalExtraction:
    """Receives {"raw", "parsed", "parsing_error"} from with_structured_output(include_raw=True)."""
    reason = finish_reason_of(result.get("raw"))
    if reason in TRUNCATION_REASONS:
        logger.warning("Truncated answer (finish_reason=%s): discarded, retrying", reason)
        raise IncompleteOutputError(f"answer cut off by the token limit (finish_reason={reason})")
    if result.get("parsing_error") is not None:
        logger.warning("Malformed or invalid answer: %s", str(result["parsing_error"]).splitlines()[0])
        raise IncompleteOutputError(f"parsing error: {result['parsing_error']}")
    parsed = result.get("parsed")
    if parsed is None:
        logger.warning("The model did not return a structured object (finish_reason=%s)", reason)
        raise IncompleteOutputError("empty structured output")
    logger.info("Valid output (finish_reason=%s, %d technologies)", reason, len(parsed.technologies))
    return parsed


RETRYABLE_ERRORS = (IncompleteOutputError, ValidationError, OutputParserException)


def build_chain(config: PipelineConfig, model: Optional[BaseChatModel] = None) -> Runnable:
    """prompt | model.with_structured_output(TechnicalExtraction) | check_output, with retries and optional fallback."""

    def structured(llm: BaseChatModel):
        kwargs = {"include_raw": True}
        if config.provider == Provider.OPENAI:
            kwargs["method"] = "function_calling"   # supports schema constraints such as minItems
        return llm.with_structured_output(TechnicalExtraction, **kwargs)

    def pipeline(llm: BaseChatModel) -> Runnable:
        return (PROMPT | structured(llm) | RunnableLambda(check_output)).with_retry(
            retry_if_exception_type=RETRYABLE_ERRORS,
            stop_after_attempt=config.max_attempts,
            wait_exponential_jitter=config.retry_backoff,
        )

    chain = pipeline(model or build_chat_model(config))
    if config.fallback_model and model is None:
        chain = chain.with_fallbacks([pipeline(build_chat_model(config, config.fallback_model))])
    return chain


class ExtractionError(Exception):
    """The pipeline could not produce a valid object after all retries."""


async def process_text(text: str, chain: Optional[Runnable] = None) -> TechnicalExtraction:
    """Runs the pipeline on a raw text and returns a validated TechnicalExtraction."""
    if not text or not text.strip():
        raise ExtractionError("the input text is empty")
    chain = chain or build_chain(PipelineConfig.from_env())
    logger.info("Processing text (%d characters)", len(text))
    try:
        result: TechnicalExtraction = await chain.ainvoke({"text": text.strip()})
    except RETRYABLE_ERRORS as e:
        logger.error("No valid output after all attempts: %s", str(e).splitlines()[0])
        raise ExtractionError(str(e).splitlines()[0]) from e
    except Exception as e:  # provider errors (auth, quota, network) after the SDK's own retries
        logger.error("Provider error: %s: %s", type(e).__name__, str(e)[:200])
        raise ExtractionError(f"{type(e).__name__}: {str(e)[:200]}") from e
    logger.info("Extraction OK: criticality=%s, technologies=%s", result.criticality.value, result.technologies)
    return result
