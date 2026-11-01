"""Tests with a fake model: no API keys or network needed."""

import asyncio
import json

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from chain import (ExtractionError, PipelineConfig, Provider, build_chain, check_output,
                   IncompleteOutputError, PROMPT, process_text)
from schemas import Criticality, TechnicalExtraction

VALID = {"tecnologias": ["FastAPI", "Redis", "PostgreSQL"], "nivel_de_criticidad": "alta",
         "resumen_tecnico": "API con caché en Redis; el pool de conexiones de PostgreSQL se agota."}


def ok(payload=VALID, reason="stop"):
    return {"raw": AIMessage(content="", response_metadata={"finish_reason": reason}),
            "parsed": TechnicalExtraction.model_validate(payload), "parsing_error": None}


def truncated():
    return {"raw": AIMessage(content='{"tecnologias": ["Fast', response_metadata={"finish_reason": "length"}),
            "parsed": None, "parsing_error": None}


def malformed():
    return {"raw": AIMessage(content="not json", response_metadata={"finish_reason": "stop"}),
            "parsed": None, "parsing_error": ValueError("Invalid JSON")}


class FakeModel:
    """Mimics model.with_structured_output(..., include_raw=True): returns the queued answers in order."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = 0

    def with_structured_output(self, schema, **kwargs):
        assert schema is TechnicalExtraction and kwargs.get("include_raw") is True

        def answer(_prompt_value):
            self.calls += 1
            return self.answers.pop(0)
        return RunnableLambda(answer)


def config(attempts=3):
    return PipelineConfig(provider=Provider.GEMINI, model="fake", api_key="x",
                          max_attempts=attempts, retry_backoff=False)


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------- schema
def test_contract_json_uses_spanish_keys():
    data = json.loads(TechnicalExtraction.model_validate(VALID).to_contract_json())
    assert set(data) == {"tecnologias", "nivel_de_criticidad", "resumen_tecnico"}
    assert data["nivel_de_criticidad"] == "alta"


def test_technologies_are_cleaned_and_deduplicated():
    t = TechnicalExtraction.model_validate({**VALID, "tecnologias": [" Redis ", "redis", "", "Kafka"]})
    assert t.technologies == ["Redis", "Kafka"]


@pytest.mark.parametrize("change", [
    {"tecnologias": []},
    {"tecnologias": ["  ", ""]},
    {"nivel_de_criticidad": "critica"},
    {"resumen_tecnico": "corto"},
])
def test_invalid_payloads_are_rejected(change):
    with pytest.raises(ValidationError):
        TechnicalExtraction.model_validate({**VALID, **change})


def test_criticality_enum_values():
    assert [c.value for c in Criticality] == ["baja", "media", "alta"]


# ---------------------------------------------------------------- prompt
def test_prompt_is_modular():
    assert PROMPT.input_variables == ["text"]          # format_instructions is already filled in with .partial()
    messages = PROMPT.invoke({"text": "log de prueba"}).to_messages()
    assert "nivel_de_criticidad" in messages[0].content and "log de prueba" in messages[1].content


# ---------------------------------------------------------------- output checks
def test_truncated_output_is_detected():
    with pytest.raises(IncompleteOutputError, match="finish_reason=length"):
        check_output(truncated())


@pytest.mark.parametrize("reason", ["max_tokens", "MAX_TOKENS"])
def test_truncation_detected_for_anthropic_and_gemini(reason):
    with pytest.raises(IncompleteOutputError):
        check_output({"raw": AIMessage(content="", response_metadata={"stop_reason": reason}),
                      "parsed": None, "parsing_error": None})


# ---------------------------------------------------------------- chain + retries
def test_valid_answer_on_first_attempt():
    model = FakeModel([ok()])
    result = run(process_text("FastAPI con Redis y PostgreSQL", build_chain(config(), model)))
    assert result.criticality == Criticality.HIGH and model.calls == 1


def test_retries_after_truncated_and_malformed_answers():
    model = FakeModel([truncated(), malformed(), ok()])
    result = run(process_text("texto", build_chain(config(attempts=3), model)))
    assert result.technologies == ["FastAPI", "Redis", "PostgreSQL"] and model.calls == 3


def test_gives_up_after_max_attempts():
    model = FakeModel([malformed(), malformed(), malformed()])
    with pytest.raises(ExtractionError, match="parsing error"):
        run(process_text("texto", build_chain(config(attempts=2), model)))
    assert model.calls == 2


def test_empty_text_is_rejected_without_calling_the_model():
    model = FakeModel([])
    with pytest.raises(ExtractionError, match="empty"):
        run(process_text("   ", build_chain(config(), model)))
    assert model.calls == 0


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", "g-x")
    monkeypatch.setenv("LLM_FALLBACK_MODEL", "gemini-flash-lite-latest")
    c = PipelineConfig.from_env()
    assert c.provider == Provider.GEMINI and c.fallback_model == "gemini-flash-lite-latest"
    assert "g-x" not in repr(c)


def test_config_requires_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        PipelineConfig.from_env()


def test_gemini_reuses_module_2_variables(monkeypatch):
    for var in ("LLM_MODEL", "LLM_FALLBACK_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", "g-x")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "gemini-flash-lite-latest")
    c = PipelineConfig.from_env()
    assert c.model == "gemini-3.8-flash" and c.fallback_model == "gemini-flash-lite-latest"


@pytest.mark.parametrize("placeholder", ["No especificada", "N/A", "desconocido"])
def test_placeholders_are_rejected(placeholder):
    # Found in the stress test: with an ambiguous text the model returned ["No especificada"]
    with pytest.raises(ValidationError, match="placeholder"):
        TechnicalExtraction.model_validate({**VALID, "tecnologias": [placeholder]})


def test_placeholder_answer_triggers_a_retry():
    bad = {"raw": AIMessage(content="", response_metadata={"finish_reason": "stop"}), "parsed": None,
           "parsing_error": ValueError("'No especificada' is a placeholder")}
    model = FakeModel([bad, ok({**VALID, "tecnologias": ["base de datos", "servidor"], "nivel_de_criticidad": "media"})])
    result = run(process_text("Anda raro, creo que es la base o el server", build_chain(config(), model)))
    assert result.technologies == ["base de datos", "servidor"] and model.calls == 2
