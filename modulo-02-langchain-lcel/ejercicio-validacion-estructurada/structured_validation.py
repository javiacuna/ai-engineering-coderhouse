import asyncio
import logging
import os
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("structured_validation")
# Silence the Google library's internal logs (only show its warnings/errors)
logging.getLogger("google_genai").setLevel(logging.WARNING)


# TODO 1: Define la clase Pydantic 'EntityExtraction'
# Debe tener: topic (str), entities (Lista de str), y sentiment_score (float entre 0 y 1)
class EntityExtraction(BaseModel):
    topic: str = Field(min_length=1, description="Main topic of the text")
    entities: List[str] = Field(description="Mentioned entities: technologies, companies, people, etc.")
    sentiment_score: float = Field(
        ge=0, le=1,
        description="Sentiment of the text: 0 = very negative, 0.5 = neutral, 1 = very positive",
    )
    language: Optional[str] = Field(
        default=None,
        pattern=r"^[a-z]{2}$",
        description="Language of the text as a two-letter lowercase ISO 639-1 code (e.g. 'es', 'en')",
    )

    @field_validator("entities")
    @classmethod
    def clean_entities(cls, entities: List[str]) -> List[str]:
        """Trim whitespace, drop empty items and deduplicate (case-insensitive), keeping the order."""
        seen = set()
        cleaned = []
        for entity in entities:
            entity = entity.strip()
            if entity and entity.lower() not in seen:
                seen.add(entity.lower())
                cleaned.append(entity)
        if not cleaned:
            raise ValueError("entities must not be empty")
        return cleaned


def degraded_result(reason: str) -> EntityExtraction:
    """Fallback object when everything fails: the program continues with a neutral, flagged value.
    model_construct() builds the object WITHOUT validation, on purpose: an empty entities list would fail the validator."""
    logger.warning("Returning degraded result (%s)", reason)
    return EntityExtraction.model_construct(
        topic="unknown", entities=[], sentiment_score=0.5, language=None
    )


def create_llm(env_var: str) -> ChatGoogleGenerativeAI:
    """The model comes only from environment variables, so no hardcoded ID breaks when Google retires it."""
    model_name = os.getenv(env_var)
    if not model_name:
        raise RuntimeError(f"Missing {env_var} in .env (see .env.example)")
    return ChatGoogleGenerativeAI(model=model_name, temperature=0)


async def run_validated_chain(text: str) -> EntityExtraction:
    llm = create_llm("GEMINI_MODEL")
    fallback_llm = create_llm("GEMINI_FALLBACK_MODEL")

    # TODO 2: Configura el modelo para usar la salida estructurada con Pydantic
    structured_llm = llm.with_structured_output(EntityExtraction)

    # TODO 3: Agrega una estrategia de reintento con .with_retry()
    # para que sea resiliente ante fallos de conexión (máximo 3 intentos).
    resilient_llm = structured_llm.with_retry(
        stop_after_attempt=3,
        wait_exponential_jitter=True,
    )

    # Fallback: once the 3 attempts are exhausted, try another model (also structured and with retry)
    llm_with_fallback = resilient_llm.with_fallbacks([
        fallback_llm.with_structured_output(EntityExtraction).with_retry(stop_after_attempt=2),
    ])

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Analyze the text and extract the entities."),
        ("human", "{input}")
    ])

    # TODO 4: Une el prompt con el resilient_llm y ejecuta asíncronamente
    # No olvides manejar excepciones con try/except para capturar fallos de validación
    chain = prompt | llm_with_fallback
    try:
        logger.info("Analyzing text (%d characters)", len(text))
        result: EntityExtraction = await chain.ainvoke({"input": text})
        logger.info("Extraction OK")
        return result
    except ValidationError as e:
        logger.error("The response does not match the schema: %s", e)
        return degraded_result("validation")
    except Exception as e:
        logger.error("Both the primary and the fallback model failed: %s", e)
        return degraded_result("provider")


if __name__ == "__main__":
    sample_text = "LangGraph es una extensión de LangChain para agentes cíclicos."
    result = asyncio.run(run_validated_chain(sample_text))
    print(result.model_dump_json(indent=2))
