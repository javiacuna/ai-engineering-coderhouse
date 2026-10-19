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
logger = logging.getLogger("validacion_estructurada")
# Silencia los logs internos de la librería de Google (solo muestra sus warnings/errores)
logging.getLogger("google_genai").setLevel(logging.WARNING)


# TODO 1: Define la clase Pydantic 'EntityExtraction'
# Debe tener: topic (str), entities (Lista de str), y sentiment_score (float entre 0 y 1)
class EntityExtraction(BaseModel):
    topic: str = Field(min_length=1, description="Tema principal del texto")
    entities: List[str] = Field(description="Entidades mencionadas: tecnologías, empresas, personas, etc.")
    sentiment_score: float = Field(
        ge=0, le=1,
        description="Sentimiento del texto: 0 = muy negativo, 0.5 = neutral, 1 = muy positivo",
    )
    language: Optional[str] = Field(
        default=None,
        pattern=r"^[a-z]{2}$",
        description="Idioma del texto en código ISO 639-1 de dos letras minúsculas (ej. 'es', 'en')",
    )

    @field_validator("entities")
    @classmethod
    def limpiar_entidades(cls, entities: List[str]) -> List[str]:
        """Quita espacios, descarta vacías y deduplica (sin importar mayúsculas), manteniendo el orden."""
        vistas = set()
        limpias = []
        for entidad in entities:
            entidad = entidad.strip()
            if entidad and entidad.lower() not in vistas:
                vistas.add(entidad.lower())
                limpias.append(entidad)
        if not limpias:
            raise ValueError("entities no puede estar vacía")
        return limpias


def resultado_degradado(motivo: str) -> EntityExtraction:
    """Objeto de respaldo cuando todo falla: el programa sigue con un valor neutral y marcado.
    model_construct() crea el objeto SIN validar, a propósito: entities vacía no pasaría el validador."""
    logger.warning("Devolviendo resultado degradado (%s)", motivo)
    return EntityExtraction.model_construct(
        topic="desconocido", entities=[], sentiment_score=0.5, language=None
    )


def crear_llm(variable: str) -> ChatGoogleGenerativeAI:
    """El modelo sale solo de variables de entorno, así no queda un ID hardcodeado que Google retire."""
    modelo = os.getenv(variable)
    if not modelo:
        raise RuntimeError(f"Falta la variable {variable} en el .env (ver .env.example)")
    return ChatGoogleGenerativeAI(model=modelo, temperature=0)


async def run_validated_chain(text: str) -> EntityExtraction:
    llm = crear_llm("GEMINI_MODEL")
    llm_alternativo = crear_llm("GEMINI_FALLBACK_MODEL")

    # TODO 2: Configura el modelo para usar la salida estructurada con Pydantic
    structured_llm = llm.with_structured_output(EntityExtraction)

    # TODO 3: Agrega una estrategia de reintento con .with_retry()
    # para que sea resiliente ante fallos de conexión (máximo 3 intentos).
    resilient_llm = structured_llm.with_retry(
        stop_after_attempt=3,
        wait_exponential_jitter=True,
    )

    # Fallback: si se agotan los 3 intentos, prueba con otro modelo (también estructurado y con retry)
    llm_con_fallback = resilient_llm.with_fallbacks([
        llm_alternativo.with_structured_output(EntityExtraction).with_retry(stop_after_attempt=2),
    ])

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Analiza el texto y extrae las entidades."),
        ("human", "{input}")
    ])

    # TODO 4: Une el prompt con el resilient_llm y ejecuta asíncronamente
    # No olvides manejar excepciones con try/except para capturar fallos de validación
    chain = prompt | llm_con_fallback
    try:
        logger.info("Analizando texto (%d caracteres)", len(text))
        result: EntityExtraction = await chain.ainvoke({"input": text})
        logger.info("Extracción OK")
        return result
    except ValidationError as e:
        logger.error("La respuesta no cumple el esquema: %s", e)
        return resultado_degradado("validación")
    except Exception as e:
        logger.error("Fallaron el modelo principal y el alternativo: %s", e)
        return resultado_degradado("proveedor")


if __name__ == "__main__":
    sample_text = "LangGraph es una extensión de LangChain para agentes cíclicos."
    resultado = asyncio.run(run_validated_chain(sample_text))
    print(resultado.model_dump_json(indent=2))
