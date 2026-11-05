import asyncio
import logging
from typing import List, Union

from pydantic import BaseModel, Field, ValidationError, field_validator
from langchain_core.tools import tool

logger = logging.getLogger("tools.search")

SEARCH_TIMEOUT_SECONDS = 10   # a slow vector store must not block the agent forever


# TODO: Define una clase Pydantic llamada 'SearchInput'
# con los campos 'query' (str) y 'limit' (int).
class SearchInput(BaseModel):
    """Input contract of search_knowledge_base: validated BEFORE the function touches the database."""

    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Pregunta o tema a buscar, en lenguaje natural (por ejemplo: 'políticas de vacaciones').",
    )
    limit: int = Field(
        3,
        ge=1,
        le=10,   # keeps the context small: more fragments add noise (lost in the middle) and tokens
        description="Cantidad máxima de fragmentos a devolver, entre 1 y 10. Por defecto, 3.",
    )

    @field_validator("query")
    @classmethod
    def clean_query(cls, value: str) -> str:
        value = " ".join(value.split())   # the LLM may send extra spaces or line breaks
        if len(value) < 3:
            raise ValueError("query must contain at least 3 visible characters")
        return value


# Simulación de base de datos vectorial
class MockVectorDB:
    async def similarity_search(self, query: str, k: int):
        return [f"Resultado para {query}"]

db = MockVectorDB()

# TODO: Implementa la herramienta 'search_knowledge_base'
# 1. Usa el decorador @tool y asocia el esquema de Pydantic.
# 2. Asegúrate de que la función sea asíncrona (async def).
# 3. Agrega un docstring descriptivo para que el LLM sepa qué hace la herramienta.

@tool(args_schema=SearchInput)
async def search_knowledge_base(query: str, limit: int = 3) -> Union[List[str], str]:
    """Busca información en la base de conocimiento interna de la empresa (políticas de RR. HH.,
    procedimientos, documentación técnica y preguntas frecuentes) mediante búsqueda semántica.

    Usala cuando el usuario pregunte por reglas, procesos o datos propios de la empresa que no
    forman parte del conocimiento general, por ejemplo: "¿cuántos días de vacaciones tengo?" o
    "¿cómo pido acceso a producción?". No la uses para saludos, cálculos ni temas generales.

    Devuelve una lista con los fragmentos de texto más relevantes, ordenados del más al menos
    parecido a la consulta. Si la búsqueda falla, devuelve un texto que empieza con "Error:";
    en ese caso, informale al usuario que la base no está disponible en lugar de inventar la respuesta.
    """
    # Implementar llamada a db.similarity_search
    try:
        async with asyncio.timeout(SEARCH_TIMEOUT_SECONDS):
            results = await db.similarity_search(query, k=limit)
    except TimeoutError:
        logger.warning("Search timed out after %d s: %r", SEARCH_TIMEOUT_SECONDS, query)
        return f"Error: la base de conocimiento no respondió en {SEARCH_TIMEOUT_SECONDS} segundos."
    except Exception as e:   # any infrastructure failure goes back to the LLM as text, not as a crash
        logger.error("Search failed for %r: %s: %s", query, type(e).__name__, e)
        return f"Error: no se pudo consultar la base de conocimiento ({type(e).__name__})."

    if not results:
        return "No se encontraron documentos relevantes para esta consulta."
    return results


if __name__ == "__main__":
    # 1. The contract the LLM sees: name, description and argument schema
    print(search_knowledge_base.name, search_knowledge_base.args)

    # 2. Valid call
    print(asyncio.run(search_knowledge_base.ainvoke({"query": "políticas de vacaciones", "limit": 2})))

    # 3. Invalid limit: Pydantic rejects it before the function runs
    try:
        asyncio.run(search_knowledge_base.ainvoke({"query": "políticas de vacaciones", "limit": 50}))
    except ValidationError as e:
        print(f"Validation error: {e.errors()[0]['loc']} -> {e.errors()[0]['msg']}")

    # 4. Infrastructure failure: the error comes back as a message, the program keeps running
    class BrokenVectorDB:
        async def similarity_search(self, query: str, k: int):
            raise ConnectionError("vector store unreachable")

    db = BrokenVectorDB()
    print(asyncio.run(search_knowledge_base.ainvoke({"query": "políticas de vacaciones"})))
