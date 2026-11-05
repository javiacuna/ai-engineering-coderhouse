import asyncio
import uuid
from datetime import datetime
from typing import List, Dict, Any

from pydantic import BaseModel, ConfigDict, Field


# Mock del índice de Pinecone para testing
class MockPineconeIndex:
    async def upsert(self, vectors: List[Dict]):
        # Simula la inserción en la base de datos
        print(len(vectors))   # one line per call: shows the size of each batch
        return {"upserted_count": len(vectors)}

    async def query(self, vector: List[float], top_k: int, filter: Dict = None):
        # Simula una consulta con filtro
        print(f"query(top_k={top_k}, filter={filter})")   # shows the filter that reaches the index
        return {"matches": []}


EMBEDDING_DIMENSION = 8   # simulated embedding size (a real index would use the model's dimension, e.g. 1536)


class DocumentMetadata(BaseModel):
    """Strict metadata interface, to avoid "schema drift".

    Every process writes the same field names and types: extra="forbid" rejects an unknown key
    (e.g. "date_created" instead of "ingested_at") instead of silently storing it.
    """

    model_config = ConfigDict(extra="forbid")

    category: str = Field(min_length=1)
    author: str = Field(min_length=1)
    ingested_at: str            # ISO 8601 date and time
    char_count: int = Field(ge=0)


class IngestionPipeline:
    def __init__(self, index: MockPineconeIndex):
        self.index = index

    def create_metadata(self, doc_text: str, category: str, author: str) -> Dict[str, Any]:
        """
        TODO: Implementa la creación de un diccionario de metadatos.
        Debe incluir: category, author, ingested_at (ISO format) y el tamaño del texto.
        """
        metadata = DocumentMetadata(
            category=category.strip().lower(),   # normalized: "Legal" and "legal" must match the same filter
            author=author.strip(),
            ingested_at=datetime.now().isoformat(),
            char_count=len(doc_text),
        )
        return metadata.model_dump()

    async def process_and_upsert_batches(self, documents: List[Dict], batch_size: int = 100):
        """
        TODO: Implementa la lógica de batching.
        1. Divide la lista 'documents' en trozos de tamaño 'batch_size'.
        2. Para cada batch, prepara la lista de vectores (id, values, metadata).
        3. Llama a self.index.upsert() por cada batch.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        total_processed = 0
        for start in range(0, len(documents), batch_size):
            batch = documents[start:start + batch_size]
            vectors = [
                {
                    "id": str(uuid.uuid4()),
                    "values": [0.1] * EMBEDDING_DIMENSION,   # simulated embedding of doc["text"]
                    "metadata": self.create_metadata(doc["text"], doc["category"], doc["author"]),
                }
                for doc in batch
            ]
            try:
                response = await self.index.upsert(vectors)
            except Exception as e:   # with the real API: network errors or rate limits
                raise RuntimeError(f"batch starting at document {start} failed: {e}") from e
            total_processed += response["upserted_count"]
        return total_processed

    async def search_by_category(self, query_vector: List[float], category: str):
        """
        TODO: Implementa una consulta que use un filtro de metadatos.
        Usa el operador "$eq" de Pinecone para la categoría.
        """
        # Hard filter: only vectors whose category is exactly this one are compared semantically
        return await self.index.query(
            vector=query_vector,
            top_k=5,
            filter={"category": {"$eq": category.strip().lower()}},   # same normalization as create_metadata
        )


async def main():
    index = MockPineconeIndex()
    pipeline = IngestionPipeline(index)

    # Simulación de documentos
    docs = [
        {"text": "Contrato de confidencialidad firmado con el proveedor de servicios cloud.", "category": "legal", "author": "User1"},
        {"text": "Política de protección de datos personales actualizada para 2026.", "category": "legal", "author": "User3"},
        {"text": "Procedimiento de onboarding para nuevos empleados del área técnica.", "category": "hr", "author": "User2"},
        {"text": "Calendario de vacaciones y días libres del equipo.", "category": "hr", "author": "User2"},
        {"text": "Presupuesto trimestral de infraestructura y licencias.", "category": "finanzas", "author": "User4"},
    ]

    # 1. Ejecutar ingesta
    total = await pipeline.process_and_upsert_batches(docs, batch_size=2)
    print(f"Total processed: {total}")

    # 2. Ejecutar búsqueda filtrada
    results = await pipeline.search_by_category([0.1] * EMBEDDING_DIMENSION, "legal")
    print(f"Results: {results}")

if __name__ == "__main__":
    asyncio.run(main())
