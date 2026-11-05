import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("ingestion")


# Mock del índice de Pinecone para testing
class MockPineconeIndex:
    """Simulated Pinecone index. Stores the vectors by ID (an upsert with an existing ID overwrites it)
    and applies the $eq / $and filters, so deduplication and tenant isolation can be checked."""

    def __init__(self):
        self.vectors: Dict[str, Dict] = {}

    async def upsert(self, vectors: List[Dict]):
        # Simula la inserción en la base de datos
        print(len(vectors))   # one line per call: shows the size of each batch
        for vector in vectors:
            self.vectors[vector["id"]] = vector
        return {"upserted_count": len(vectors)}

    async def query(self, vector: List[float], top_k: int, filter: Dict = None):
        # Simula una consulta con filtro
        print(f"query(top_k={top_k}, filter={filter})")   # shows the filter that reaches the index
        matches = [{"id": v["id"], "metadata": v["metadata"]}
                   for v in self.vectors.values() if matches_filter(v["metadata"], filter)]
        return {"matches": matches[:top_k]}

    def count(self) -> int:
        return len(self.vectors)


def matches_filter(metadata: Dict[str, Any], filter: Optional[Dict]) -> bool:
    """Minimal evaluator of the Pinecone filter syntax used here: {"field": {"$eq": value}} and {"$and": [...]}."""
    if not filter:
        return True
    for key, condition in filter.items():
        if key == "$and":
            if not all(matches_filter(metadata, sub) for sub in condition):
                return False
        elif metadata.get(key) != condition["$eq"]:
            return False
    return True


class FlakyMockIndex(MockPineconeIndex):
    """Mock that fails the first `failures` upserts with a network error, to exercise the retries."""

    def __init__(self, failures: int):
        super().__init__()
        self.failures = failures

    async def upsert(self, vectors: List[Dict]):
        if self.failures > 0:
            self.failures -= 1
            raise ConnectionError("simulated network error")
        return await super().upsert(vectors)


EMBEDDING_DIMENSION = 8   # simulated embedding size (a real index would use the model's dimension, e.g. 1536)

# Temporary errors worth retrying. With the real SDK: RateLimitError, ServiceError,
# PineconeConnectionError and PineconeTimeoutError (see the Pinecone Serverless exercise).
TRANSIENT_ERRORS = (ConnectionError, TimeoutError)


class DocumentMetadata(BaseModel):
    """Strict metadata interface, to avoid "schema drift".

    Every process writes the same field names and types: extra="forbid" rejects an unknown key
    (e.g. "date_created" instead of "ingested_at") instead of silently storing it.
    """

    model_config = ConfigDict(extra="forbid")

    tenant_id: str = Field(min_length=1)        # owner of the document (multi-tenant isolation)
    category: str = Field(min_length=1)
    author: str = Field(min_length=1)
    ingested_at: str                            # ISO 8601 date and time
    char_count: int = Field(ge=0)
    content_hash: str                           # SHA-256 of the text, to detect duplicates


@dataclass
class FailedBatch:
    start: int                  # position of the batch's first document in the input list
    ids: List[str]
    error: str


@dataclass
class IngestionReport:
    processed: int = 0
    duplicates_skipped: int = 0
    failed_batches: List[FailedBatch] = field(default_factory=list)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def document_id(tenant_id: str, text: str) -> str:
    """Deterministic ID: the same text of the same tenant always gets the same ID, so a repeated load
    overwrites the vector instead of creating a duplicate. The tenant is part of the ID because two
    tenants may upload the same text and each one must keep its own copy."""
    return hashlib.sha256(f"{tenant_id}:{content_hash(text)}".encode("utf-8")).hexdigest()[:32]


class IngestionPipeline:
    def __init__(self, index: MockPineconeIndex, max_retries: int = 3, base_delay: float = 0.5):
        self.index = index
        self.max_retries = max_retries      # total attempts per batch
        self.base_delay = base_delay        # first backoff wait, in seconds (then x2 each time)

    def create_metadata(self, doc_text: str, category: str, author: str, tenant_id: str) -> Dict[str, Any]:
        """
        Builds and validates the metadata of a document: tenant_id, category, author,
        ingested_at (ISO 8601), char_count and content_hash. Category and tenant are normalized
        (trimmed, lowercase) so they match the filters of search_by_category.
        Raises pydantic.ValidationError if a field is missing, empty or unknown.
        """
        metadata = DocumentMetadata(
            tenant_id=tenant_id.strip().lower(),
            category=category.strip().lower(),   # "Legal" and "legal" must match the same filter
            author=author.strip(),
            ingested_at=datetime.now().isoformat(),
            char_count=len(doc_text),
            content_hash=content_hash(doc_text),
        )
        return metadata.model_dump()

    async def _upsert_with_retry(self, vectors: List[Dict], batch_number: int) -> int:
        """Upserts one batch, retrying temporary errors with exponential backoff. Raises the last error."""
        for attempt in range(1, self.max_retries + 1):
            try:
                response = await self.index.upsert(vectors)
                return response["upserted_count"]
            except TRANSIENT_ERRORS as e:
                if attempt == self.max_retries:
                    raise
                wait = self.base_delay * 2 ** (attempt - 1)
                logger.warning("Batch %d: %s (attempt %d/%d), retrying in %.1f s",
                               batch_number, e, attempt, self.max_retries, wait)
                await asyncio.sleep(wait)

    async def process_and_upsert_batches(self, documents: List[Dict], batch_size: int = 100) -> IngestionReport:
        """
        Ingests the documents in batches of `batch_size` with one upsert call per batch.

        - Each vector gets a deterministic ID (tenant + text hash): loading the same documents
          again overwrites them instead of duplicating them, and repeated documents inside the
          same load are skipped.
        - A batch that fails with a temporary error is retried with exponential backoff. If it
          still fails (or the error is permanent), it is logged and recorded in the report, and the
          ingestion continues with the next batch instead of aborting.

        Returns an IngestionReport with the processed count, the skipped duplicates and the failed batches.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        report = IngestionReport()
        seen_ids = set()
        for start in range(0, len(documents), batch_size):
            batch_number = start // batch_size + 1
            vectors = []
            for doc in documents[start:start + batch_size]:
                doc_id = document_id(doc["tenant_id"].strip().lower(), doc["text"])
                if doc_id in seen_ids:
                    report.duplicates_skipped += 1
                    continue
                seen_ids.add(doc_id)
                vectors.append({
                    "id": doc_id,
                    "values": [0.1] * EMBEDDING_DIMENSION,   # simulated embedding of doc["text"]
                    "metadata": self.create_metadata(doc["text"], doc["category"], doc["author"], doc["tenant_id"]),
                })
            if not vectors:
                continue
            try:
                report.processed += await self._upsert_with_retry(vectors, batch_number)
            except Exception as e:   # retries exhausted, or a permanent error: record it and go on
                logger.error("Batch %d failed: %s: %s", batch_number, type(e).__name__, e)
                report.failed_batches.append(FailedBatch(start=start, ids=[v["id"] for v in vectors],
                                                         error=f"{type(e).__name__}: {e}"))
        return report

    async def search_by_category(self, query_vector: List[float], category: str, tenant_id: str):
        """
        Semantic search restricted to ONE tenant and ONE category, with a combined hard filter:
        {"$and": [{"tenant_id": {"$eq": ...}}, {"category": {"$eq": ...}}]}.
        tenant_id is mandatory, so no query can return another tenant's documents.
        """
        if not tenant_id or not tenant_id.strip():
            raise ValueError("tenant_id is required: queries without it could leak other tenants' data")
        return await self.index.query(
            vector=query_vector,
            top_k=5,
            filter={"$and": [
                {"tenant_id": {"$eq": tenant_id.strip().lower()}},   # security filter: always first
                {"category": {"$eq": category.strip().lower()}},     # same normalization as create_metadata
            ]},
        )


async def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    index = MockPineconeIndex()
    pipeline = IngestionPipeline(index)

    # Simulación de documentos: 2 legal, 2 hr, 1 finanzas, from two companies (tenants)
    docs = [
        {"text": "Contrato de confidencialidad firmado con el proveedor de servicios cloud.", "category": "legal", "author": "User1", "tenant_id": "acme"},
        {"text": "Política de protección de datos personales actualizada para 2026.", "category": "legal", "author": "User3", "tenant_id": "globex"},
        {"text": "Procedimiento de onboarding para nuevos empleados del área técnica.", "category": "hr", "author": "User2", "tenant_id": "acme"},
        {"text": "Calendario de vacaciones y días libres del equipo.", "category": "hr", "author": "User2", "tenant_id": "globex"},
        {"text": "Presupuesto trimestral de infraestructura y licencias.", "category": "finanzas", "author": "User4", "tenant_id": "acme"},
    ]

    # 1. Ejecutar ingesta
    print("== 1) Ingestion (batch_size=2)")
    report = await pipeline.process_and_upsert_batches(docs, batch_size=2)
    print(f"Total processed: {report.processed} | vectors in the index: {index.count()}")

    print("\n== 2) Same load again (deduplication)")
    report = await pipeline.process_and_upsert_batches(docs + [docs[0]], batch_size=2)
    print(f"Processed: {report.processed} | duplicates skipped: {report.duplicates_skipped} | "
          f"vectors in the index: {index.count()} (still 5)")

    # 2. Ejecutar búsqueda filtrada
    print("\n== 3) Search 'legal' for tenant 'acme' (combined filter)")
    results = await pipeline.search_by_category([0.1] * EMBEDDING_DIMENSION, "legal", tenant_id="acme")
    for match in results["matches"]:
        print(f"   {match['metadata']['tenant_id']} | {match['metadata']['category']} | {match['id'][:8]}...")
    print(f"   {len(results['matches'])} match(es): globex's legal document is not visible to acme")

    print("\n== 4) Temporary error: the batch is retried with backoff")
    flaky = IngestionPipeline(FlakyMockIndex(failures=1), base_delay=0.1)
    report = await flaky.process_and_upsert_batches(docs, batch_size=2)
    print(f"Processed: {report.processed} | failed batches: {len(report.failed_batches)}")

    print("\n== 5) Error that persists: the batch is recorded and the rest continues")
    broken = IngestionPipeline(FlakyMockIndex(failures=3), base_delay=0.1)
    report = await broken.process_and_upsert_batches(docs, batch_size=2)
    print(f"Processed: {report.processed} | failed batches: "
          f"{[(b.start, len(b.ids), b.error) for b in report.failed_batches]}")

if __name__ == "__main__":
    asyncio.run(main())
