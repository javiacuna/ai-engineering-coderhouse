import os
import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from pinecone.errors import (
    PineconeConnectionError,
    PineconeError,
    PineconeTimeoutError,
    RateLimitError,
    ServiceError,
)

load_dotenv()   # reads PINECONE_API_KEY from the repo .env

# TODO: Asegúrate de tener tu PINECONE_API_KEY en las variables de entorno
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

NAMESPACE = "dev-environment"

# Index design (both values are fixed when the index is created and cannot be changed later):
#
# - DIMENSION = 1536: the size of the vectors produced by OpenAI's text-embedding-3-small (and ada-002).
#   The index dimension must be exactly the model's output size: a 768 or 3072 vector would be rejected
#   on upsert (dimension mismatch). Changing the embedding model means creating a new index.
#   (Gemini's gemini-embedding-001 returns 3072 by default; output_dimensionality=1536 makes it compatible.)
#
# - METRIC = "cosine": text-embedding models are trained so that the ANGLE between vectors encodes semantic
#   similarity, and OpenAI returns unit-length vectors. Cosine compares only the direction, so two texts
#   with the same meaning match even if their vectors differ in magnitude. With normalized vectors,
#   cosine and dot product give the same ranking; euclidean distance would mix magnitude into the score
#   and is the wrong choice for a model optimized for cosine similarity.
DIMENSION = 1536
METRIC = "cosine"

BATCH_SIZE = 100          # vectors per upsert request (Pinecone recommends batches of up to ~100-1000 vectors)
MAX_RETRIES = 3           # attempts per batch for transient errors

# Transient errors worth retrying: rate limit (429), server errors (5xx), network failures and timeouts.
# Any other PineconeError (bad dimension, invalid metadata, wrong key) would fail again, so it is not retried.
RETRYABLE_ERRORS = (RateLimitError, ServiceError, PineconeConnectionError, PineconeTimeoutError)

VectorTuple = Tuple[str, List[float], Dict[str, Any]]


class VectorUploadError(Exception):
    """A batch could not be upserted after all retries."""


def build_test_vectors(dimension: int) -> List[VectorTuple]:
    """Test vectors with a richer metadata schema.

    Metadata fields:
    - doc_id / chunk_index: which document and fragment the vector comes from (rebuild or delete a document)
    - source: where the text comes from (filter by origin)
    - created_at: Unix timestamp as a number, so range filters work ({"created_at": {"$gte": ...}});
      created_date keeps the readable date
    - text: the original fragment, so a search result can be shown or sent to the LLM without another lookup
      (Pinecone allows up to 40 KB of metadata per vector, enough for a chunk of ~500 tokens)
    """
    now = datetime.now(timezone.utc)
    records = [
        ("infraestructura", "high", "arquitectura.md", "La plataforma corre en Kubernetes con tres réplicas por servicio."),
        ("seguridad", "medium", "seguridad.md", "Los secretos se guardan en un gestor de secretos y se rotan cada 90 días."),
        ("costos", "low", "costos.md", "El almacenamiento serverless se cobra por uso y escala automáticamente."),
    ]
    vectors = []
    for n, (topic, priority, source, text) in enumerate(records, start=1):
        vectors.append((
            f"vec{n}",
            [0.1 * n] * dimension,   # placeholder values; in a real pipeline this is the text's embedding
            {
                "doc_id": f"doc-{source.removesuffix('.md')}",
                "chunk_index": 0,
                "source": source,
                "topic": topic,
                "priority": priority,
                "created_at": int(now.timestamp()),
                "created_date": now.date().isoformat(),
                "text": text,
            },
        ))
    return vectors


async def upsert_in_batches(index, vectors: List[VectorTuple], namespace: str,
                            batch_size: int = BATCH_SIZE) -> int:
    """Upserts the vectors in batches, retrying transient errors with exponential backoff."""
    total = 0
    for start in range(0, len(vectors), batch_size):
        batch = vectors[start:start + batch_size]
        batch_number = start // batch_size + 1
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = index.upsert(vectors=batch, namespace=namespace, show_progress=False)
                total += response.upserted_count
                print(f"Batch {batch_number}: {response.upserted_count} vectors upserted")
                break
            except RETRYABLE_ERRORS as e:
                if attempt == MAX_RETRIES:
                    raise VectorUploadError(
                        f"batch {batch_number} failed after {MAX_RETRIES} attempts: {type(e).__name__}: {e}") from e
                # Rate limit: wait what the API asks for (retry_after); otherwise 1 s, 2 s, 4 s...
                wait = getattr(e, "retry_after", None) or 2 ** (attempt - 1)
                print(f"Batch {batch_number}: {type(e).__name__}, retrying in {wait:.0f} s "
                      f"(attempt {attempt}/{MAX_RETRIES})")
                await asyncio.sleep(wait)
            except PineconeError as e:
                # Permanent error (e.g. dimension mismatch, invalid metadata): retrying would not help
                raise VectorUploadError(f"batch {batch_number} rejected: {type(e).__name__}: {e}") from e
    return total


async def setup_vector_infrastructure(index_name: str, dimension: int):
    """
    Configura la infraestructura de Pinecone Serverless y realiza una carga inicial.
    """
    if not PINECONE_API_KEY:
        raise ValueError("Missing PINECONE_API_KEY (add it to the .env file)")

    # 1. Inicializar el cliente de Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # 2. Crear el índice si no existe
    # Tip: Usa pc.list_indexes().names() para verificar existencia
    # Tip: Usa ServerlessSpec para definir la nube (aws) y región (us-east-1)
    if index_name not in pc.list_indexes().names():
        print(f"Creating index '{index_name}' (dimension={dimension}, metric={METRIC})...")
        pc.create_index(
            name=index_name,
            dimension=dimension,          # must match the embedding model (see DIMENSION above)
            metric=METRIC,                # see METRIC above
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),   # the region the free plan allows
        )
    else:
        print(f"Index '{index_name}' already exists: reusing it")

    # Wait until the index is ready before using it
    while not pc.describe_index(index_name).status.ready:
        await asyncio.sleep(1)

    # An existing index with another dimension would reject every upsert (dimension mismatch)
    existing_dimension = pc.describe_index(index_name).dimension
    if existing_dimension != dimension:
        raise ValueError(f"Index '{index_name}' has dimension {existing_dimension}, expected {dimension}")

    # 3. Conectar al índice
    index = pc.Index(index_name)

    # 4. Implementar un Upsert de prueba usando un Namespace
    # Crea una pequeña lista de tuplas (id, vector, metadata)
    vectors = build_test_vectors(dimension)
    # upsert: same ID -> updated, so running the script twice does not duplicate vectors
    upserted = await upsert_in_batches(index, vectors, NAMESPACE)
    print(f"Upserted {upserted} vectors into namespace '{NAMESPACE}'")

    # 5. Retornar las estadísticas del índice usando describe_index_stats()
    # Stats are eventually consistent: wait (up to ~30 s) until the namespace shows the vectors
    stats = index.describe_index_stats()
    deadline = time.monotonic() + 30
    while stats.namespaces.get(NAMESPACE, None) is None or stats.namespaces[NAMESPACE].vector_count < len(vectors):
        if time.monotonic() > deadline:
            print("Stats not updated yet: run the script again in a few seconds")
            break
        await asyncio.sleep(2)
        stats = index.describe_index_stats()
    return stats


if __name__ == "__main__":
    # Ejemplo con dimensión 1536 (OpenAI)
    try:
        stats = asyncio.run(setup_vector_infrastructure("mi-indice-serverless", DIMENSION))
    except (VectorUploadError, PineconeError, ValueError) as e:
        print(f"Error: {e}")
        raise SystemExit(1)
    print(stats)
    # The SDK summary only shows how many namespaces exist; print each one with its vector count
    print(f"dimension: {stats.dimension}")
    for name, summary in stats.namespaces.items():
        print(f"namespace '{name}': {summary.vector_count} vectors")
    print(f"total_vector_count: {stats.total_vector_count}")
