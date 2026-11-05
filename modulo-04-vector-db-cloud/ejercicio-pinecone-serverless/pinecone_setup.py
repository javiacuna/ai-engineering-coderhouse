import os
import asyncio
import time

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()   # reads PINECONE_API_KEY from the repo .env

# TODO: Asegúrate de tener tu PINECONE_API_KEY en las variables de entorno
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

NAMESPACE = "dev-environment"


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
        print(f"Creating index '{index_name}' (dimension={dimension}, metric=cosine)...")
        pc.create_index(
            name=index_name,
            dimension=dimension,          # must match the embedding model (1536 = OpenAI text-embedding-3-small)
            metric="cosine",              # the metric text-embedding models are trained for
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
    vectors = [
        ("vec1", [0.1] * dimension, {"topic": "infraestructura", "priority": "high"}),
        ("vec2", [0.2] * dimension, {"topic": "seguridad", "priority": "medium"}),
        ("vec3", [0.3] * dimension, {"topic": "costos", "priority": "low"}),
    ]
    # upsert: same ID -> updated, so running the script twice does not duplicate vectors
    response = index.upsert(vectors=vectors, namespace=NAMESPACE)
    print(f"Upserted {response.upserted_count} vectors into namespace '{NAMESPACE}'")

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
    stats = asyncio.run(setup_vector_infrastructure("mi-indice-serverless", 1536))
    print(stats)
    # The SDK summary only shows how many namespaces exist; print each one with its vector count
    print(f"dimension: {stats.dimension}")
    for name, summary in stats.namespaces.items():
        print(f"namespace '{name}': {summary.vector_count} vectors")
    print(f"total_vector_count: {stats.total_vector_count}")
