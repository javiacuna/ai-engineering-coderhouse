import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.errors import ChromaError
from chromadb.utils import embedding_functions

logger = logging.getLogger("vector_memory")

# Errors the database layer can raise:
# - ChromaError: base class of every Chroma error (NotFoundError, DuplicateIDError, InvalidArgumentError,
#   InternalError for disk/storage failures, etc.). Chroma 1.x has no DatabaseError class; ChromaError covers it.
# - ValueError: Chroma's client-side validation (malformed `where` filter, unsupported metadata types).
# - OSError: the persist folder cannot be read or written (permissions, full disk).
DATABASE_ERRORS = (ChromaError, ValueError, OSError)


class VectorMemoryError(Exception):
    """Single, controlled error for any failure while talking to the vector database."""


class VectorMemoryManager:
    def __init__(self, persist_path: str, collection_name: str, embedding_function=None):
        try:
            # TODO: Inicializar el PersistentClient de ChromaDB con la ruta dada
            # PersistentClient stores the data on disk, so it survives a restart of the script
            self.client = chromadb.PersistentClient(path=persist_path)

            # TODO: Crear u obtener la colección usando self.client.get_or_create_collection
            # No olvides asignar una función de embedding por defecto
            # Always use the same embedding model the collection was created with: vectors from
            # another model live in a different space and the search would return noise.
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=embedding_function or embedding_functions.DefaultEmbeddingFunction(),
                # Cosine distance (1 - cosine similarity), the usual metric for text embeddings.
                # Chroma's default is squared L2. It only applies when the collection is first created.
                metadata={"hnsw:space": "cosine"},
            )
        except DATABASE_ERRORS as e:
            logger.error("Could not open collection '%s' at %s: %s: %s",
                         collection_name, persist_path, type(e).__name__, e)
            raise VectorMemoryError(f"could not open collection '{collection_name}': {e}") from e

    def upsert_documents(self, ids: List[str], documents: List[str], metadatas: List[Dict[str, Any]]):
        """
        Implementa el método upsert para evitar errores por IDs duplicados.
        """
        # TODO: Usar self.collection.upsert
        # Input check before touching the database (a programming error, so it stays a ValueError)
        if not (len(ids) == len(documents) == len(metadatas)):
            raise ValueError("ids, documents and metadatas must have the same length")
        try:
            # upsert: if the ID already exists the document (and its vector) is updated; otherwise it is created
            self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        except DATABASE_ERRORS as e:
            logger.error("Upsert of %d documents failed: %s: %s", len(ids), type(e).__name__, e)
            raise VectorMemoryError(f"upsert failed: {e}") from e
        logger.info("Upserted %d documents", len(ids))

    def semantic_search(self, query_text: str, n_results: int = 3,
                        where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Realiza una búsqueda por similitud.
        Retorna los documentos más cercanos al query_text.
        """
        # TODO: Implementar query() y retornar los resultados
        try:
            n_results = min(n_results, self.collection.count())   # Chroma warns if more results are requested than exist
            if n_results == 0:
                return []
            results = self.collection.query(query_texts=[query_text], n_results=n_results, where=where)
        except DATABASE_ERRORS as e:
            logger.error("Search for %r (where=%s) failed: %s: %s", query_text, where, type(e).__name__, e)
            raise VectorMemoryError(f"semantic search failed: {e}") from e

        # query() returns one list per query text; only one query is sent, hence the [0]
        return [
            {"id": doc_id, "document": document, "metadata": metadata, "distance": round(distance, 4)}
            for doc_id, document, metadata, distance in zip(
                results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
            )
        ]

    def delete_by_id(self, ids: List[str]):
        """Deletes documents (and their vectors) by ID. IDs that do not exist are ignored."""
        try:
            self.collection.delete(ids=ids)
        except DATABASE_ERRORS as e:
            logger.error("Delete of %s failed: %s: %s", ids, type(e).__name__, e)
            raise VectorMemoryError(f"delete failed: {e}") from e
        logger.info("Deleted ids %s", ids)


def content_id(text: str) -> str:
    """Deterministic ID from the content (the lesson recommends it over random IDs, so upsert can update)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(name)s | %(message)s")

    # The database lives next to this script, regardless of the folder it is run from
    db_path = str(Path(__file__).resolve().parent / "chroma_db")

    manager = VectorMemoryManager(db_path, "knowledge_base")
    manager.upsert_documents(
        ["doc1", "doc2", "doc3"],
        ["Python 3.12 introduce mejoras en asyncio y tipos.",
         "Las bases de datos vectoriales son clave para la memoria de largo plazo en IA.",
         "FastAPI permite crear APIs asíncronas en Python."],
        [{"source": "news"}, {"source": "tech_blog"}, {"source": "tech_blog"}],
    )
    print(f"Documents after upsert: {manager.collection.count()}")

    print("\n1) semantic_search('mejoras en python')")
    for r in manager.semantic_search("mejoras en python"):
        print(f"   {r['distance']:.4f}  [{r['id']}] {r['document']}  {r['metadata']}")

    print("\n2) semantic_search('memoria de IA', where={'source': 'tech_blog'})")
    for r in manager.semantic_search("memoria de IA", where={"source": "tech_blog"}):
        print(f"   {r['distance']:.4f}  [{r['id']}] {r['document']}  {r['metadata']}")

    manager.delete_by_id(["doc2"])
    print(f"\n3) count() after deleting doc2: {manager.collection.count()}")

    # Upserting the same ID again updates instead of duplicating
    manager.upsert_documents(["doc1"], ["Python 3.12 introduce mejoras en asyncio, tipos y mensajes de error."],
                             [{"source": "news", "version": 2}])
    print(f"4) count() after upserting doc1 again: {manager.collection.count()} "
          f"(updated: {manager.collection.get(ids=['doc1'])['metadatas'][0]})")

    # Failures are reported as VectorMemoryError instead of crashing with a raw Chroma traceback
    print("\n5) Error handling")
    failing_calls = {
        "duplicate IDs in one upsert": lambda: manager.upsert_documents(
            ["dup", "dup"], ["a", "b"], [{"source": "test"}, {"source": "test"}]),
        "malformed where filter": lambda: manager.semantic_search("python", where={"source": {"$foo": "x"}}),
    }
    for label, call in failing_calls.items():
        try:
            call()
        except VectorMemoryError as e:
            print(f"   {label}: handled -> {str(e)[:80]}")
