import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils import embedding_functions


class VectorMemoryManager:
    def __init__(self, persist_path: str, collection_name: str, embedding_function=None):
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

    def upsert_documents(self, ids: List[str], documents: List[str], metadatas: List[Dict[str, Any]]):
        """
        Implementa el método upsert para evitar errores por IDs duplicados.
        """
        # TODO: Usar self.collection.upsert
        if not (len(ids) == len(documents) == len(metadatas)):
            raise ValueError("ids, documents and metadatas must have the same length")
        # upsert: if the ID already exists the document (and its vector) is updated; otherwise it is created
        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def semantic_search(self, query_text: str, n_results: int = 3,
                        where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Realiza una búsqueda por similitud.
        Retorna los documentos más cercanos al query_text.
        """
        # TODO: Implementar query() y retornar los resultados
        n_results = min(n_results, self.collection.count())   # Chroma warns if more results are requested than exist
        if n_results == 0:
            return []
        results = self.collection.query(query_texts=[query_text], n_results=n_results, where=where)

        # query() returns one list per query text; only one query is sent, hence the [0]
        return [
            {"id": doc_id, "document": document, "metadata": metadata, "distance": round(distance, 4)}
            for doc_id, document, metadata, distance in zip(
                results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
            )
        ]

    def delete_by_id(self, ids: List[str]):
        """Deletes documents (and their vectors) by ID. IDs that do not exist are ignored."""
        self.collection.delete(ids=ids)


def content_id(text: str) -> str:
    """Deterministic ID from the content (the lesson recommends it over random IDs, so upsert can update)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


if __name__ == "__main__":
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
