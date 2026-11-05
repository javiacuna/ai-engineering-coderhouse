import hashlib
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List

import pytest
from langchain_core.embeddings import Embeddings

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from config import Settings  # noqa: E402
from rag_system import tokenize  # noqa: E402


class FakeEmbeddings(Embeddings):
    """Deterministic bag-of-words embeddings hashed into `dimension` buckets. Counts the embedded texts."""

    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        self.embedded_texts = 0

    def _vector(self, text: str) -> List[float]:
        vector = [0.0] * self.dimension
        for token in tokenize(text):
            vector[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dimension] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.embedded_texts += len(texts)
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vector(text)


class FakeIndex:
    """In-memory stand-in for a Pinecone index (upsert, query, list, fetch, delete)."""

    def __init__(self, fail_upserts=()):
        self.namespaces = {}
        self.fail_upserts = list(fail_upserts)   # exceptions raised by the next upsert calls
        self.upsert_calls = 0

    def upsert(self, vectors, namespace="", show_progress=False):
        self.upsert_calls += 1
        if self.fail_upserts:
            raise self.fail_upserts.pop(0)
        store = self.namespaces.setdefault(namespace, {})
        for vid, values, metadata in vectors:
            store[vid] = (values, metadata)

    def query(self, vector, top_k, namespace="", filter=None, include_metadata=False):
        def cosine(a, b):
            na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
            return sum(x * y for x, y in zip(a, b)) / (na * nb or 1.0)
        store = self.namespaces.get(namespace, {})
        scored = sorted(((cosine(vector, v), vid, m) for vid, (v, m) in store.items()), reverse=True)[:top_k]
        return SimpleNamespace(matches=[SimpleNamespace(id=vid, score=s, metadata=m) for s, vid, m in scored])

    def list(self, namespace=""):
        ids = sorted(self.namespaces.get(namespace, {}))
        for start in range(0, len(ids), 3):   # small pages, to exercise the pagination
            yield SimpleNamespace(vectors=[SimpleNamespace(id=i) for i in ids[start:start + 3]])

    def fetch(self, ids, namespace=""):
        store = self.namespaces.get(namespace, {})
        return SimpleNamespace(vectors={i: SimpleNamespace(metadata=store[i][1]) for i in ids if i in store})

    def delete(self, ids=None, delete_all=False, namespace=""):
        store = self.namespaces.setdefault(namespace, {})
        if delete_all:
            store.clear()
        for i in ids or []:
            store.pop(i, None)


def word_count(text: str) -> int:
    """Stand-in for the tiktoken counter, so the tests run offline."""
    return len(text.split())


@pytest.fixture
def settings(tmp_path) -> Settings:
    import shutil
    data = tmp_path / "data"
    shutil.copytree(PROJECT_DIR / "data", data)
    return Settings(pinecone_api_key="test", embedding_api_key="test", data_dir=data,
                    chunk_size=300, chunk_overlap=40)


@pytest.fixture
def index() -> FakeIndex:
    return FakeIndex()


@pytest.fixture
def embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()
