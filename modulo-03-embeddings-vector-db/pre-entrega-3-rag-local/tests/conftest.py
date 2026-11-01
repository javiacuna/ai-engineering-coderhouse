import hashlib
import math
import re
import shutil
import sys
from pathlib import Path
from typing import List

import pytest
from langchain_core.embeddings import Embeddings

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from config import EmbeddingProvider, RAGConfig  # noqa: E402


class FakeEmbeddings(Embeddings):
    """Deterministic bag-of-words embeddings (hashed into 256 dimensions). Counts how many texts it embeds."""

    def __init__(self):
        self.embedded_texts = 0

    def _vector(self, text: str) -> List[float]:
        vector = [0.0] * 256
        for word in re.findall(r"\w+", text.lower()):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % 256] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.embedded_texts += len(texts)
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vector(text)


def word_count(text: str) -> int:
    """Stand-in for the tiktoken counter, so the tests run offline."""
    return len(text.split())


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    target = tmp_path / "data"
    shutil.copytree(PROJECT_DIR / "data", target)
    return target


@pytest.fixture
def config(tmp_path: Path, data_dir: Path) -> RAGConfig:
    return RAGConfig(
        data_dir=data_dir,
        persist_dir=tmp_path / "vectorstore",
        collection_name="test_collection",
        embedding_provider=EmbeddingProvider.LOCAL,
        embedding_model="fake-bag-of-words",
    )


@pytest.fixture
def embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()
