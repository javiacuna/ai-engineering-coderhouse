"""Configuration of the cloud RAG system, read from environment variables (.env) and validated with Pydantic."""

import os
from enum import Enum
from pathlib import Path
from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

BASE_DIR = Path(__file__).resolve().parent


class EmbeddingProvider(str, Enum):
    GEMINI = "gemini"
    OPENAI = "openai"


DEFAULT_EMBEDDING_MODELS = {
    EmbeddingProvider.GEMINI: "gemini-embedding-001",
    EmbeddingProvider.OPENAI: "text-embedding-3-small",
}

KEY_ENV_VARS = {
    EmbeddingProvider.GEMINI: ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    EmbeddingProvider.OPENAI: ("OPENAI_API_KEY",),
}


def _secret(*names: str, required: bool = True) -> Optional[SecretStr]:
    value = next((os.getenv(n) for n in names if os.getenv(n)), None)
    if not value and required:
        raise ValueError(f"Missing {' or '.join(names)} in the environment (.env)")
    return SecretStr(value) if value else None


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    # Pinecone
    pinecone_api_key: SecretStr
    index_name: str = "rag-asyncio-docs"
    namespace: str = "python-asyncio-docs"    # one logical partition per dataset / tenant
    cloud: str = "aws"
    region: str = "us-east-1"                 # the region the free plan allows

    # Index design: both are fixed when the index is created.
    # 1536 = OpenAI text-embedding-3-small. gemini-embedding-001 produces 3072 by default but is
    # trained with Matryoshka representation learning, so output_dimensionality=1536 keeps the first
    # 1536 dimensions with little quality loss and makes both providers fit the same index size.
    dimension: int = 1536
    # Cosine: text embedding models encode meaning in the angle between vectors. Gemini vectors
    # truncated to 1536 are not unit-length, and cosine ignores the magnitude, so it stays correct.
    metric: str = "cosine"

    # Embeddings: the SAME model indexes and queries (the model name is stored in every vector)
    embedding_provider: EmbeddingProvider = EmbeddingProvider.GEMINI
    embedding_model: str = DEFAULT_EMBEDDING_MODELS[EmbeddingProvider.GEMINI]
    embedding_api_key: SecretStr

    # Data and chunking (tokens, cl100k_base): the assignment suggests ~500-800 tokens
    data_dir: Path = BASE_DIR / "data"
    chunk_size: int = Field(default=500, ge=300, le=800)
    chunk_overlap: int = Field(default=75, ge=0)
    batch_size: int = Field(default=100, ge=1, le=1000)

    # Retrieval
    top_k: int = Field(default=5, ge=1, le=10)
    candidates_per_retriever: int = Field(default=10, ge=1, le=50)   # each retriever's list before the fusion
    vector_weight: float = Field(default=0.5, ge=0, le=1)            # BM25 gets 1 - vector_weight

    # Optional answer generation (ask.py)
    llm_model: str = "gemini-3.8-flash"
    llm_fallback_model: Optional[str] = None
    google_api_key: Optional[SecretStr] = None

    @model_validator(mode="after")
    def check_values(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.candidates_per_retriever < self.top_k:
            raise ValueError("candidates_per_retriever must be at least top_k")
        return self

    @classmethod
    def from_env(cls) -> "Settings":
        provider = EmbeddingProvider(os.getenv("EMBEDDING_PROVIDER", "gemini").lower())
        optional = {
            "index_name": os.getenv("INDEX_NAME"),
            "namespace": os.getenv("PINECONE_NAMESPACE"),
            "chunk_size": os.getenv("CHUNK_SIZE"),
            "chunk_overlap": os.getenv("CHUNK_OVERLAP"),
            "top_k": os.getenv("TOP_K"),
            "vector_weight": os.getenv("VECTOR_WEIGHT"),
            "llm_model": os.getenv("GEMINI_MODEL"),
            "llm_fallback_model": os.getenv("GEMINI_FALLBACK_MODEL"),
        }
        return cls(
            pinecone_api_key=_secret("PINECONE_API_KEY"),
            embedding_provider=provider,
            embedding_model=os.getenv("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODELS[provider],
            embedding_api_key=_secret(*KEY_ENV_VARS[provider]),
            google_api_key=_secret("GOOGLE_API_KEY", "GEMINI_API_KEY", required=False),
            **{k: v for k, v in optional.items() if v},
        )


def build_embeddings(settings: Settings) -> Embeddings:
    """Embedding model used BOTH by the ingestion and by the vector retriever, at the index dimension."""
    key = settings.embedding_api_key.get_secret_value()
    if settings.embedding_provider == EmbeddingProvider.OPENAI:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=settings.embedding_model, api_key=key, dimensions=settings.dimension)
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    return GoogleGenerativeAIEmbeddings(model=settings.embedding_model, google_api_key=key,
                                        output_dimensionality=settings.dimension)


def build_chat_model(settings: Settings, model: Optional[str] = None) -> BaseChatModel:
    if settings.google_api_key is None:
        raise ValueError("GOOGLE_API_KEY is needed to generate answers")
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=model or settings.llm_model, temperature=0,
                                  google_api_key=settings.google_api_key.get_secret_value())
