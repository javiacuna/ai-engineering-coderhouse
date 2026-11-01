"""Configuration of the RAG system, read from environment variables (.env) and validated with Pydantic."""

import os
from enum import Enum
from pathlib import Path
from typing import List, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

BASE_DIR = Path(__file__).resolve().parent


class LLMProvider(str, Enum):
    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class EmbeddingProvider(str, Enum):
    GEMINI = "gemini"
    OPENAI = "openai"
    LOCAL = "local"   # Chroma's built-in all-MiniLM-L6-v2 (ONNX): free and offline, weaker in Spanish


DEFAULT_LLM_MODELS = {
    LLMProvider.GEMINI: "gemini-3.8-flash",
    LLMProvider.OPENAI: "gpt-4o-mini",
    LLMProvider.ANTHROPIC: "claude-haiku-4-5",
}

DEFAULT_EMBEDDING_MODELS = {
    EmbeddingProvider.GEMINI: "gemini-embedding-001",
    EmbeddingProvider.OPENAI: "text-embedding-3-small",
    EmbeddingProvider.LOCAL: "all-MiniLM-L6-v2",
}

KEY_ENV_VARS = {
    "gemini": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
}


def _api_key(provider: str) -> Optional[SecretStr]:
    """First non-empty key among the env vars of the provider (None for the local embeddings)."""
    if provider not in KEY_ENV_VARS:
        return None
    value = next((os.getenv(v) for v in KEY_ENV_VARS[provider] if os.getenv(v)), None)
    if not value:
        raise ValueError(f"Missing {' or '.join(KEY_ENV_VARS[provider])} for provider '{provider}'")
    return SecretStr(value)


class RAGConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    # Paths
    data_dir: Path = BASE_DIR / "data"
    persist_dir: Path = BASE_DIR / "vectorstore"
    collection_name: str = "knowledge_base"
    file_patterns: List[str] = ["*.md", "*.txt"]

    # Chunking, measured in tokens (cl100k_base). The assignment asks for at least 500 tokens with 50 of overlap.
    chunk_size: int = Field(default=500, ge=500)
    chunk_overlap: int = Field(default=50, ge=0)

    # Retrieval: 3 to 5 fragments, to avoid the "infinite context" / lost-in-the-middle problem
    top_k: int = Field(default=4, ge=3, le=5)
    max_distance: Optional[float] = Field(
        default=None, gt=0, le=2,
        description="Optional cosine-distance cut-off: fragments farther than this are discarded",
    )

    # Embeddings: the SAME model must be used to index and to query (checked against the index manifest)
    embedding_provider: EmbeddingProvider = EmbeddingProvider.GEMINI
    embedding_model: str = DEFAULT_EMBEDDING_MODELS[EmbeddingProvider.GEMINI]
    embedding_api_key: Optional[SecretStr] = None

    # Generation
    llm_provider: LLMProvider = LLMProvider.GEMINI
    llm_model: str = DEFAULT_LLM_MODELS[LLMProvider.GEMINI]
    llm_fallback_model: Optional[str] = None
    llm_api_key: Optional[SecretStr] = None
    max_tokens: int = Field(default=2048, gt=0)
    max_attempts: int = Field(default=3, ge=1, le=5, description="Total attempts when the answer cannot be parsed")

    @model_validator(mode="after")
    def check_overlap(self) -> "RAGConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

    @classmethod
    def from_env(cls, require_llm: bool = True) -> "RAGConfig":
        """Builds the config from env vars. require_llm=False skips the LLM key check (ingestion only)."""
        llm_provider = LLMProvider(os.getenv("LLM_PROVIDER", "gemini").lower())
        emb_provider = EmbeddingProvider(os.getenv("EMBEDDING_PROVIDER", "gemini").lower())
        is_gemini = llm_provider == LLMProvider.GEMINI
        optional = {
            "chunk_size": os.getenv("RAG_CHUNK_SIZE"),
            "chunk_overlap": os.getenv("RAG_CHUNK_OVERLAP"),
            "top_k": os.getenv("RAG_TOP_K"),
            "max_distance": os.getenv("RAG_MAX_DISTANCE"),
            "max_tokens": os.getenv("LLM_MAX_TOKENS"),
            "max_attempts": os.getenv("LLM_MAX_ATTEMPTS"),
        }
        return cls(
            embedding_provider=emb_provider,
            embedding_model=os.getenv("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODELS[emb_provider],
            embedding_api_key=_api_key(emb_provider.value),
            llm_provider=llm_provider,
            # With Gemini, the GEMINI_MODEL / GEMINI_FALLBACK_MODEL variables of the repo .env are also accepted
            llm_model=os.getenv("LLM_MODEL") or (is_gemini and os.getenv("GEMINI_MODEL")) or DEFAULT_LLM_MODELS[llm_provider],
            llm_fallback_model=os.getenv("LLM_FALLBACK_MODEL") or (is_gemini and os.getenv("GEMINI_FALLBACK_MODEL")) or None,
            llm_api_key=_api_key(llm_provider.value) if require_llm else None,
            **{k: v for k, v in optional.items() if v},
        )


class LocalEmbeddings(Embeddings):
    """LangChain adapter for Chroma's default embedding function (all-MiniLM-L6-v2, runs locally with ONNX)."""

    def __init__(self):
        from chromadb.utils import embedding_functions
        self._function = embedding_functions.DefaultEmbeddingFunction()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[float(x) for x in vector] for vector in self._function(texts)]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def build_embeddings(config: RAGConfig) -> Embeddings:
    """Embedding model used BOTH by the ingestion and by the retriever."""
    key = config.embedding_api_key.get_secret_value() if config.embedding_api_key else None
    if config.embedding_provider == EmbeddingProvider.GEMINI:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model=config.embedding_model, google_api_key=key)
    if config.embedding_provider == EmbeddingProvider.OPENAI:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=config.embedding_model, api_key=key)
    return LocalEmbeddings()


def build_chat_model(config: RAGConfig, model: Optional[str] = None) -> BaseChatModel:
    """LangChain chat model of the configured provider."""
    if config.llm_api_key is None:
        raise ValueError("The LLM API key is missing (build the config with require_llm=True)")
    name = model or config.llm_model
    key = config.llm_api_key.get_secret_value()
    if config.llm_provider == LLMProvider.OPENAI:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=name, api_key=key, temperature=0, max_tokens=config.max_tokens)
    if config.llm_provider == LLMProvider.ANTHROPIC:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=name, api_key=key, max_tokens=config.max_tokens)
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=name, google_api_key=key, temperature=0, max_output_tokens=config.max_tokens)
