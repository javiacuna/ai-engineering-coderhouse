"""Ingestion pipeline: Markdown documents -> chunks -> embeddings -> Pinecone Serverless.

Every chunk is stored with its original text and advanced metadata (source, page/section, category,
tags), so retrieval needs no extra database. The load is incremental: IDs are deterministic
(<doc_id>#<chunk>), unchanged chunks are not embedded again, and chunks that no longer exist are deleted.

Usage:
    python ingest.py             # sync data/ with the index
    python ingest.py --rebuild   # delete the namespace and load everything again
"""

import argparse
import asyncio
import hashlib
import logging
import re
import sys
import time
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from pinecone.errors import (
    NotFoundError,
    PineconeConnectionError,
    PineconeError,
    PineconeTimeoutError,
    RateLimitError,
    ServiceError,
)
from pydantic import BaseModel, ConfigDict, Field

from config import Settings, build_embeddings
from setup_index import IndexConfigError, ensure_index

logger = logging.getLogger("rag.ingest")

SEPARATORS = ["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]
HEADING = re.compile(r"^## +(.+)$", re.MULTILINE)
INTRO_SECTION = "Introducción"

RETRYABLE_ERRORS = (RateLimitError, ServiceError, PineconeConnectionError, PineconeTimeoutError)
MAX_RETRIES = 3
FETCH_BATCH = 100   # IDs per fetch request


class IngestError(Exception):
    """The documents could not be processed or uploaded."""


class ChunkMetadata(BaseModel):
    """Strict metadata schema of every vector (extra="forbid" prevents schema drift)."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str                 # file name without extension: the unit the golden set refers to
    source: str                 # file name
    title: str
    category: str
    tags: List[str]             # Pinecone supports lists of strings: filter with {"tags": {"$in": [...]}}
    section: str                # "## " heading the chunk belongs to
    page: int = Field(ge=1)     # section number (Markdown has no pages; a PDF loader would store the real page)
    chunk_index: int = Field(ge=0)
    tokens: int = Field(ge=1)
    content_hash: str           # SHA-256 of the text: unchanged chunks are not embedded again
    embedding_model: str        # model that produced the vector: checked at query time
    ingested_at: int            # Unix timestamp (a number, so range filters work)
    text: str                   # original chunk: what BM25 indexes and what the LLM receives


class IngestReport(BaseModel):
    documents: int = 0
    chunks: int = 0
    embedded: int = 0
    unchanged: int = 0
    deleted: int = 0
    failed_batches: List[str] = []


# ---------------------------------------------------------------------------
# 1. Loading: Markdown with a small front matter (category, tags)
# ---------------------------------------------------------------------------
def parse_front_matter(text: str) -> Tuple[Dict[str, Any], str]:
    """Reads a "---\\nkey: value\\n---" header. tags accepts "[a, b]" or "a, b"."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    meta = {}
    for line in text[3:end].strip().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    if "tags" in meta:
        meta["tags"] = [t.strip() for t in meta["tags"].strip("[]").split(",") if t.strip()]
    return meta, text[end + 4:].lstrip()


def load_documents(settings: Settings) -> List[Document]:
    if not settings.data_dir.is_dir():
        raise IngestError(f"data folder not found: {settings.data_dir}")
    documents = []
    for path in sorted(settings.data_dir.glob("*.md")):
        try:
            meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError) as e:
            raise IngestError(f"could not read {path.name}: {e}") from e
        if not body.strip():
            logger.warning("Skipping empty file %s", path.name)
            continue
        title = next((l[2:].strip() for l in body.splitlines() if l.startswith("# ")), path.stem)
        documents.append(Document(page_content=body, metadata={
            "doc_id": path.stem, "source": path.name, "title": title,
            "category": meta.get("category", "general").lower(),
            "tags": [t.lower() for t in meta.get("tags", [])],
        }))
    if not documents:
        raise IngestError(f"no Markdown files in {settings.data_dir}")
    return documents


# ---------------------------------------------------------------------------
# 2. Chunking (tokens) with section / page tracking
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _encoding():
    import tiktoken
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text))


def build_splitter(settings: Settings, length_function: Callable[[str], int]) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap,
        length_function=length_function, separators=SEPARATORS, keep_separator="start",
    )


def split_document(document: Document, splitter, length_function: Callable[[str], int]) -> List[Document]:
    """Chunks of one document. Each chunk keeps the "## " section it belongs to and its number (page)."""
    headings = [INTRO_SECTION] + HEADING.findall(document.page_content)
    current = INTRO_SECTION
    chunks = []
    for index, text in enumerate(splitter.split_text(document.page_content)):
        match = HEADING.match(text)
        section = match.group(1).strip() if match else current   # a chunk starting mid-section inherits it
        inside = HEADING.findall(text)
        if inside:
            current = inside[-1].strip()
        chunks.append(Document(
            id=f"{document.metadata['doc_id']}#{index:03d}",
            page_content=text,
            metadata={**document.metadata, "section": section,
                      "page": headings.index(section) + 1 if section in headings else 1,
                      "chunk_index": index, "tokens": length_function(text)},
        ))
    return chunks


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 3. Upload: incremental sync with batching and retries
# ---------------------------------------------------------------------------
def existing_metadata(index, namespace: str) -> Dict[str, Dict[str, Any]]:
    """IDs already in the namespace with their metadata (content_hash, embedding_model)."""
    try:
        ids = [item.id for page in index.list(namespace=namespace) for item in page.vectors]
    except NotFoundError:
        return {}
    found = {}
    for start in range(0, len(ids), FETCH_BATCH):
        response = index.fetch(ids=ids[start:start + FETCH_BATCH], namespace=namespace)
        found.update({vid: (vec.metadata or {}) for vid, vec in response.vectors.items()})
    return found


async def upsert_with_retry(index, vectors: List[Tuple[str, List[float], Dict[str, Any]]], namespace: str,
                            batch_size: int, report: IngestReport) -> None:
    """One upsert per batch. Temporary errors are retried with backoff; failed batches are recorded."""
    for start in range(0, len(vectors), batch_size):
        batch = vectors[start:start + batch_size]
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                index.upsert(vectors=batch, namespace=namespace, show_progress=False)
                logger.info("Upserted batch of %d vectors", len(batch))
                break
            except RETRYABLE_ERRORS as e:
                if attempt == MAX_RETRIES:
                    logger.error("Batch starting at %s failed: %s", batch[0][0], e)
                    report.failed_batches.extend(v[0] for v in batch)
                    break
                wait = getattr(e, "retry_after", None) or 2 ** (attempt - 1)
                logger.warning("%s, retrying in %.0f s (attempt %d/%d)", type(e).__name__, wait, attempt, MAX_RETRIES)
                await asyncio.sleep(wait)
            except PineconeError as e:   # permanent (bad dimension, invalid metadata): retrying would not help
                logger.error("Batch starting at %s rejected: %s", batch[0][0], e)
                report.failed_batches.extend(v[0] for v in batch)
                break


async def ingest(settings: Settings, index=None, embeddings: Optional[Embeddings] = None, rebuild: bool = False,
                 length_function: Optional[Callable[[str], int]] = None) -> IngestReport:
    length_function = length_function or count_tokens
    if index is None:
        index = ensure_index(Pinecone(api_key=settings.pinecone_api_key.get_secret_value()), settings)
    embeddings = embeddings or build_embeddings(settings)
    report = IngestReport()

    documents = load_documents(settings)
    splitter = build_splitter(settings, length_function)
    chunks = [c for doc in documents for c in split_document(doc, splitter, length_function)]
    report.documents, report.chunks = len(documents), len(chunks)

    if rebuild:
        try:
            index.delete(delete_all=True, namespace=settings.namespace)
            logger.info("Namespace '%s' emptied (--rebuild)", settings.namespace)
        except NotFoundError:
            pass
    existing = existing_metadata(index, settings.namespace)

    # Only new or changed chunks (or chunks embedded with another model) are embedded and uploaded
    pending = [c for c in chunks
               if existing.get(c.id, {}).get("content_hash") != content_hash(c.page_content)
               or existing.get(c.id, {}).get("embedding_model") != settings.embedding_model]
    report.unchanged = len(chunks) - len(pending)

    if pending:
        vectors_values = embeddings.embed_documents([c.page_content for c in pending])
        # Dimension mismatch check BEFORE touching the index
        wrong = {len(v) for v in vectors_values if len(v) != settings.dimension}
        if wrong:
            raise IngestError(f"the embedding model returned {sorted(wrong)} dimensions, the index expects "
                              f"{settings.dimension}")
        now = int(time.time())
        vectors = []
        for chunk, values in zip(pending, vectors_values):
            metadata = ChunkMetadata(**chunk.metadata, content_hash=content_hash(chunk.page_content),
                                     embedding_model=settings.embedding_model, ingested_at=now,
                                     text=chunk.page_content)
            vectors.append((chunk.id, values, metadata.model_dump()))
        report.embedded = len(vectors)
        await upsert_with_retry(index, vectors, settings.namespace, settings.batch_size, report)

    # Chunks that disappeared (document deleted or shortened) are removed from the index
    stale = sorted(set(existing) - {c.id for c in chunks})
    if stale:
        for start in range(0, len(stale), 1000):
            index.delete(ids=stale[start:start + 1000], namespace=settings.namespace)
        report.deleted = len(stale)
        logger.info("Deleted %d stale vectors: %s", len(stale), stale[:5])
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload data/ to the Pinecone index")
    parser.add_argument("--rebuild", action="store_true", help="empty the namespace and upload everything again")
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("httpx", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        settings = Settings.from_env()
        report = asyncio.run(ingest(settings, rebuild=args.rebuild))
    except (IngestError, IndexConfigError, ValueError, PineconeError) as e:
        logger.error("%s", e)
        return 1
    print(f"\nDocuments: {report.documents} | chunks: {report.chunks} | embedded now: {report.embedded} | "
          f"unchanged: {report.unchanged} | deleted: {report.deleted} | failed: {len(report.failed_batches)}")
    print(f"Index '{settings.index_name}', namespace '{settings.namespace}' "
          f"({settings.embedding_provider.value}/{settings.embedding_model}, {settings.dimension} dims)")
    return 1 if report.failed_batches else 0


if __name__ == "__main__":
    sys.exit(main())
