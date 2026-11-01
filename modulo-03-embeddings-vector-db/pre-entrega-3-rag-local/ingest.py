"""Ingestion: reads the files in data/, splits them into chunks and persists them in a local ChromaDB.

The run is incremental: a manifest (vectorstore/manifest.json) stores the hash of every file and the
embedding model used. Unchanged files are skipped, so a second run computes no embeddings at all.

Usage:
    python ingest.py             # index only what changed
    python ingest.py --rebuild   # delete the collection and index everything again
"""

import argparse
import hashlib
import json
import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from chromadb.errors import ChromaError
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

from config import RAGConfig, build_embeddings

logger = logging.getLogger("rag.ingest")

MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1

# Markdown-aware separators: first split by section, then by paragraph, line, sentence and word
SEPARATORS = ["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]

# Errors of the database layer (same approach as the ChromaDB CRUD exercise)
DATABASE_ERRORS = (ChromaError, ValueError, OSError)


class IngestError(Exception):
    """The documents could not be loaded or persisted."""


class IngestReport(BaseModel):
    added: List[str] = []
    updated: List[str] = []
    removed: List[str] = []
    unchanged: List[str] = []
    rebuilt: bool = False
    chunks_written: int = 0
    total_chunks: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.added or self.updated or self.removed)


# ---------------------------------------------------------------------------
# 1. Loading
# ---------------------------------------------------------------------------
def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_documents(config: RAGConfig) -> List[Document]:
    """One Document per .md/.txt file in data_dir, with its file name and title as metadata."""
    if not config.data_dir.is_dir():
        raise IngestError(f"data folder not found: {config.data_dir}")
    paths = sorted({p for pattern in config.file_patterns for p in config.data_dir.glob(pattern)})
    documents = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError) as e:
            raise IngestError(f"could not read {path.name}: {e}") from e
        if not text:
            logger.warning("Skipping empty file %s", path.name)
            continue
        # Title: first Markdown heading, or the file name
        title = next((line.lstrip("# ").strip() for line in text.splitlines() if line.startswith("# ")), path.stem)
        documents.append(Document(page_content=text, metadata={"source": path.name, "title": title}))
    if not documents:
        raise IngestError(f"no .md or .txt files with content in {config.data_dir}")
    return documents


# ---------------------------------------------------------------------------
# 2. Chunking
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _encoding():
    import tiktoken
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text))


def build_splitter(config: RAGConfig, length_function: Optional[Callable[[str], int]] = None):
    """RecursiveCharacterTextSplitter measuring the chunk size in tokens instead of characters."""
    return RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        length_function=length_function or count_tokens,
        separators=SEPARATORS,
        keep_separator="start",   # a "## Heading" stays at the start of its chunk
        strip_whitespace=True,
    )


def split_document(document: Document, splitter, length_function: Callable[[str], int]) -> List[Document]:
    """Chunks of one file, with deterministic IDs (<file>::<index>) and their size in tokens."""
    chunks = splitter.split_documents([document])
    for index, chunk in enumerate(chunks):
        chunk.id = f"{document.metadata['source']}::{index:03d}"
        chunk.metadata.update({"chunk_index": index, "chunk_id": chunk.id, "tokens": length_function(chunk.page_content)})
    return chunks


# ---------------------------------------------------------------------------
# 3. Persistence (ChromaDB + manifest)
# ---------------------------------------------------------------------------
def open_vectorstore(config: RAGConfig, embeddings: Embeddings) -> Chroma:
    """LangChain wrapper over a local PersistentClient. The collection uses cosine distance."""
    client = chromadb.PersistentClient(path=str(config.persist_dir), settings=Settings(anonymized_telemetry=False))
    return Chroma(
        client=client,
        collection_name=config.collection_name,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )


def manifest_path(config: RAGConfig) -> Path:
    return config.persist_dir / MANIFEST_NAME


def load_manifest(config: RAGConfig) -> Optional[dict]:
    path = manifest_path(config)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Unreadable manifest (%s): the index will be rebuilt", e)
        return None


def index_settings(config: RAGConfig) -> dict:
    """Everything that, if changed, makes the stored vectors unusable."""
    return {
        "manifest_version": MANIFEST_VERSION,
        "embedding_provider": config.embedding_provider.value,
        "embedding_model": config.embedding_model,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
    }


def save_manifest(config: RAGConfig, files: Dict[str, dict]) -> None:
    content = {**index_settings(config), "files": files}
    tmp = manifest_path(config).with_suffix(".tmp")
    tmp.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(manifest_path(config))   # atomic: a crash never leaves a half-written manifest


def ingest(config: RAGConfig, rebuild: bool = False, embeddings: Optional[Embeddings] = None,
           length_function: Optional[Callable[[str], int]] = None) -> IngestReport:
    """Loads, chunks and persists the documents, re-indexing only the files that changed."""
    length_function = length_function or count_tokens
    documents = load_documents(config)
    config.persist_dir.mkdir(parents=True, exist_ok=True)
    report = IngestReport()

    try:
        store = open_vectorstore(config, embeddings or build_embeddings(config))
        manifest = load_manifest(config)

        # Rebuild when asked, when there is no manifest, or when the embedding model / chunking changed:
        # vectors from another model live in a different space and would make the search meaningless.
        # It is also rebuilt if the collection does not hold the chunks the manifest lists (e.g. deleted by hand).
        settings_changed = manifest is not None and {k: manifest.get(k) for k in index_settings(config)} != index_settings(config)
        out_of_sync = manifest is not None and store._collection.count() != sum(
            len(f["chunk_ids"]) for f in manifest.get("files", {}).values())
        if manifest is None or rebuild or settings_changed or out_of_sync:
            if settings_changed:
                logger.warning("Embedding model or chunking settings changed: rebuilding the whole index")
            elif out_of_sync:
                logger.warning("The collection does not match the manifest: rebuilding the whole index")
            store.reset_collection()
            manifest, report.rebuilt = {"files": {}}, True

        indexed: Dict[str, dict] = manifest.get("files", {})
        current = {doc.metadata["source"]: doc for doc in documents}
        splitter = build_splitter(config, length_function)
        files: Dict[str, dict] = {}

        for name, doc in current.items():
            digest = file_hash(config.data_dir / name)
            previous = indexed.get(name)
            if previous and previous["sha256"] == digest:
                report.unchanged.append(name)
                files[name] = previous
                continue
            if previous:                                     # changed file: remove its old chunks first
                store.delete(ids=previous["chunk_ids"])
                report.updated.append(name)
            else:
                report.added.append(name)
            chunks = split_document(doc, splitter, length_function)
            store.add_documents(chunks, ids=[c.id for c in chunks])
            report.chunks_written += len(chunks)
            files[name] = {"sha256": digest, "chunk_ids": [c.id for c in chunks],
                           "tokens": [c.metadata["tokens"] for c in chunks]}
            logger.info("Indexed %s: %d chunks (%s tokens)", name, len(chunks), files[name]["tokens"])

        for name in set(indexed) - set(current):              # file deleted from data/
            store.delete(ids=indexed[name]["chunk_ids"])
            report.removed.append(name)

        save_manifest(config, files)
        report.total_chunks = store._collection.count()
    except DATABASE_ERRORS as e:
        logger.error("Ingestion failed: %s: %s", type(e).__name__, e)
        raise IngestError(f"ingestion failed: {e}") from e

    if not report.changed and not report.rebuilt:
        logger.info("Index up to date (%d chunks): nothing to embed", report.total_chunks)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Index the files of data/ into the local ChromaDB")
    parser.add_argument("--rebuild", action="store_true", help="delete the collection and index everything again")
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                        datefmt="%H:%M:%S")
    try:
        config = RAGConfig.from_env(require_llm=False)
        report = ingest(config, rebuild=args.rebuild)
    except (IngestError, ValueError) as e:
        logger.error("%s", e)
        return 1
    print(f"\nAdded: {report.added}\nUpdated: {report.updated}\nRemoved: {report.removed}\n"
          f"Unchanged: {report.unchanged}\nChunks embedded now: {report.chunks_written}\n"
          f"Chunks in the collection: {report.total_chunks}  ({config.embedding_provider.value}/{config.embedding_model})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
