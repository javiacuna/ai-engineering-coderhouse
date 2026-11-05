import asyncio
import json
from collections import Counter

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from pinecone.errors import ServiceError, ValidationError as PineconeValidationError
from pydantic import ValidationError

from conftest import PROJECT_DIR, FakeEmbeddings, FakeIndex, word_count
from evaluate import GoldenItem, evaluate, load_golden_set, score_question
from ingest import ChunkMetadata, IngestError, build_splitter, ingest, load_documents, parse_front_matter, split_document
from rag_system import IndexNotReadyError, RAGSystem, tokenize


def run_ingest(settings, index, embeddings, **kwargs):
    return asyncio.run(ingest(settings, index=index, embeddings=embeddings, length_function=word_count, **kwargs))


def stored(index, settings):
    return index.namespaces[settings.namespace]


# ---------------------------------------------------------------- loading and chunking
def test_front_matter():
    meta, body = parse_front_matter("---\ncategory: colas\ntags: [queue, join]\n---\n# Título\ntexto")
    assert meta == {"category": "colas", "tags": ["queue", "join"]}
    assert body.startswith("# Título")


def test_documents_have_category_and_tags(settings):
    docs = load_documents(settings)
    assert len(docs) == 6
    first = docs[0].metadata
    assert first["doc_id"] == "01-corutinas-y-event-loop" and first["category"] == "fundamentos"
    assert "asyncio.run" in first["tags"]


def test_chunks_track_section_and_page(settings):
    doc = load_documents(settings)[4]   # colas
    chunks = split_document(doc, build_splitter(settings, word_count), word_count)
    assert len(chunks) >= 2
    assert all(c.metadata["tokens"] <= settings.chunk_size for c in chunks)
    assert chunks[0].id == "05-colas-productor-consumidor#000"
    assert chunks[0].metadata["section"] == "Introducción" and chunks[0].metadata["page"] == 1
    sections = [c.metadata["section"] for c in chunks]
    assert "task_done y join" in sections
    assert [c.metadata["page"] for c in chunks] == sorted(c.metadata["page"] for c in chunks)


# ---------------------------------------------------------------- ingestion
def test_metadata_keeps_text_source_and_page(settings, index, embeddings):
    report = run_ingest(settings, index, embeddings)
    assert report.embedded == report.chunks and not report.failed_batches
    values, metadata = stored(index, settings)["04-sincronizacion#000"]
    assert len(values) == 1536
    assert metadata["source"] == "04-sincronizacion.md" and metadata["category"] == "sincronizacion"
    assert metadata["text"].startswith("# Primitivas de sincronización")
    assert set(metadata) == set(ChunkMetadata.model_fields)


def test_metadata_schema_is_strict():
    with pytest.raises(ValidationError):
        ChunkMetadata(doc_id="d", source="d.md", title="t", category="c", tags=[], section="s", page=1,
                      chunk_index=0, tokens=1, content_hash="h", embedding_model="m", ingested_at=0, text="x",
                      date_created="2026-01-01")   # unknown field -> schema drift


def test_second_ingest_embeds_nothing(settings, index, embeddings):
    first = run_ingest(settings, index, embeddings)
    calls = embeddings.embedded_texts
    second = run_ingest(settings, index, embeddings)
    assert second.embedded == 0 and second.unchanged == first.chunks
    assert embeddings.embedded_texts == calls


def test_changed_and_deleted_documents(settings, index, embeddings):
    run_ingest(settings, index, embeddings)
    (settings.data_dir / "04-sincronizacion.md").write_text(
        "---\ncategory: sincronizacion\ntags: [lock]\n---\n# Sincronización\n\nVersión corta.", encoding="utf-8")
    (settings.data_dir / "06-codigo-bloqueante.md").unlink()
    report = run_ingest(settings, index, embeddings)
    ids = set(stored(index, settings))
    assert report.embedded == 1                                   # only the new, shorter document
    assert "04-sincronizacion#000" in ids and "04-sincronizacion#001" not in ids
    assert not any(i.startswith("06-") for i in ids)
    assert report.deleted >= 2


def test_new_embedding_model_reembeds_everything(settings, index, embeddings):
    first = run_ingest(settings, index, embeddings)
    other = settings.model_copy(update={"embedding_model": "another-model"})
    assert run_ingest(other, index, embeddings).embedded == first.chunks


def test_dimension_mismatch_is_caught_before_upload(settings, index):
    with pytest.raises(IngestError, match="768"):
        run_ingest(settings, index, FakeEmbeddings(dimension=768))
    assert index.upsert_calls == 0


def test_batches_and_retries(settings, embeddings):
    index = FakeIndex(fail_upserts=[ServiceError("unavailable", 503)])
    small_batches = settings.model_copy(update={"batch_size": 5})
    report = run_ingest(small_batches, index, embeddings)
    assert not report.failed_batches
    assert len(stored(index, settings)) == report.chunks
    assert index.upsert_calls == -(-report.chunks // 5) + 1        # one call per batch + the retried one


def test_permanent_error_is_recorded(settings, embeddings):
    index = FakeIndex(fail_upserts=[PineconeValidationError("bad metadata")])
    report = run_ingest(settings.model_copy(update={"batch_size": 5}), index, embeddings)
    assert len(report.failed_batches) == 5                         # the first batch, not retried
    assert len(stored(index, settings)) == report.chunks - 5


# ---------------------------------------------------------------- retrieval
def test_rag_system_requires_an_index(settings, index, embeddings):
    with pytest.raises(IndexNotReadyError, match="ingest.py"):
        RAGSystem(settings, index=index, embeddings=embeddings)


def test_rag_system_rejects_another_embedding_model(settings, index, embeddings):
    run_ingest(settings, index, embeddings)
    with pytest.raises(IndexNotReadyError, match="--rebuild"):
        RAGSystem(settings.model_copy(update={"embedding_model": "other"}), index=index, embeddings=embeddings)


def test_tokenizer_keeps_identifiers():
    assert tokenize("¿Para qué sirven task_done() y join()?") == ["sirven", "task_done", "join"]
    assert tokenize("Cancelación") == ["cancelacion"]


def test_hybrid_returns_top_k_unique_chunks(settings, index, embeddings):
    run_ingest(settings, index, embeddings)
    rag = RAGSystem(settings, index=index, embeddings=embeddings)
    for mode in ("hybrid", "vector", "bm25"):
        docs = rag.retrieve("¿Para qué sirven task_done() y join() en una asyncio.Queue?", mode=mode)
        ids = [d.metadata["chunk_id"] for d in docs]
        assert len(ids) == settings.top_k and len(set(ids)) == len(ids)
    bm25_first = rag.retrieve("task_done join", mode="bm25")[0]
    assert bm25_first.metadata["doc_id"] == "05-colas-productor-consumidor"


def test_answer_uses_the_retrieved_context(settings, index, embeddings):
    run_ingest(settings, index, embeddings)
    rag = RAGSystem(settings, index=index, embeddings=embeddings)
    result = asyncio.run(rag.answer("¿Qué es task_done?", llm=FakeListChatModel(responses=["Resta uno [1]."])))
    assert result.answer == "Resta uno [1]." and len(result.sources) == settings.top_k


def test_empty_query(settings, index, embeddings):
    run_ingest(settings, index, embeddings)
    with pytest.raises(ValueError, match="empty"):
        RAGSystem(settings, index=index, embeddings=embeddings).retrieve("  ")


# ---------------------------------------------------------------- evaluation
def test_metrics():
    item = GoldenItem(pregunta="pregunta", documento_id_esperado="a", documentos_relacionados=["b"])
    result = score_question(item, ["c", "a", "b", "a", "c"], k=5, chunks_per_doc=Counter({"a": 2, "b": 1, "c": 9}))
    assert result.recall == 1.0
    assert result.precision == pytest.approx(3 / 5)
    assert result.precision_ceiling == pytest.approx(3 / 5)      # a + b only have 3 chunks
    assert result.reciprocal_rank == pytest.approx(1 / 2)
    miss = score_question(item, ["c"] * 5, k=5, chunks_per_doc=Counter({"a": 2, "b": 1}))
    assert miss.recall == 0 and miss.reciprocal_rank == 0


def test_golden_set_references_real_documents(tmp_path):
    doc_ids = {p.stem for p in (PROJECT_DIR / "data").glob("*.md")}
    items = load_golden_set(PROJECT_DIR / "golden_set.json", doc_ids)
    assert len(items) == 5
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"pregunta": "algo raro", "documento_id_esperado": "no-existe"}]))
    with pytest.raises(ValueError, match="no-existe"):
        load_golden_set(bad, doc_ids)


def test_evaluation_runs_for_the_three_retrievers(settings, index, embeddings):
    run_ingest(settings, index, embeddings)
    rag = RAGSystem(settings, index=index, embeddings=embeddings)
    items = load_golden_set(PROJECT_DIR / "golden_set.json", {d.metadata["doc_id"] for d in rag.corpus})
    results = asyncio.run(evaluate(rag, items))
    assert set(results) == {"vector", "bm25", "hybrid"}
    assert all(len(rows) == 5 for rows in results.values())
    assert sum(r.recall for r in results["bm25"]) >= 4           # exact API names: BM25's strength
