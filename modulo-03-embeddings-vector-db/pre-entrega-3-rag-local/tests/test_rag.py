import asyncio
import json

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from pydantic import ValidationError

from config import RAGConfig
from conftest import FakeEmbeddings, word_count
from ingest import IngestError, build_splitter, ingest, load_documents, manifest_path, split_document
from rag import IndexNotReadyError, RAGError, Retriever, build_rag_chain, format_context, get_rag_response
from schemas import NOT_FOUND_ANSWER, LLMAnswer


def llm_json(answer: str, fragment_ids=()) -> str:
    return json.dumps({"answer": answer, "fragment_ids": list(fragment_ids)}, ensure_ascii=False)


def make_chain(config, embeddings, responses):
    ingest(config, embeddings=embeddings, length_function=word_count)
    retriever = Retriever(config, embeddings=embeddings)
    return build_rag_chain(config, retriever=retriever, llm=FakeListChatModel(responses=responses), retry_backoff=False)


# ---------------------------------------------------------------- configuration
def test_config_enforces_the_assignment_limits():
    with pytest.raises(ValidationError):
        RAGConfig(chunk_size=300)               # at least 500 tokens
    with pytest.raises(ValidationError):
        RAGConfig(top_k=10)                     # 3 to 5 fragments
    with pytest.raises(ValidationError):
        RAGConfig(chunk_size=500, chunk_overlap=500)


def test_from_env_requires_the_keys(monkeypatch):
    for var in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        RAGConfig.from_env()


def test_from_env_reads_the_repo_variables(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test")
    monkeypatch.setenv("RAG_TOP_K", "3")
    for var in ("LLM_MODEL", "EMBEDDING_MODEL"):
        monkeypatch.delenv(var, raising=False)
    config = RAGConfig.from_env()
    assert config.llm_model == "gemini-test"
    assert config.embedding_model == "gemini-embedding-001"
    assert config.top_k == 3
    assert "test-key" not in repr(config)        # keys are SecretStr


# ---------------------------------------------------------------- ingestion
def test_documents_are_loaded_with_metadata(config):
    documents = load_documents(config)
    assert [d.metadata["source"] for d in documents] == [
        "01-arquitectura.md", "02-despliegues.md", "03-incidentes-y-guardias.md", "04-seguridad-y-accesos.md"]
    assert documents[0].metadata["title"] == "Arquitectura de la plataforma Nimbus Envíos"


def test_chunks_respect_size_overlap_and_ids(config):
    splitter = build_splitter(config, word_count)
    document = load_documents(config)[2]
    chunks = split_document(document, splitter, word_count)
    assert len(chunks) >= 2
    assert all(c.metadata["tokens"] <= config.chunk_size for c in chunks)
    assert [c.id for c in chunks] == [f"03-incidentes-y-guardias.md::{i:03d}" for i in range(len(chunks))]


def test_long_sections_are_split_with_overlap(config):
    # The overlap only applies when a section has to be cut inside; clean "## " boundaries do not repeat text
    text = " ".join(f"Oración número {i} del documento de prueba." for i in range(300))
    document = Document(page_content=text, metadata={"source": "long.md", "title": "long"})
    chunks = split_document(document, build_splitter(config, word_count), word_count)
    assert len(chunks) >= 3
    tail = " ".join(chunks[0].page_content.split()[-10:])
    assert tail in chunks[1].page_content


def test_second_ingest_embeds_nothing(config, embeddings):
    first = ingest(config, embeddings=embeddings, length_function=word_count)
    assert first.rebuilt and len(first.added) == 4
    embedded = embeddings.embedded_texts

    second = ingest(config, embeddings=embeddings, length_function=word_count)
    assert not second.changed and len(second.unchanged) == 4
    assert embeddings.embedded_texts == embedded           # no embedding was computed again
    assert second.total_chunks == first.total_chunks


def test_only_changed_files_are_reindexed(config, embeddings):
    first = ingest(config, embeddings=embeddings, length_function=word_count)
    (config.data_dir / "02-despliegues.md").write_text("# Despliegues\n\nAhora se despliega solo los martes.", encoding="utf-8")
    (config.data_dir / "04-seguridad-y-accesos.md").unlink()
    (config.data_dir / "05-nuevo.txt").write_text("Documento nuevo sobre el panel de comercios.", encoding="utf-8")

    report = ingest(config, embeddings=embeddings, length_function=word_count)
    assert report.updated == ["02-despliegues.md"]
    assert report.removed == ["04-seguridad-y-accesos.md"]
    assert report.added == ["05-nuevo.txt"]
    assert report.chunks_written == 2                      # one chunk for each changed file
    manifest = json.loads(manifest_path(config).read_text())
    assert report.total_chunks == sum(len(f["chunk_ids"]) for f in manifest["files"].values())
    assert report.total_chunks < first.total_chunks


def test_changing_the_embedding_model_rebuilds_the_index(config, embeddings):
    ingest(config, embeddings=embeddings, length_function=word_count)
    other = config.model_copy(update={"embedding_model": "another-model"})
    report = ingest(other, embeddings=embeddings, length_function=word_count)
    assert report.rebuilt and len(report.added) == 4


def test_missing_data_folder(config, embeddings):
    broken = config.model_copy(update={"data_dir": config.data_dir / "does-not-exist"})
    with pytest.raises(IngestError, match="not found"):
        ingest(broken, embeddings=embeddings, length_function=word_count)


# ---------------------------------------------------------------- retriever
def test_retriever_requires_an_index(config, embeddings):
    with pytest.raises(IndexNotReadyError, match="ingest.py"):
        Retriever(config, embeddings=embeddings)


def test_retriever_rejects_a_different_embedding_model(config, embeddings):
    ingest(config, embeddings=embeddings, length_function=word_count)
    other = config.model_copy(update={"embedding_model": "another-model"})
    with pytest.raises(IndexNotReadyError, match="--rebuild"):
        Retriever(other, embeddings=embeddings)


def test_retriever_returns_top_k_relevant_fragments(config, embeddings):
    ingest(config, embeddings=embeddings, length_function=word_count)
    fragments = asyncio.run(Retriever(config, embeddings=embeddings).aretrieve("SEV1 guardia acuse escalamiento PagerDuty"))
    assert len(fragments) == config.top_k
    assert fragments[0].metadata["source"] == "03-incidentes-y-guardias.md"
    distances = [f.metadata["distance"] for f in fragments]
    assert distances == sorted(distances)
    assert format_context(fragments).startswith("[1] (fuente: 03-incidentes-y-guardias.md")


# ---------------------------------------------------------------- generation
def test_grounded_answer_with_references(config, embeddings):
    chain = make_chain(config, embeddings, [llm_json("La guardia tiene 5 minutos para el acuse.", [1, 1, 9])])
    response = asyncio.run(get_rag_response("¿Cuánto tiempo tiene la guardia para un SEV1?", chain=chain))
    assert response.answered
    assert [r.fragment for r in response.references] == [1]   # duplicate removed, invented [9] discarded
    assert response.references[0].source.endswith(".md")


def test_trick_question_answers_no_lo_se(config, embeddings):
    chain = make_chain(config, embeddings, [llm_json("No lo sé.", [2])])
    response = asyncio.run(get_rag_response("¿Quién ganó el Mundial de 2022?", chain=chain))
    assert response.answer == NOT_FOUND_ANSWER
    assert not response.answered and response.references == []


def test_malformed_output_is_retried(config, embeddings):
    chain = make_chain(config, embeddings, ["esto no es JSON", llm_json("Los viernes no se despliega.", [1])])
    response = asyncio.run(get_rag_response("¿Se puede desplegar un viernes?", chain=chain))
    assert response.answered


def test_gives_up_after_max_attempts(config, embeddings):
    chain = make_chain(config, embeddings, ["mal", "mal", "mal"])
    with pytest.raises(RAGError, match="valid answer"):
        asyncio.run(get_rag_response("¿Se puede desplegar un viernes?", chain=chain))


def test_no_relevant_fragments_skips_the_llm(config, embeddings):
    strict = config.model_copy(update={"max_distance": 1e-9})
    llm = FakeListChatModel(responses=["the LLM must not be called"])
    ingest(strict, embeddings=embeddings, length_function=word_count)
    chain = build_rag_chain(strict, retriever=Retriever(strict, embeddings=embeddings), llm=llm)
    response = asyncio.run(get_rag_response("pregunta sin relación", chain=chain))
    assert response.answer == NOT_FOUND_ANSWER and llm.i == 0


def test_empty_question():
    with pytest.raises(RAGError, match="empty"):
        asyncio.run(get_rag_response("   "))


def test_not_found_variants_are_normalized():
    for text in ("No lo sé", "no lo se.", "No lo sé, el contexto no lo menciona"):
        answer = LLMAnswer(answer=text, fragment_ids=[1])
        assert answer.answer == NOT_FOUND_ANSWER and answer.fragment_ids == []
