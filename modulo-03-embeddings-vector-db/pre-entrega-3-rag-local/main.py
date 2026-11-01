"""End-to-end test of the RAG system: incremental ingestion + answerable questions + trick questions.

Usage:
    python main.py                       # runs the test questions
    python main.py "your own question"   # asks a single question
"""

import asyncio
import logging
import sys

from dotenv import load_dotenv

from config import RAGConfig
from ingest import IngestError, ingest
from rag import RAGError, get_rag_response

# (question, is the answer in the documents?)
TEST_QUESTIONS = [
    ("¿Cuánto tiempo tiene la guardia para responder a un incidente SEV1 y a quién hay que avisar?", True),
    ("¿Se puede hacer un deploy a producción un viernes?", True),
    # Trick question 1: on topic (Nimbus infrastructure) but the documents never mention it
    ("¿Cuál es el presupuesto mensual de AWS de Nimbus?", False),
    # Trick question 2: the model knows the answer, but it is not in the documents
    ("¿Quién ganó el Mundial de fútbol de 2022?", False),
]


def print_response(response, expected_in_docs=None) -> bool:
    print(f"\n❓ {response.question}")
    print(f"💬 {response.answer}")
    for ref in response.references:
        print(f"   📄 [{ref.fragment}] {ref.source} ({ref.chunk_id}, distancia {ref.distance})")
    ok = True
    if expected_in_docs is not None:
        ok = response.answered == expected_in_docs
        verdict = "OK" if ok else ("FALLO: alucinó una respuesta" if response.answered else "FALLO: no respondió")
        print(f"   → esperado {'respuesta' if expected_in_docs else 'No lo sé'}: {verdict}")
    print(response.model_dump_json(indent=2))
    return ok


async def main() -> int:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                        datefmt="%H:%M:%S")
    # Quiet the HTTP client logs
    for noisy in ("httpx", "google_genai", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    try:
        # Incremental: if data/ did not change since the last run, nothing is embedded again
        report = ingest(RAGConfig.from_env(require_llm=False))
        print(f"Índice: {report.total_chunks} chunks (nuevos: {report.added or '-'}, "
              f"actualizados: {report.updated or '-'}, sin cambios: {len(report.unchanged)})")

        if len(sys.argv) > 1:
            print_response(await get_rag_response(" ".join(sys.argv[1:])))
            return 0

        # The questions run concurrently: get_rag_response is async end to end
        responses = await asyncio.gather(*(get_rag_response(q) for q, _ in TEST_QUESTIONS))
    except (IngestError, RAGError, ValueError) as e:
        logging.getLogger("rag").error("%s", e)
        return 1

    results = [print_response(r, expected) for r, (_, expected) in zip(responses, TEST_QUESTIONS)]
    print(f"\nResultado: {sum(results)}/{len(results)} pruebas OK")
    return 0 if all(results) else 2


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
