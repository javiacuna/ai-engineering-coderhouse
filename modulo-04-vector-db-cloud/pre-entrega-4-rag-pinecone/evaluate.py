"""Retrieval evaluation with a golden set: Precision@k, Recall@k and MRR for vector, BM25 and hybrid.

Definitions (k = 5, per question, then averaged):
- Recall@k: 1 if a chunk of the expected document is among the k retrieved, otherwise 0
  (with one expected document per question, it is the "hit rate").
- Precision@k: share of the k retrieved chunks that come from a useful document (the expected one or
  one listed in "documentos_relacionados").
- MRR: 1 / position of the first chunk of the expected document (0 if it does not appear).
- Precision ceiling: the best Precision@k possible for that question. If the useful documents only
  have 2 chunks in total, at most 2 of the 5 results can be useful: the ceiling is 0.40.

Usage:
    python evaluate.py
"""

import asyncio
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from pinecone.errors import PineconeError
from pydantic import BaseModel, Field

from config import BASE_DIR, Settings
from rag_system import IndexNotReadyError, RAGSystem

logger = logging.getLogger("rag.evaluate")

GOLDEN_SET_PATH = BASE_DIR / "golden_set.json"
RESULTS_PATH = BASE_DIR / "results" / "evaluation.json"
MODES = ("vector", "bm25", "hybrid")


class GoldenItem(BaseModel):
    pregunta: str = Field(min_length=5)
    documento_id_esperado: str
    documentos_relacionados: List[str] = []
    tipo: Optional[str] = None


class QuestionResult(BaseModel):
    question: str
    expected: str
    retrieved_doc_ids: List[str]
    recall: float
    precision: float
    precision_ceiling: float
    reciprocal_rank: float


def load_golden_set(path: Path, known_doc_ids: set) -> List[GoldenItem]:
    items = [GoldenItem(**item) for item in json.loads(path.read_text(encoding="utf-8"))]
    unknown = {i.documento_id_esperado for i in items} | {d for i in items for d in i.documentos_relacionados}
    unknown -= known_doc_ids
    if unknown:   # a typo in the golden set would silently count as a miss
        raise ValueError(f"golden set references documents that are not in the index: {sorted(unknown)}")
    return items


def score_question(item: GoldenItem, retrieved_doc_ids: List[str], k: int, chunks_per_doc: Counter) -> QuestionResult:
    useful = {item.documento_id_esperado, *item.documentos_relacionados}
    hits = [doc_id in useful for doc_id in retrieved_doc_ids]
    first = next((n for n, d in enumerate(retrieved_doc_ids, start=1) if d == item.documento_id_esperado), None)
    return QuestionResult(
        question=item.pregunta,
        expected=item.documento_id_esperado,
        retrieved_doc_ids=retrieved_doc_ids,
        recall=1.0 if first else 0.0,
        precision=sum(hits) / k,
        precision_ceiling=min(k, sum(chunks_per_doc[d] for d in useful)) / k,
        reciprocal_rank=1 / first if first else 0.0,
    )


async def evaluate(rag: RAGSystem, items: List[GoldenItem]) -> Dict[str, List[QuestionResult]]:
    k = rag.settings.top_k
    chunks_per_doc = Counter(d.metadata["doc_id"] for d in rag.corpus)
    results = {}
    for mode in MODES:
        retrieved = await asyncio.gather(*(rag.aretrieve(item.pregunta, mode=mode) for item in items))
        results[mode] = [score_question(item, [d.metadata["doc_id"] for d in docs], k, chunks_per_doc)
                         for item, docs in zip(items, retrieved)]
    return results


def averages(rows: List[QuestionResult]) -> Dict[str, float]:
    n = len(rows)
    return {
        "recall": sum(r.recall for r in rows) / n,
        "precision": sum(r.precision for r in rows) / n,
        "precision_ceiling": sum(r.precision_ceiling for r in rows) / n,
        "mrr": sum(r.reciprocal_rank for r in rows) / n,
    }


def print_report(results: Dict[str, List[QuestionResult]], k: int) -> None:
    print(f"\n=== Evaluación de recuperación (k={k}, {len(results['hybrid'])} preguntas) ===\n")
    print(f"{'Retriever':<10} {'Recall@' + str(k):>9} {'Precision@' + str(k):>12} {'Techo P@' + str(k):>10} {'MRR':>6}")
    for mode in MODES:
        a = averages(results[mode])
        print(f"{mode:<10} {a['recall']:>9.2f} {a['precision']:>12.2f} {a['precision_ceiling']:>10.2f} {a['mrr']:>6.2f}")

    print("\nDetalle del retriever híbrido:")
    for n, r in enumerate(results["hybrid"], start=1):
        mark = "OK " if r.recall else "MISS"
        print(f"  {n}. [{mark}] P@{k}={r.precision:.2f} (techo {r.precision_ceiling:.2f}) RR={r.reciprocal_rank:.2f}  "
              f"{r.question}")
        print(f"     esperado: {r.expected} | recuperados: {r.retrieved_doc_ids}")


def main() -> int:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("httpx", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        settings = Settings.from_env()
        rag = RAGSystem(settings)
        items = load_golden_set(GOLDEN_SET_PATH, {d.metadata["doc_id"] for d in rag.corpus})
        results = asyncio.run(evaluate(rag, items))
    except (IndexNotReadyError, ValueError, PineconeError) as e:
        logger.error("%s", e)
        return 1

    print_report(results, settings.top_k)
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(
        {"k": settings.top_k, "embedding_model": settings.embedding_model, "vector_weight": settings.vector_weight,
         "summary": {m: averages(rows) for m, rows in results.items()},
         "questions": {m: [r.model_dump() for r in rows] for m, rows in results.items()}},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultados completos en {RESULTS_PATH.relative_to(BASE_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
