"""
Module 3 exercise - Embeddings and similarity (Coderhouse AI Engineering) - v2

Checks whether a search tells semantic relevance apart from keyword matching.
Compares, over a labeled evaluation set and two queries (one affirmative, one negated):
  1. Keywords (TF-IDF) + cosine similarity.
  2. Gemini embeddings + cosine similarity (several models / dimensions).
  3. Re-ranker: an LLM that reads query and document TOGETHER and scores relevance.
Metrics: precision@k, calibrated threshold and separation margin.

Usage:
  python semantic_similarity.py            # everything (needs GOOGLE_API_KEY and GEMINI_MODEL in .env)
  python semantic_similarity.py --no-api   # TF-IDF only, no API calls
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# ---------------------------------------------------------------------------
# 1. Evaluation set (the texts are in Spanish on purpose: the experiment is in Spanish)
#    R = describes deploying microservices with different words
#    I = irrelevant (keyword traps or nearby-domain traps)
#    C = contradicts / negates deploying microservices
# ---------------------------------------------------------------------------
DOCUMENTS = {
    "R1": "Publicamos cada componente del backend como un contenedor independiente en el clúster de Kubernetes.",
    "R2": "La aplicación está dividida en servicios pequeños que se ponen en producción por separado.",
    "R3": "Cada módulo de la plataforma se empaqueta con Docker y se libera sin afectar al resto del sistema.",
    "R4": "Nuestro pipeline de CI/CD lleva automáticamente cada API autónoma desde el repositorio hasta los servidores.",
    "R5": "En lugar de un monolito, entregamos piezas de software desacopladas que escalan cada una por su cuenta.",
    "I1": "Contratamos un servicio de micro limpieza con despliegue de personal en todo el edificio.",
    "I2": "Refactorizamos las funciones del backend para que el código sea más fácil de leer.",
    "I3": "El equipo de diseño actualizó la paleta de colores de la aplicación móvil.",
    "I4": "Los servicios de streaming publican sus series nuevas cada viernes en distintas plataformas.",
    "C1": "No hacemos despliegue de microservicios: todo el sistema es un único monolito.",
    "C2": "Volvimos a un monolito porque mantener servicios separados en producción era demasiado costoso.",
    "C3": "Desistimos de partir la aplicación: seguimos publicando un único ejecutable con todo el sistema.",
}
IDS = list(DOCUMENTS)
TEXTS = list(DOCUMENTS.values())

# For each query, which documents are the correct answer
QUERIES = {
    "affirmative": {
        "text": "despliegue de microservicios",
        "relevant": ["R1", "R2", "R3", "R4", "R5"],
    },
    "negated": {
        "text": "aplicación monolítica sin microservicios",
        "relevant": ["C1", "C2", "C3"],
    },
}

# Spanish stop words: they carry no meaning and only add noise to keyword matching
STOPWORDS_ES = ["de", "la", "el", "en", "un", "una", "y", "que", "se", "con", "por", "del", "los",
                "las", "al", "a", "su", "sus", "cada", "es", "como", "sin", "hasta", "desde", "no",
                "para", "era", "todo", "muy"]

K_VALUES = (3, 5)


# ---------------------------------------------------------------------------
# 2. Metrics
# ---------------------------------------------------------------------------
def manual_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """cos(θ) = (A · B) / (‖A‖ · ‖B‖)"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def evaluate(scores: dict[str, float], relevant: list[str]) -> dict:
    """Ranking, precision@k, margin and best threshold for one method and one query."""
    ranking = sorted(scores, key=scores.get, reverse=True)
    rel = [scores[i] for i in IDS if i in relevant]
    non_rel = [scores[i] for i in IDS if i not in relevant]

    # Only documents with score > 0 count as "retrieved" (TF-IDF returns nothing without matches)
    retrieved = [i for i in ranking if scores[i] > 0]
    precision = {f"p@{k}": round(sum(i in relevant for i in retrieved[:k]) / k, 3) for k in K_VALUES}

    # Margin: worst relevant - best non-relevant. Positive = some threshold separates them perfectly.
    margin = min(rel) - max(non_rel)

    # Calibrated threshold: the cut-off that maximizes accuracy (ties -> higher threshold)
    candidates = sorted(set(scores.values()))
    best = max(
        candidates,
        key=lambda t: (sum((scores[i] >= t) == (i in relevant) for i in IDS), t),
    )
    hits = sum((scores[i] >= best) == (i in relevant) for i in IDS)

    return {
        "scores": {i: round(scores[i], 4) for i in IDS},
        "ranking": ranking,
        **precision,
        "margin": round(margin, 4),
        "threshold": round(best, 4),
        "threshold_accuracy": round(hits / len(IDS), 3),
        "group_mean": {
            g: round(float(np.mean([scores[i] for i in IDS if i.startswith(g)])), 4) for g in "RIC"
        },
    }


# ---------------------------------------------------------------------------
# 3. Method 1: keywords (TF-IDF)
# ---------------------------------------------------------------------------
def tfidf_method() -> dict:
    vectorizer = TfidfVectorizer(stop_words=STOPWORDS_ES)
    queries = [q["text"] for q in QUERIES.values()]
    matrix = vectorizer.fit_transform(queries + TEXTS).toarray()
    query_vecs, doc_vecs = matrix[: len(queries)], matrix[len(queries):]
    sims = cosine_similarity(query_vecs, doc_vecs)              # (n_queries x n_docs)
    return {
        name: evaluate(dict(zip(IDS, map(float, row))), q["relevant"])
        for (name, q), row in zip(QUERIES.items(), sims)
    }


# ---------------------------------------------------------------------------
# 4. Method 2: embeddings (one or more models)
# ---------------------------------------------------------------------------
def embeddings_method(model: str, dimensions: int | None = None) -> dict:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    emb = GoogleGenerativeAIEmbeddings(model=model, output_dimensionality=dimensions)
    doc_vecs = np.array(emb.embed_documents(TEXTS))                                # (12, dim)
    query_vecs = np.array([emb.embed_query(q["text"]) for q in QUERIES.values()])  # (2, dim)

    # The manual formula and scikit-learn give the same value
    assert abs(manual_cosine_similarity(query_vecs[0], doc_vecs[0])
               - cosine_similarity(query_vecs[:1], doc_vecs[:1])[0][0]) < 1e-9

    sims = cosine_similarity(query_vecs, doc_vecs)                                 # (2, 12)
    output = {
        "model": model,
        "dimensions": int(doc_vecs.shape[1]),
        "shapes": {"query": [1, int(doc_vecs.shape[1])], "documents": list(map(int, doc_vecs.shape))},
        "sample_vector": [round(float(x), 4) for x in query_vecs[0][:5]],   # first 5 values
        "doc_matrix": cosine_similarity(doc_vecs).round(4).tolist(),
    }
    for (name, q), row in zip(QUERIES.items(), sims):
        output[name] = evaluate(dict(zip(IDS, map(float, row))), q["relevant"])
    return output


# ---------------------------------------------------------------------------
# 5. Method 3: LLM re-ranker (reads query + document together)
# ---------------------------------------------------------------------------
class Score(BaseModel):
    id: str
    relevance: float = Field(ge=0, le=1, description="0 = does not answer the query, 1 = fully answers it")
    reason: str = Field(description="Short justification, at most 15 words")


class Reranking(BaseModel):
    scores: List[Score]

    @model_validator(mode="after")
    def all_documents_scored(self):
        missing = set(IDS) - {s.id for s in self.scores}
        if missing:
            raise ValueError(f"Missing scores for: {sorted(missing)}")
        return self


async def rerank(llm, name: str, query: dict) -> tuple[str, dict]:
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are the re-ranker of a search engine. For EACH document, read the query and the document together "
         "and score from 0 to 1 whether the document ANSWERS what the query asks for. "
         "Talking about the same topic is not enough: if the document contradicts or negates what the query asks for, "
         "the score must be low. Return a score for every id. Write the reasons in Spanish."),
        ("human", "Query: {query}\n\nDocuments:\n{documents}"),
    ])
    chain = prompt | llm.with_structured_output(Reranking).with_retry(stop_after_attempt=3)
    documents = "\n".join(f"[{i}] {t}" for i, t in DOCUMENTS.items())
    result: Reranking = await chain.ainvoke({"query": query["text"], "documents": documents})

    scores = {s.id: s.relevance for s in result.scores if s.id in DOCUMENTS}
    output = evaluate(scores, query["relevant"])
    output["reasons"] = {s.id: s.reason for s in result.scores if s.id in DOCUMENTS}
    return name, output


async def reranker_method() -> dict:
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(model=os.environ["GEMINI_MODEL"], temperature=0)
    # Both queries are re-ranked concurrently (asyncio.gather, as in Module 1)
    pairs = await asyncio.gather(*(rerank(llm, n, q) for n, q in QUERIES.items()))
    return {"model": os.environ["GEMINI_MODEL"], **dict(pairs)}


# ---------------------------------------------------------------------------
# 6. Output
# ---------------------------------------------------------------------------
def show(title: str, ev: dict) -> None:
    print(f"\n{title}")
    print("  ranking:", "  ".join(f"{i}={ev['scores'][i]:.3f}" for i in ev["ranking"]))
    print(f"  p@3={ev['p@3']}  p@5={ev['p@5']}  margin={ev['margin']:+.4f}  "
          f"threshold={ev['threshold']} (accuracy {ev['threshold_accuracy']:.0%})")


def main() -> None:
    results = {"documents": DOCUMENTS, "queries": QUERIES, "methods": {}}

    results["methods"]["tfidf"] = tfidf_method()
    for q in QUERIES:
        show(f"[TF-IDF] {q} query", results["methods"]["tfidf"][q])

    if "--no-api" not in sys.argv:
        model_a = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
        model_b = os.getenv("GEMINI_EMBEDDING_MODEL_B", "gemini-embedding-2-preview")
        variants = [
            (f"{model_a}", model_a, None),
            (f"{model_a}@768", model_a, 768),     # same model, vector truncated to 768 dims
            (f"{model_b}", model_b, None),        # another model (if your account has it)
        ]
        for key, model, dims in variants:
            try:
                r = embeddings_method(model, dims)
                results["methods"][f"emb:{key}"] = r
                print(f"\nEmbeddings {key}: query {r['shapes']['query']}, documents {r['shapes']['documents']}")
                for q in QUERIES:
                    show(f"[{key}] {q} query", r[q])
            except Exception as e:
                print(f"\n[warning] Could not use {key}: {e}")
                results["methods"][f"emb:{key}"] = {"error": str(e)}

        try:
            results["methods"]["reranker"] = asyncio.run(reranker_method())
            for q in QUERIES:
                show(f"[re-ranker {os.environ['GEMINI_MODEL']}] {q} query", results["methods"]["reranker"][q])
        except Exception as e:
            print(f"\n[warning] Re-ranker failed: {e}")
            results["methods"]["reranker"] = {"error": str(e)}

    output_file = Path(__file__).with_name("results_v2.json")
    output_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults saved to {output_file.name}")


if __name__ == "__main__":
    main()
