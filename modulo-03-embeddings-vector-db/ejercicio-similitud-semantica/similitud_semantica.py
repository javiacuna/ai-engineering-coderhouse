"""
Ejercicio Módulo 3 - Embeddings y similitud (Coderhouse AI Engineering) - v2

Evalúa si una búsqueda distingue relevancia semántica de coincidencia de palabras clave.
Compara, sobre un conjunto de evaluación etiquetado y dos consultas (una afirmativa y una negada):
  1. Palabras clave (TF-IDF) + similitud coseno.
  2. Embeddings de Gemini + similitud coseno (varios modelos / dimensiones).
  3. Re-ranker: un LLM que lee consulta y documento JUNTOS y puntúa la relevancia.
Métricas: precisión@k, umbral calibrado y margen de separación.

Uso:
  python similitud_semantica.py             # todo (requiere GOOGLE_API_KEY y GEMINI_MODEL en .env)
  python similitud_semantica.py --sin-api   # solo TF-IDF, sin llamar a ninguna API
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
# 1. Conjunto de evaluación
#    R = describe el despliegue de microservicios con otras palabras
#    I = irrelevante (trampas de palabras clave o de dominio cercano)
#    C = contradice / niega el despliegue de microservicios
# ---------------------------------------------------------------------------
DOCUMENTOS = {
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
IDS = list(DOCUMENTOS)
TEXTOS = list(DOCUMENTOS.values())

# Para cada consulta, qué documentos son la respuesta correcta
CONSULTAS = {
    "afirmativa": {
        "texto": "despliegue de microservicios",
        "relevantes": ["R1", "R2", "R3", "R4", "R5"],
    },
    "negada": {
        "texto": "aplicación monolítica sin microservicios",
        "relevantes": ["C1", "C2", "C3"],
    },
}

STOPWORDS_ES = ["de", "la", "el", "en", "un", "una", "y", "que", "se", "con", "por", "del", "los",
                "las", "al", "a", "su", "sus", "cada", "es", "como", "sin", "hasta", "desde", "no",
                "para", "era", "todo", "muy"]

K_VALUES = (3, 5)


# ---------------------------------------------------------------------------
# 2. Métricas
# ---------------------------------------------------------------------------
def similitud_coseno_manual(a: np.ndarray, b: np.ndarray) -> float:
    """cos(θ) = (A · B) / (‖A‖ · ‖B‖)"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def evaluar(puntajes: dict[str, float], relevantes: list[str]) -> dict:
    """Ranking, precisión@k, margen y mejor umbral para un método y una consulta."""
    orden = sorted(puntajes, key=puntajes.get, reverse=True)
    rel = [puntajes[i] for i in IDS if i in relevantes]
    no_rel = [puntajes[i] for i in IDS if i not in relevantes]

    # Solo cuenta como "recuperado" lo que tiene puntaje > 0 (TF-IDF no devuelve textos sin coincidencias)
    recuperados = [i for i in orden if puntajes[i] > 0]
    precision = {f"p@{k}": round(sum(i in relevantes for i in recuperados[:k]) / k, 3) for k in K_VALUES}

    # Margen: peor relevante - mejor no relevante. Positivo = hay un umbral que separa perfecto.
    margen = min(rel) - max(no_rel)

    # Umbral calibrado: el punto de corte que maximiza la exactitud (empate -> umbral más alto)
    candidatos = sorted(set(puntajes.values()))
    mejor = max(
        candidatos,
        key=lambda u: (sum((puntajes[i] >= u) == (i in relevantes) for i in IDS), u),
    )
    aciertos = sum((puntajes[i] >= mejor) == (i in relevantes) for i in IDS)

    return {
        "puntajes": {i: round(puntajes[i], 4) for i in IDS},
        "ranking": orden,
        **precision,
        "margen": round(margen, 4),
        "umbral": round(mejor, 4),
        "exactitud_umbral": round(aciertos / len(IDS), 3),
        "promedio_por_grupo": {
            g: round(float(np.mean([puntajes[i] for i in IDS if i.startswith(g)])), 4) for g in "RIC"
        },
    }


# ---------------------------------------------------------------------------
# 3. Método 1: palabras clave (TF-IDF)
# ---------------------------------------------------------------------------
def metodo_tfidf() -> dict:
    vectorizer = TfidfVectorizer(stop_words=STOPWORDS_ES)
    consultas = [c["texto"] for c in CONSULTAS.values()]
    matriz = vectorizer.fit_transform(consultas + TEXTOS).toarray()
    vec_consultas, vec_docs = matriz[: len(consultas)], matriz[len(consultas):]
    sims = cosine_similarity(vec_consultas, vec_docs)              # (n_consultas x n_docs)
    return {
        nombre: evaluar(dict(zip(IDS, map(float, fila))), c["relevantes"])
        for (nombre, c), fila in zip(CONSULTAS.items(), sims)
    }


# ---------------------------------------------------------------------------
# 4. Método 2: embeddings (uno o más modelos)
# ---------------------------------------------------------------------------
def metodo_embeddings(modelo: str, dimensiones: int | None = None) -> dict:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    emb = GoogleGenerativeAIEmbeddings(model=modelo, output_dimensionality=dimensiones)
    vec_docs = np.array(emb.embed_documents(TEXTOS))                              # (12, dim)
    vec_consultas = np.array([emb.embed_query(c["texto"]) for c in CONSULTAS.values()])  # (2, dim)

    # La fórmula manual y scikit-learn dan lo mismo
    assert abs(similitud_coseno_manual(vec_consultas[0], vec_docs[0])
               - cosine_similarity(vec_consultas[:1], vec_docs[:1])[0][0]) < 1e-9

    sims = cosine_similarity(vec_consultas, vec_docs)                             # (2, 12)
    salida = {
        "modelo": modelo,
        "dimensiones": int(vec_docs.shape[1]),
        "formas": {"consulta": [1, int(vec_docs.shape[1])], "documentos": list(map(int, vec_docs.shape))},
        "vector_ejemplo": [round(float(x), 4) for x in vec_consultas[0][:5]],   # primeros 5 valores
        "matriz_docs": cosine_similarity(vec_docs).round(4).tolist(),
    }
    for (nombre, c), fila in zip(CONSULTAS.items(), sims):
        salida[nombre] = evaluar(dict(zip(IDS, map(float, fila))), c["relevantes"])
    return salida


# ---------------------------------------------------------------------------
# 5. Método 3: re-ranker con LLM (lee consulta + documento juntos)
# ---------------------------------------------------------------------------
class Puntaje(BaseModel):
    id: str
    relevancia: float = Field(ge=0, le=1, description="0 = no responde la consulta, 1 = la responde por completo")
    motivo: str = Field(description="Justificación breve, máximo 15 palabras")


class Reranking(BaseModel):
    puntajes: List[Puntaje]

    @model_validator(mode="after")
    def todos_los_documentos(self):
        faltan = set(IDS) - {p.id for p in self.puntajes}
        if faltan:
            raise ValueError(f"Faltan puntajes para: {sorted(faltan)}")
        return self


async def reranker(llm, nombre: str, consulta: dict) -> tuple[str, dict]:
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Sos el re-ranker de un buscador. Para CADA documento, leé la consulta y el documento juntos "
         "y puntuá de 0 a 1 si el documento RESPONDE a lo que pide la consulta. "
         "Hablar del mismo tema no alcanza: si el documento contradice o niega lo que pide la consulta, "
         "el puntaje debe ser bajo. Devolvé un puntaje para cada id."),
        ("human", "Consulta: {consulta}\n\nDocumentos:\n{documentos}"),
    ])
    chain = prompt | llm.with_structured_output(Reranking).with_retry(stop_after_attempt=3)
    documentos = "\n".join(f"[{i}] {t}" for i, t in DOCUMENTOS.items())
    resultado: Reranking = await chain.ainvoke({"consulta": consulta["texto"], "documentos": documentos})

    puntajes = {p.id: p.relevancia for p in resultado.puntajes if p.id in DOCUMENTOS}
    salida = evaluar(puntajes, consulta["relevantes"])
    salida["motivos"] = {p.id: p.motivo for p in resultado.puntajes if p.id in DOCUMENTOS}
    return nombre, salida


async def metodo_reranker() -> dict:
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(model=os.environ["GEMINI_MODEL"], temperature=0)
    # Las dos consultas se re-rankean en paralelo (asyncio.gather, como en el Módulo 1)
    pares = await asyncio.gather(*(reranker(llm, n, c) for n, c in CONSULTAS.items()))
    return {"modelo": os.environ["GEMINI_MODEL"], **dict(pares)}


# ---------------------------------------------------------------------------
# 6. Salida
# ---------------------------------------------------------------------------
def imprimir(titulo: str, ev: dict) -> None:
    print(f"\n{titulo}")
    print("  ranking:", "  ".join(f"{i}={ev['puntajes'][i]:.3f}" for i in ev["ranking"]))
    print(f"  p@3={ev['p@3']}  p@5={ev['p@5']}  margen={ev['margen']:+.4f}  "
          f"umbral={ev['umbral']} (exactitud {ev['exactitud_umbral']:.0%})")


def main() -> None:
    resultados = {"documentos": DOCUMENTOS, "consultas": CONSULTAS, "metodos": {}}

    resultados["metodos"]["tfidf"] = metodo_tfidf()
    for q in CONSULTAS:
        imprimir(f"[TF-IDF] consulta {q}", resultados["metodos"]["tfidf"][q])

    if "--sin-api" not in sys.argv:
        modelo_a = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
        modelo_b = os.getenv("GEMINI_EMBEDDING_MODEL_B", "gemini-embedding-2-preview")
        variantes = [
            (f"{modelo_a}", modelo_a, None),
            (f"{modelo_a}@768", modelo_a, 768),     # mismo modelo, vector recortado a 768 dims
            (f"{modelo_b}", modelo_b, None),        # otro modelo (si tu cuenta lo tiene disponible)
        ]
        for clave, modelo, dims in variantes:
            try:
                r = metodo_embeddings(modelo, dims)
                resultados["metodos"][f"emb:{clave}"] = r
                print(f"\nEmbeddings {clave}: consulta {r['formas']['consulta']}, documentos {r['formas']['documentos']}")
                for q in CONSULTAS:
                    imprimir(f"[{clave}] consulta {q}", r[q])
            except Exception as e:
                print(f"\n[aviso] No se pudo usar {clave}: {e}")
                resultados["metodos"][f"emb:{clave}"] = {"error": str(e)}

        try:
            resultados["metodos"]["reranker"] = asyncio.run(metodo_reranker())
            for q in CONSULTAS:
                imprimir(f"[re-ranker {os.environ['GEMINI_MODEL']}] consulta {q}", resultados["metodos"]["reranker"][q])
        except Exception as e:
            print(f"\n[aviso] Falló el re-ranker: {e}")
            resultados["metodos"]["reranker"] = {"error": str(e)}

    salida = Path(__file__).with_name("resultados_v2.json")
    salida.write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResultados guardados en {salida.name}")


if __name__ == "__main__":
    main()
