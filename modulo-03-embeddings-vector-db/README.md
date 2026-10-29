# Módulo 3 — Persistencia de datos y vector DBs

## Ejercicio: Embeddings y similitud semántica

📁 [`ejercicio-similitud-semantica/`](ejercicio-similitud-semantica/)

Evalúa si un modelo de embeddings distingue **relevancia semántica** de **coincidencia de palabras clave**:

- Consulta: *"despliegue de microservicios"*.
- 5 oraciones relevantes que hablan del mismo concepto con vocabulario distinto (R1–R5).
- 2 oraciones trampa que comparten palabras clave pero no el significado (T1 irrelevante, T2 opuesta).
- Se rankean las 7 oraciones con dos métodos y **similitud coseno** (`sklearn.metrics.pairwise.cosine_similarity`):
  - **TF-IDF** (palabras clave) con `TfidfVectorizer`.
  - **Embeddings** de Gemini (`gemini-embedding-001`) con `GoogleGenerativeAIEmbeddings`.
- Los resultados se guardan en `resultados.json`. El análisis completo y el diagrama de flujo están en el PDF entregable.

```bash
python ejercicio-similitud-semantica/similitud_semantica.py            # requiere GOOGLE_API_KEY
python ejercicio-similitud-semantica/similitud_semantica.py --sin-api  # solo TF-IDF
```
