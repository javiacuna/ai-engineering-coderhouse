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

## Ejercicio: Limpieza y chunking por tokens

📁 [`ejercicio-chunking/`](ejercicio-chunking/)

`DocumentProcessor` prepara un documento convertido desde PDF para un pipeline RAG: **limpieza → fragmentación**.

- `clean_text` (regex): borra el encabezado que se repite en cada página (`ACME S.A. — Manual interno — Pág. N`), elimina caracteres basura (`■`), reemplaza espacios múltiples por uno y deja como máximo un salto doble entre párrafos. No borra todos los saltos de línea, porque el splitter los usa para cortar por párrafo.
- `calculate_tokens`: cuenta tokens con `tiktoken` (encoding `cl100k_base`).
- `RecursiveCharacterTextSplitter` con `length_function=calculate_tokens`, `chunk_size=500` y `chunk_overlap=50`: el tamaño se mide en **tokens**. Corta por párrafo, línea, oración, palabra y carácter, en ese orden.
- `process_document`: limpia, divide, imprime los tokens de cada chunk y devuelve la lista.

El script corre tres configuraciones: 40/10 (5 chunks, ninguno de más de 40 tokens), 500/50 (1 chunk de 147 tokens) y 25/10, donde se ve el overlap.

```bash
python ejercicio-chunking/document_processor.py
```
