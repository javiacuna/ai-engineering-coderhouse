# Módulo 3 — Persistencia de datos y vector DBs

## Ejercicio: Embeddings y similitud semántica (v2)

📁 [`ejercicio-similitud-semantica/`](ejercicio-similitud-semantica/) · 📄 Informe: [`embeddings-similitud-semantica.pdf`](ejercicio-similitud-semantica/embeddings-similitud-semantica.pdf)

Evalúa si una búsqueda distingue **relevancia semántica** de **coincidencia de palabras clave**, y qué pasa cuando un documento habla del mismo tema pero **dice lo contrario**.

**Conjunto de evaluación:** 12 oraciones etiquetadas (R = relevantes, I = irrelevantes con trampas de palabras clave y de dominio cercano, C = contradictorias) y dos consultas: *"despliegue de microservicios"* y la negada *"aplicación monolítica sin microservicios"*.

**Métodos comparados** (todos con similitud coseno de `sklearn.metrics.pairwise.cosine_similarity`, salvo el re-ranker):

| Método | Detalle |
|---|---|
| Palabras clave | `TfidfVectorizer` |
| Embeddings | `gemini-embedding-001` (3072 y 768 dimensiones) y `gemini-embedding-2-preview` |
| Re-ranker | LLM que lee consulta + documento juntos y puntúa 0–1 (salida estructurada con Pydantic, `with_retry`, `asyncio.gather`) |

**Métricas:** precisión@3, precisión@5, margen de separación y umbral calibrado.

**Resultado principal:** los embeddings resuelven la rigidez de las palabras clave pero agrupan por *tema*, no por *postura* (relevantes y contradictorias puntúan igual en promedio). El re-ranker separa perfecto ambas consultas (margen +0,80 / +0,85).

```bash
python ejercicio-similitud-semantica/semantic_similarity.py           # requiere GOOGLE_API_KEY y GEMINI_MODEL
python ejercicio-similitud-semantica/semantic_similarity.py --no-api  # solo TF-IDF
```

Genera `results_v2.json` con todos los puntajes y métricas.

> El informe PDF se entregó antes de pasar el código a inglés, por eso menciona los nombres anteriores de los archivos (`similitud_semantica.py`, `resultados_v2.json`). Los resultados son los mismos.

## Ejercicio: Limpieza y chunking por tokens

📁 [`ejercicio-chunking/`](ejercicio-chunking/)

`DocumentProcessor` prepara un documento convertido desde PDF para un pipeline RAG: **limpieza → fragmentación**.

- `clean_text`: aplica una lista declarativa de reglas (`CLEANING_RULES`) que borra el encabezado repetido en cada página (`ACME S.A. — Manual interno — Pág. N`), elimina caracteres basura (`■`), reemplaza espacios múltiples por uno y deja como máximo un salto doble entre párrafos. Valida la entrada (`None`, tipos incorrectos) y, si una regla falla, la registra y sigue con las demás.
- `calculate_tokens`: cuenta tokens con `tiktoken` (encoding `cl100k_base`).
- `RecursiveCharacterTextSplitter` con `length_function=calculate_tokens`, `chunk_size=500` y `chunk_overlap=50`: el tamaño se mide en **tokens**. Corta por párrafo, línea, oración, palabra y carácter, en ese orden.
- `process_document`: limpia, divide, imprime los tokens de cada chunk y devuelve la lista.

El script corre tres configuraciones: 40/10 (5 chunks, ninguno de más de 40 tokens), 500/50 (1 chunk de 147 tokens) y 25/10, donde se ve el overlap. Al final prueba entradas inesperadas (`None`, `123`, texto vacío).

```bash
python ejercicio-chunking/document_processor.py
```

## Ejercicio: Persistencia local con ChromaDB (CRUD)

📁 [`ejercicio-chromadb-crud/`](ejercicio-chromadb-crud/)

`VectorMemoryManager` es una memoria vectorial que se guarda en disco con `chromadb.PersistentClient`:

- `upsert_documents`: inserta o actualiza (`collection.upsert`). Si se repite un ID, actualiza el documento y su vector en lugar de duplicarlo.
- `semantic_search(query_text, n_results, where=None)`: búsqueda por similitud (`collection.query`) con filtro opcional por metadatos. Devuelve una lista de diccionarios con `id`, `document`, `metadata` y `distance`.
- `delete_by_id(ids)`: borra documentos y sus vectores.
- La colección usa distancia **coseno** (`hnsw:space`), la métrica recomendada para embeddings de texto.
- `content_id(text)`: genera IDs deterministas (hash del contenido) para que `upsert` pueda actualizar.

### Manejo de errores

Cada operación contra la base (abrir la colección, `upsert_documents`, `semantic_search` y `delete_by_id`) está dentro de un `try...except`. Los fallos se registran con `logging` y se relanzan como una única excepción, `VectorMemoryError`, que conserva el error original en `__cause__`. Así, quien usa la clase captura un solo tipo de error y no un traceback crudo de Chroma.

Se capturan tres familias de errores (`DATABASE_ERRORS`):

| Excepción | Cuándo ocurre |
|---|---|
| `chromadb.errors.ChromaError` | Clase base de todos los errores de Chroma: `NotFoundError` (colección borrada), `DuplicateIDError`, `InvalidArgumentError`, `InternalError` (fallos de disco o almacenamiento). En Chroma 1.x no existe `DatabaseError`: este es su equivalente. |
| `ValueError` | Validación del cliente: un filtro `where` mal armado o metadatos con tipos no soportados. |
| `OSError` | La carpeta de persistencia no se puede leer o escribir (permisos, disco lleno). |

Que las listas de `upsert_documents` tengan distinto largo sigue lanzando `ValueError`, porque es un error de programación y se detecta antes de tocar la base. Al final, el script provoca dos errores a propósito (IDs duplicados y un filtro inválido) para mostrar que se manejan.

### Observación: `upsert` mezcla la metadata

Al volver a correr el script, `doc1` seguía teniendo `version: 2` de la corrida anterior, aunque el primer `upsert` solo manda `{"source": "news"}`. El `upsert` de Chroma reemplaza el texto y el vector, pero **mezcla las claves de metadata**: las que no se envían quedan como estaban. Para reemplazar la metadata completa hay que hacer `delete` + `upsert`, o enviar todas las claves.

```bash
python ejercicio-chromadb-crud/vector_memory.py
```

La primera vez, ChromaDB descarga su modelo de embeddings (all-MiniLM-L6-v2). La base se guarda en `ejercicio-chromadb-crud/chroma_db/`, que está excluida del repo en `.gitignore`.
