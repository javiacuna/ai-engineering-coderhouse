# Pre-entrega 4 · Sistema RAG escalable en la nube con Pinecone

Módulo de recuperación para un sistema RAG en la nube:

- **Ingesta** de documentos técnicos a un índice **Pinecone Serverless**, con el texto original y metadatos avanzados guardados en cada vector.
- **Recuperador híbrido**, que combina la búsqueda vectorial de Pinecone con la búsqueda léxica **BM25** mediante un `EnsembleRetriever` de LangChain.
- **Evaluación** con un *golden set*: **Precision@5**, **Recall@5** y MRR, comparando el retriever vectorial, el BM25 y el híbrido.

| Requisito de la consigna | Dónde está |
|---|---|
| `.env` con `PINECONE_API_KEY`, key del proveedor de embeddings e `INDEX_NAME` | `.env.example` → `config.py` (`Settings.from_env`) |
| Script que verifica si el índice existe y lo crea en modo Serverless | `setup_index.py` → `ensure_index()` |
| Ingesta: dataset técnico, `RecursiveCharacterTextSplitter`, embeddings y upsert con contenido y fuente en la metadata | `ingest.py` → `ingest()` |
| Metadatos avanzados: fuente, página, categoría y etiquetas | `ingest.py` → `ChunkMetadata` |
| Clase `RAGSystem` con un `EnsembleRetriever` (BM25 + vectorial) que devuelve el top-5 | `rag_system.py` → `RAGSystem` |
| `evaluate.py` con un benchmark de 5 preguntas, Recall@5 y Precision@5 | `evaluate.py` + `golden_set.json` |
| Reporte en consola | `evaluate.py` → `print_report()` |

## Estructura

```
pre-entrega-4-rag-pinecone/
├── data/                       # Dataset: 6 guías técnicas de asyncio (Markdown con categoría y etiquetas)
├── config.py                   # Settings (.env + Pydantic) y fábricas de embeddings y LLM
├── setup_index.py              # Crea o verifica el índice Serverless
├── ingest.py                   # Carga, chunking, embeddings y upsert incremental en Pinecone
├── rag_system.py               # PineconeRetriever, BM25, EnsembleRetriever y RAGSystem
├── evaluate.py                 # Precision@5, Recall@5 y MRR sobre el golden set
├── golden_set.json             # 5 preguntas con su documento esperado
├── ask.py                      # Consultar el sistema desde la terminal (opcional: --answer)
├── results/evaluation.json     # Se genera al correr evaluate.py
├── tests/                      # 20 tests con un Pinecone en memoria (sin keys ni internet)
├── requirements.txt
└── .env.example
```

## El dataset

Son seis guías sobre la librería **`asyncio`** de Python, escritas para este ejercicio:

| Documento | Categoría | Tema |
|---|---|---|
| `01-corutinas-y-event-loop` | fundamentos | `async def`, event loop, `asyncio.run` |
| `02-tareas-gather-taskgroup` | tareas | `create_task`, `gather`, `TaskGroup`, `as_completed` |
| `03-timeouts-y-cancelacion` | control-de-tiempo | `asyncio.timeout`, `wait_for`, `CancelledError`, `shield` |
| `04-sincronizacion` | sincronizacion | `Semaphore`, `Lock`, `Event`, `Condition` |
| `05-colas-productor-consumidor` | colas | `asyncio.Queue`, `task_done`, `join`, backpressure |
| `06-codigo-bloqueante` | rendimiento | `to_thread`, `run_in_executor`, modo debug |

Es un buen caso para la búsqueda híbrida, porque las preguntas mezclan lenguaje natural ("se queda trabado") con identificadores exactos (`task_done`, `TaskGroup`, `requests.get`). Los identificadores son el punto débil de los embeddings y el fuerte de BM25.

Cada archivo empieza con un pequeño encabezado de metadatos:

```markdown
---
category: colas
tags: [queue, productor, consumidor, task_done, join, backpressure]
---
```

## Cómo replicar el índice

**1. Instalar** (Python 3.12):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Configurar el `.env`** a partir de `.env.example`:

```bash
PINECONE_API_KEY=...          # app.pinecone.io → API Keys (alcanza con el plan gratuito)
GOOGLE_API_KEY=...            # aistudio.google.com (embeddings gratuitos de Gemini)
INDEX_NAME=rag-asyncio-docs
```

Para usar OpenAI en lugar de Gemini: `EMBEDDING_PROVIDER=openai` y `OPENAI_API_KEY=...`. Las dos opciones generan vectores de 1536 dimensiones.

**3. Crear el índice** (Serverless, `aws` / `us-east-1`, 1536 dimensiones, coseno):

```bash
python setup_index.py
```

Si el índice ya existe, verifica que su dimensión y su métrica coincidan con la configuración. Si no coinciden, frena con un error en lugar de fallar más tarde en el upsert.

**4. Cargar los documentos:**

```bash
python ingest.py
```

**5. Evaluar:**

```bash
python evaluate.py
```

**6. Consultar** (opcional):

```bash
python ask.py "¿Para qué sirve asyncio.shield?"            # top-5 híbrido
python ask.py "¿Para qué sirve asyncio.shield?" --mode bm25
python ask.py "¿Para qué sirve asyncio.shield?" --answer   # + respuesta generada con Gemini
```

**Tests** (no usan Pinecone ni APIs):

```bash
python -m pytest -q
```

## Resultados (corrida real)

Con `gemini-embedding-001` a 1536 dimensiones y el índice Serverless `rag-asyncio-docs`:

**Ingesta:** 6 documentos → 13 chunks, subidos en un lote. La segunda ejecución de `ingest.py` no embebió nada:

```
Documents: 6 | chunks: 13 | embedded now: 13 | unchanged: 0 | deleted: 0 | failed: 0
Documents: 6 | chunks: 13 | embedded now: 0 | unchanged: 13 | deleted: 0 | failed: 0
```

**Evaluación** (`python evaluate.py`):

```
Retriever   Recall@5  Precision@5  Techo P@5    MRR
vector          1.00         0.48       0.56   1.00
bm25            1.00         0.40       0.56   1.00
hybrid          1.00         0.44       0.56   1.00
```

Precision@5 por pregunta (entre paréntesis, el techo):

| # | Pregunta | Vector | BM25 | Híbrido |
|---|---|---|---|---|
| 1 | Excepción de `asyncio.timeout()` | 0.40 (0.40) | 0.40 | 0.40 |
| 2 | No más de 5 pedidos a la vez (paráfrasis) | 0.60 (1.00) | 0.40 | 0.60 |
| 3 | Qué pasa si falla una tarea en un `TaskGroup` | 0.40 (0.40) | 0.40 | 0.40 |
| 4 | Programa trabado con `requests.get` | **0.40** (0.40) | 0.20 | 0.20 |
| 5 | `task_done()` y `join()` en una cola | 0.60 (0.60) | 0.60 | 0.60 |

### Análisis

- **Recall@5 y MRR saturados (1.00 en los tres).** En las 5 preguntas, el chunk del documento correcto quedó **primero** con cualquier retriever. Con un corpus chico (13 chunks) y temas bien separados, el benchmark no alcanza para distinguir los retrievers por recall: la diferencia está en la precisión.
- **El vectorial fue el más preciso** (0.48, el 86 % del techo de 0.56). Los embeddings de Gemini resolvieron bien incluso las preguntas con identificadores exactos (`TaskGroup`, `task_done`), que en teoría son el terreno de BM25.
- **El híbrido quedó en el medio** (0.44), y la diferencia se explica por la pregunta 4. El vectorial trajo los dos chunks de `06-codigo-bloqueante`, el segundo en el puesto 5. BM25 no lo tenía arriba, porque las palabras comunes de la pregunta ("programa", "corutina") aparecen en todos los documentos. Al fusionar con RRF, ese chunk salió del top 5.
- **En la pregunta 2**, la paráfrasis que no menciona `Semaphore`, el vectorial y el híbrido trajeron también el documento de colas (relacionado). BM25 no lo encontró, porque no comparte palabras con la pregunta.

**Conclusión:** combinar retrievers no siempre mejora el resultado. El híbrido es más **robusto**: no depende de que un solo método acierte. Pero cuando uno de los dos ya es muy bueno, mezclarlo con uno más flojo puede bajar un poco la precisión. La ventaja de BM25 se espera en corpus grandes con documentos parecidos entre sí, donde los embeddings confunden términos cercanos.

**Por qué no ajusté los pesos:** subir `VECTOR_WEIGHT` probablemente recuperaría la pregunta 4, pero sería ajustar el sistema mirando las respuestas del propio benchmark: con 5 preguntas, eso es sobreajuste. Para calibrar los pesos hace falta un golden set más grande, separado en preguntas de ajuste y preguntas de prueba.

## Diseño

### Índice: dimensión, métrica y namespace

| Parámetro | Valor | Por qué |
|---|---|---|
| Tipo | Serverless (`aws`, `us-east-1`) | No hay nodos que administrar, y el almacenamiento escala solo. Es la región que permite el plan gratuito. |
| `dimension` | 1536 | Es el tamaño de los vectores de `text-embedding-3-small` de OpenAI. `gemini-embedding-001` genera 3072 por defecto, pero se entrena con *Matryoshka Representation Learning*: con `output_dimensionality=1536` conserva las primeras 1536 dimensiones con poca pérdida de calidad, así que **los dos proveedores entran en el mismo índice**. |
| `metric` | `cosine` | Los modelos de embeddings de texto representan el significado en el **ángulo** entre vectores. Además, los vectores de Gemini truncados a 1536 no tienen norma 1, y el coseno ignora la magnitud. |
| Namespace | `python-asyncio-docs` | Separa este dataset de otros datos del mismo índice (por ejemplo, el ejercicio de `dev-environment`). Las búsquedas solo recorren este namespace. |

### Ingesta

1. **Carga**: lee los `.md` de `data/`, con su categoría y sus etiquetas.
2. **Chunking**: usa `RecursiveCharacterTextSplitter` midiendo en **tokens** (tiktoken, `cl100k_base`): **500 tokens con 75 de solapamiento**, dentro del rango de 500 a 800 que sugiere la consigna. Primero intenta cortar por sección (`## `), después por párrafo, línea y oración.
3. **Sección y página**: cada chunk guarda la sección `## ` a la que pertenece y su número (`page`). Markdown no tiene páginas, así que `page` es el número de sección; un loader de PDF guardaría ahí la página real.
4. **Control de dimensión**: antes de subir nada, verifica que los embeddings tengan exactamente 1536 dimensiones.
5. **Upsert por lotes** de 100 vectores, con reintentos y *backoff* exponencial ante errores temporales (`RateLimitError`, `ServiceError`, red, timeout). Los lotes que fallan quedan registrados en el reporte.

Los metadatos de cada vector se validan con un esquema estricto (`ChunkMetadata`, `extra="forbid"`), para evitar el *schema drift*:

| Campo | Para qué |
|---|---|
| `text` | El **texto original del chunk**. Con él, el resultado se puede mostrar, indexar en BM25 o mandar al LLM sin consultar otra base de datos. |
| `doc_id`, `source`, `title` | De qué documento viene (`doc_id` es la unidad que usa el golden set) |
| `section`, `page`, `chunk_index` | Dónde está dentro del documento |
| `category`, `tags` | Filtros, por ejemplo `{"category": {"$eq": "colas"}}` o `{"tags": {"$in": ["semaphore"]}}` |
| `tokens` | Tamaño del chunk |
| `content_hash`, `embedding_model`, `ingested_at` | Ingesta incremental y control del modelo de embeddings |

### Ingesta incremental

Los IDs son deterministas: `05-colas-productor-consumidor#002` es el tercer chunk del documento de colas. Antes de subir, `ingest.py` consulta qué hay en el namespace (`list` + `fetch`):

| Situación | Qué hace |
|---|---|
| El chunk no cambió (mismo `content_hash` y mismo modelo) | No lo vuelve a embeber ni a subir |
| El chunk es nuevo o cambió | Lo embebe y lo sube (el upsert pisa el vector anterior) |
| El chunk ya no existe (documento borrado o más corto) | Lo borra del índice |
| Cambió el modelo de embeddings | Vuelve a embeber todo |

La segunda ejecución de `python ingest.py` no gasta llamadas a la API de embeddings. `--rebuild` vacía el namespace y carga todo de nuevo.

### Recuperador híbrido

```
consulta ─┬─► PineconeRetriever (embedding + coseno, top 10) ─┐
          │                                                   ├─► EnsembleRetriever (RRF ponderado) ─► top 5
          └─► BM25Retriever (términos exactos, top 10) ───────┘
```

- **`PineconeRetriever`**: es un `BaseRetriever` de LangChain propio, sobre el **SDK nativo** de Pinecone. No usé `PineconeVectorStore` porque `langchain-pinecone` todavía exige `pinecone<8`, y este proyecto usa `pinecone` 10.x. La consigna permite cualquiera de las dos opciones.
- **`BM25Retriever`**: su corpus se arma con el texto guardado en la metadata de Pinecone, así que **Pinecone es la única fuente de verdad** y los dos retrievers buscan sobre los mismos chunks. Usa un tokenizador propio, porque el de LangChain solo separa por espacios: pasa a minúsculas, saca tildes, conserva identificadores como `task_done` o `to_thread` y descarta palabras vacías en español.
- **`EnsembleRetriever`**: cada retriever trae 10 candidatos, y los combina con **Reciprocal Rank Fusion** ponderado (pesos 0.5 / 0.5). Cada documento suma `peso / (60 + posición)` en cada lista. Los duplicados se unifican por `chunk_id`, y `RAGSystem` devuelve los 5 mejores.
- **Control del modelo de embeddings**: `RAGSystem` verifica que el namespace se haya indexado con el mismo modelo que se usa para consultar. Si no coincide, pide `--rebuild`.

### Evaluación

`golden_set.json` tiene 5 preguntas con el documento que las responde (`documento_id_esperado`) y, opcionalmente, documentos que también son útiles (`documentos_relacionados`). Mezclan distintos tipos de consulta: términos exactos, nombres de la API y paráfrasis que no mencionan la palabra clave.

| Métrica | Definición |
|---|---|
| **Recall@5** | 1 si algún chunk del documento esperado está entre los 5 recuperados, y 0 si no. Se promedia entre las preguntas. |
| **Precision@5** | Proporción de los 5 chunks recuperados que vienen de un documento útil (el esperado o uno relacionado) |
| **Techo de P@5** | La mejor Precision@5 posible. Si los documentos útiles tienen 2 chunks en total, como mucho 2 de los 5 resultados pueden ser útiles: el techo es 0.40. Sin este dato, una Precision@5 de 0.40 parecería mala cuando en realidad es perfecta. |
| **MRR** | 1 / posición del primer chunk del documento esperado. Mide qué tan arriba aparece. |

`evaluate.py` calcula las métricas para los tres retrievers (vectorial, BM25 e híbrido), así se ve cuánto aporta combinarlos. Además de imprimir el reporte, guarda todo en `results/evaluation.json`.
