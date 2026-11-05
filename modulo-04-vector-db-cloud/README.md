# Módulo 4 · Bases vectoriales en la nube

## Ejercicio: Pinecone Serverless

📁 [`ejercicio-pinecone-serverless/`](ejercicio-pinecone-serverless/)

`setup_vector_infrastructure(index_name, dimension)` automatiza la infraestructura con el SDK de Python de Pinecone:

1. Crea el cliente con `PINECONE_API_KEY`, que se lee desde el `.env`.
2. Si el índice no existe, lo crea como **serverless** (`aws` / `us-east-1`, la región del plan gratuito) y espera a que `status.ready` sea verdadero.
3. Si el índice ya existía con otra dimensión, frena con un error claro en lugar de fallar en el upsert (*mismatch* de dimensiones).
4. Carga 3 vectores de prueba en el namespace **`dev-environment`**, **en lotes** y con reintentos. Como es un `upsert`, correrlo dos veces no duplica los vectores.
5. Devuelve `describe_index_stats()`. Las estadísticas tardan unos segundos en actualizarse, así que las vuelve a consultar hasta que aparecen los vectores (máximo 30 segundos).

### Dimensión y métrica

| Parámetro | Valor | Por qué |
|---|---|---|
| `dimension` | 1536 | Es el tamaño de los vectores de `text-embedding-3-small` de OpenAI. Tiene que coincidir exactamente con el modelo: un vector de 768 o 3072 se rechaza en el upsert. `gemini-embedding-001` devuelve 3072 por defecto, pero acepta `output_dimensionality=1536`. |
| `metric` | `cosine` | Los modelos de embeddings de texto se entrenan para que el **ángulo** entre vectores represente la similitud semántica, y OpenAI devuelve vectores normalizados. El coseno compara solo la dirección. La distancia euclidiana mezclaría la magnitud con el puntaje, así que no es la métrica adecuada para estos modelos. |

Las dos se fijan al crear el índice y no se pueden cambiar después: cambiar de modelo de embeddings implica crear un índice nuevo.

### Esquema de metadatos

| Campo | Para qué sirve |
|---|---|
| `doc_id`, `chunk_index` | Saber de qué documento y fragmento viene cada vector, para reconstruir o borrar un documento completo |
| `source` | Filtrar por origen del texto |
| `topic`, `priority` | Filtros de negocio |
| `created_at` | Fecha como timestamp Unix **numérico**, para filtros por rango (`{"created_at": {"$gte": ...}}`). Como string no se podría comparar. |
| `created_date` | La misma fecha en formato legible (`2026-11-05`) |
| `text` | El fragmento original: el resultado de una búsqueda se puede mostrar o pasar al LLM sin buscarlo en otro lado. Pinecone admite hasta 40 KB de metadatos por vector. |

### Upsert por lotes con manejo de errores

`upsert_in_batches()` divide los vectores en lotes de 100 y sube cada uno por separado, dentro de un `try/except`:

- **Errores transitorios** (`RateLimitError` 429, `ServiceError` 5xx, errores de red y timeouts): reintenta hasta 3 veces. Ante un límite de tasa espera lo que indica la API (`retry_after`); en los otros casos espera 1 y después 2 segundos (*backoff* exponencial).
- **Errores permanentes** (dimensión incorrecta, metadatos inválidos, key inválida): no reintenta, porque volvería a fallar, y lanza `VectorUploadError` con un mensaje claro.

```bash
python modulo-04-vector-db-cloud/ejercicio-pinecone-serverless/pinecone_setup.py
```

Salida esperada:

```
dimension: 1536
namespace 'dev-environment': 3 vectors
total_vector_count: 3
```

## Ejercicio: Ingesta masiva y metadatos avanzados

📁 [`ejercicio-ingesta-masiva/`](ejercicio-ingesta-masiva/)

`IngestionPipeline` carga documentos en lotes y busca filtrando por metadatos. Trabaja con un índice simulado (`MockPineconeIndex`), así que no necesita cuenta de Pinecone. El mock guarda los vectores por ID y aplica los filtros `$eq` y `$and`, para poder verificar la deduplicación y el aislamiento entre clientes.

### Metadatos con interfaz estricta

`create_metadata` arma y valida los metadatos con un modelo Pydantic, `DocumentMetadata`:

| Campo | Para qué sirve |
|---|---|
| `tenant_id` | Dueño del documento (aislamiento multi-tenant) |
| `category`, `author` | Filtros de negocio. La categoría y el tenant se normalizan a minúsculas, para que `"Legal"` y `"legal"` coincidan con el mismo filtro. |
| `ingested_at` | Fecha y hora de la ingesta (ISO 8601) |
| `char_count` | Tamaño del texto |
| `content_hash` | SHA-256 del texto, para detectar duplicados |

El modelo usa `extra="forbid"`: es la **interfaz estricta** que evita el *schema drift*. Si un proceso intenta guardar `date_created` en lugar de `ingested_at`, falla en vez de guardar un campo que después ningún filtro encuentra.

### Aislamiento multi-tenant

`search_by_category(query_vector, category, tenant_id)` aplica un **filtro combinado**:

```python
{"$and": [{"tenant_id": {"$eq": "acme"}}, {"category": {"$eq": "legal"}}]}
```

`tenant_id` es **obligatorio**: sin él, la función lanza un error, así que ninguna consulta puede devolver documentos de otro cliente. En la prueba, `globex` también tiene un documento `legal`, y la búsqueda de `acme` no lo devuelve. Para separar todavía más a los clientes, Pinecone permite usar un **namespace por tenant**, que aísla los datos a nivel de índice.

### Deduplicación

Cada vector tiene un **ID determinista**, `sha256(tenant_id + hash del texto)`, en lugar de un `uuid4` aleatorio:

- Si se vuelve a cargar la misma tanda de documentos, el `upsert` **sobrescribe** los vectores en lugar de duplicarlos.
- Los documentos repetidos dentro de una misma carga se saltean.
- El tenant forma parte del ID: si dos clientes suben el mismo texto, cada uno conserva su copia.

### Resiliencia: reintentos y registro de lotes fallidos

Cada lote se sube con `_upsert_with_retry`:

- Ante un **error temporal** (red o timeout; con el SDK real, también `RateLimitError` y `ServiceError`), reintenta hasta 3 veces con *backoff* exponencial (0.5 s y después 1 s).
- Si el lote sigue fallando, o el error es permanente, **no aborta la ingesta**: lo registra en el log y en `IngestionReport.failed_batches` (posición, IDs y error), y sigue con el lote siguiente. Así se pueden reprocesar solo los lotes fallidos.

`process_and_upsert_batches` devuelve un `IngestionReport` con `processed`, `duplicates_skipped` y `failed_batches`.

```bash
python modulo-04-vector-db-cloud/ejercicio-ingesta-masiva/batch_ingestion.py
```

### Salida

```
== 1) Ingestion (batch_size=2)
2
2
1
Total processed: 5 | vectors in the index: 5

== 2) Same load again (deduplication)
2
2
1
Processed: 5 | duplicates skipped: 1 | vectors in the index: 5 (still 5)

== 3) Search 'legal' for tenant 'acme' (combined filter)
query(top_k=5, filter={'$and': [{'tenant_id': {'$eq': 'acme'}}, {'category': {'$eq': 'legal'}}]})
   acme | legal | 709f1a25...
   1 match(es): globex's legal document is not visible to acme

== 4) Temporary error: the batch is retried with backoff
WARNING | Batch 1: simulated network error (attempt 1/3), retrying in 0.1 s
2
2
1
Processed: 5 | failed batches: 0

== 5) Error that persists: the batch is recorded and the rest continues
WARNING | Batch 1: simulated network error (attempt 1/3), retrying in 0.1 s
WARNING | Batch 1: simulated network error (attempt 2/3), retrying in 0.2 s
ERROR | Batch 1 failed: ConnectionError: simulated network error
2
1
Processed: 3 | failed batches: [(0, 2, 'ConnectionError: simulated network error')]
```

La ingesta original da 3 llamadas a `upsert` (2, 2 y 1 vectores) y un total de 5. La segunda carga agrega un documento repetido: se saltea, y el índice sigue teniendo 5 vectores.
