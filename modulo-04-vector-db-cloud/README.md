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

`IngestionPipeline` carga documentos en lotes y busca filtrando por metadatos. Trabaja con un índice simulado (`MockPineconeIndex`), así que no necesita cuenta de Pinecone.

- `create_metadata`: arma los metadatos de cada documento (`category`, `author`, `ingested_at` en formato ISO y `char_count`). Los valida con un modelo Pydantic, `DocumentMetadata`, con `extra="forbid"`: es la **interfaz estricta** que evita el *schema drift*. Si un proceso intenta guardar `date_created` en lugar de `ingested_at`, falla en vez de guardar un campo que después ningún filtro encuentra. La categoría se normaliza a minúsculas, para que `"Legal"` y `"legal"` coincidan con el mismo filtro.
- `process_and_upsert_batches`: recorre los documentos de a `batch_size` y hace **un `upsert` por lote**, no uno por vector. Cada llamada está dentro de un `try/except`.
- `search_by_category`: consulta con un filtro rígido (*hard filter*), `{"category": {"$eq": "legal"}}`. La búsqueda semántica solo compara los vectores de esa categoría.

```bash
python modulo-04-vector-db-cloud/ejercicio-ingesta-masiva/batch_ingestion.py
```

Salida con 5 documentos (2 `legal`, 2 `hr` y 1 `finanzas`) y `batch_size=2`:

```
2
2
1
Total processed: 5
query(top_k=5, filter={'category': {'$eq': 'legal'}})
Results: {'matches': []}
```

Son 3 llamadas a `upsert`, con 2, 2 y 1 vectores. El índice simulado devuelve `matches` vacío: lo que se verifica es que el filtro llegue bien armado.
