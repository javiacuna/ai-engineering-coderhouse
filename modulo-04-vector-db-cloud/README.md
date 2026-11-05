# Módulo 4 · Bases vectoriales en la nube

## Ejercicio: Pinecone Serverless

📁 [`ejercicio-pinecone-serverless/`](ejercicio-pinecone-serverless/)

`setup_vector_infrastructure(index_name, dimension)` automatiza la infraestructura con el SDK de Python de Pinecone:

1. Crea el cliente con `PINECONE_API_KEY`, que se lee desde el `.env`.
2. Si el índice no existe, lo crea como **serverless** (`aws` / `us-east-1`, la región del plan gratuito) con `dimension=1536` y métrica **coseno**. Después espera a que `status.ready` sea verdadero.
3. Si el índice ya existía con otra dimensión, frena con un error claro en lugar de fallar en el upsert (*mismatch* de dimensiones).
4. Carga 3 vectores de prueba con metadatos (`topic`, `priority`) en el namespace **`dev-environment`**. Como es un `upsert`, correrlo dos veces no duplica los vectores.
5. Devuelve `describe_index_stats()`. Las estadísticas tardan unos segundos en actualizarse, así que las vuelve a consultar hasta que aparecen los vectores (máximo 30 segundos).

```bash
python modulo-04-vector-db-cloud/ejercicio-pinecone-serverless/pinecone_setup.py
```

Salida esperada: `dimension: 1536`, el namespace `dev-environment` con 3 vectores y `total_vector_count: 3`.

**Dimensión y modelo de embeddings:** 1536 es la dimensión de `text-embedding-3-small` de OpenAI. Con Gemini, `gemini-embedding-001` devuelve 3072 dimensiones por defecto, pero acepta `output_dimensionality=1536` para generar vectores compatibles con este índice.
