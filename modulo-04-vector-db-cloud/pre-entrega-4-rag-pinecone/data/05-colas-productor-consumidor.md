---
category: colas
tags: [queue, productor, consumidor, task_done, join, backpressure]
---
# Colas: el patrón productor-consumidor

> Guía técnica de la librería estándar `asyncio` de Python (3.11 o superior), escrita para este ejercicio.

`asyncio.Queue` es una cola pensada para pasar elementos entre tareas. Es la base del patrón **productor-consumidor**: unas tareas generan trabajo y lo ponen en la cola, y un grupo de tareas trabajadoras (*workers*) lo va sacando y procesando.

## Operaciones básicas

- `await cola.put(elemento)` agrega un elemento. Si la cola está llena, espera hasta que haya lugar.
- `await cola.get()` saca el próximo elemento. Si la cola está vacía, espera hasta que llegue uno.
- `cola.put_nowait(elemento)` y `cola.get_nowait()` hacen lo mismo sin esperar: lanzan `asyncio.QueueFull` o `asyncio.QueueEmpty` si no pueden completar la operación.
- `cola.qsize()` devuelve cuántos elementos hay, y `empty()` y `full()` consultan el estado.

## maxsize y backpressure

`asyncio.Queue(maxsize=100)` crea una cola con capacidad limitada. Cuando está llena, `put()` suspende al productor hasta que un consumidor saque un elemento. Este mecanismo se llama **backpressure** (contrapresión): evita que un productor rápido llene la memoria con trabajo que los consumidores todavía no pueden procesar. Con `maxsize=0`, el valor por defecto, la cola no tiene límite.

## task_done y join

Una cola lleva la cuenta de las **tareas sin terminar**. Cada `put()` suma uno a ese contador, y cada llamada a `cola.task_done()` resta uno. `await cola.join()` espera hasta que el contador llegue a cero, es decir, hasta que **todos los elementos que entraron fueron procesados**, no solo sacados de la cola.

Por eso cada consumidor debe llamar a `task_done()` después de terminar de procesar cada elemento, idealmente en un bloque `finally` para que se llame aunque el procesamiento falle. Si se llama a `task_done()` más veces que elementos se pusieron, se lanza `ValueError`.

```python
async def worker(nombre, cola):
    while True:
        url = await cola.get()
        try:
            await descargar(url)
        finally:
            cola.task_done()

async def main(urls):
    cola = asyncio.Queue(maxsize=50)
    workers = [asyncio.create_task(worker(f"w{i}", cola)) for i in range(5)]
    for url in urls:
        await cola.put(url)
    await cola.join()          # espera a que se procesen todas las URLs
    for w in workers:
        w.cancel()             # los workers tienen un bucle infinito: hay que cancelarlos
    await asyncio.gather(*workers, return_exceptions=True)
```

En este ejemplo, la cantidad de workers (5) limita cuántas descargas ocurren a la vez, y `maxsize=50` limita cuántas URLs pueden estar esperando en la cola.

## Cerrar una cola: shutdown

Desde Python 3.13, `cola.shutdown()` cierra la cola. Después de llamarlo, `put()` lanza `asyncio.QueueShutDown`, y `get()` también la lanza cuando la cola queda vacía. Esto permite que los workers terminen solos, sin tener que cancelarlos. Con `shutdown(immediate=True)` se descartan además los elementos que quedaban pendientes.

## Otros tipos de cola

- `asyncio.PriorityQueue` devuelve primero el elemento con menor prioridad. Se suelen guardar tuplas `(prioridad, dato)`.
- `asyncio.LifoQueue` devuelve primero el último elemento agregado, como una pila.

Ninguna de estas colas es segura entre hilos: para comunicar hilos se usa `queue.Queue` del módulo `queue`.
