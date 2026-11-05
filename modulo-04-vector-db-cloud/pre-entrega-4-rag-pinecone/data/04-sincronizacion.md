---
category: sincronizacion
tags: [semaphore, lock, event, condition, concurrencia]
---
# Primitivas de sincronización

> Guía técnica de la librería estándar `asyncio` de Python (3.11 o superior), escrita para este ejercicio.

Aunque `asyncio` corre en un solo hilo, las tareas se intercalan en cada `await`, así que también pueden competir por un recurso compartido. `asyncio` incluye primitivas de sincronización parecidas a las del módulo `threading`, pero pensadas para corutinas: sus métodos se usan con `await` o con `async with`.

Estas primitivas **no son seguras entre hilos** (*thread-safe*): están pensadas para coordinar tareas del mismo event loop.

## Semaphore: limitar la concurrencia

`asyncio.Semaphore(n)` permite que como máximo `n` tareas estén dentro de una sección al mismo tiempo. Es la herramienta indicada para **limitar cuántas operaciones se ejecutan a la vez**, por ejemplo para no superar el límite de pedidos por minuto de una API o para no abrir demasiadas conexiones a una base de datos.

```python
semaforo = asyncio.Semaphore(5)

async def descargar(url):
    async with semaforo:          # como máximo 5 descargas en paralelo
        return await cliente.get(url)

resultados = await asyncio.gather(*(descargar(u) for u in urls))
```

Aunque se lancen 100 tareas con `gather`, solo 5 pueden estar dentro del bloque `async with semaforo` al mismo tiempo; las demás esperan a que se libere un lugar.

Internamente, el semáforo tiene un contador: `acquire()` lo decrementa y espera si llegó a cero, y `release()` lo incrementa. `asyncio.BoundedSemaphore` es una variante que lanza `ValueError` si se llama a `release()` más veces que a `acquire()`, lo que ayuda a detectar errores de programación.

## Lock: exclusión mutua

`asyncio.Lock()` garantiza que **una sola tarea a la vez** ejecute una sección de código. Es útil para proteger un recurso que no admite accesos simultáneos, como un archivo que se modifica en varios pasos o un caché que se recalcula:

```python
lock = asyncio.Lock()

async def actualizar_cache():
    async with lock:
        if cache_vencido():
            await recalcular_cache()
```

Un `Lock` es equivalente a un `Semaphore(1)`, pero expresa mejor la intención.

## Event: avisar que algo ocurrió

`asyncio.Event()` sirve para que varias tareas esperen una señal. Las tareas hacen `await evento.wait()` y quedan suspendidas hasta que otra tarea llama a `evento.set()`; en ese momento se despiertan todas. `evento.clear()` vuelve a poner el evento en estado "no ocurrido", e `is_set()` consulta su estado.

Un uso típico es esperar a que el sistema termine de inicializarse, por ejemplo a que se cargue un modelo, antes de empezar a atender pedidos.

## Condition: esperar una condición

`asyncio.Condition()` combina un `Lock` con la posibilidad de esperar a que se cumpla una condición. Una tarea adquiere la condición, comprueba un estado y, si no le sirve, llama a `await condicion.wait()`, que libera el lock mientras espera. Otra tarea cambia el estado y llama a `notify()` o `notify_all()` para despertar a las que esperan. El método `wait_for(predicado)` simplifica el patrón de esperar en un bucle hasta que una función devuelva verdadero.

## Cuál elegir

| Necesidad | Primitiva |
|---|---|
| Limitar cuántas tareas hacen algo a la vez | `Semaphore` |
| Que una sola tarea a la vez toque un recurso | `Lock` |
| Avisar a varias tareas que algo ya ocurrió | `Event` |
| Esperar a que un estado compartido cumpla una condición | `Condition` |
