---
category: tareas
tags: [create_task, gather, taskgroup, as_completed, exceptiongroup]
---
# Tareas: create_task, gather y TaskGroup

> Guía técnica de la librería estándar `asyncio` de Python (3.11 o superior), escrita para este ejercicio.

Para que varias corutinas avancen al mismo tiempo hay que convertirlas en **tareas** (`asyncio.Task`). Una tarea envuelve una corutina y la programa en el event loop, que la ejecuta intercalada con las demás.

## asyncio.create_task

`asyncio.create_task(corutina)` programa la corutina y devuelve un objeto `Task` de inmediato, sin esperar a que termine. La tarea empieza a correr en el próximo momento en que el loop recupere el control.

```python
tarea = asyncio.create_task(descargar(url))
# ... otro trabajo ...
resultado = await tarea
```

Es importante **guardar una referencia** a cada tarea creada. El event loop solo mantiene referencias débiles a las tareas, así que una tarea sin ninguna variable que la apunte puede ser eliminada por el recolector de basura antes de terminar. El patrón habitual es guardarlas en una lista o en un conjunto.

## asyncio.gather

`asyncio.gather(*awaitables)` ejecuta varios awaitables en forma concurrente y devuelve una lista con sus resultados **en el mismo orden en que se pasaron**, no en el orden en que terminaron.

El parámetro `return_exceptions` cambia qué pasa cuando una tarea falla:

- Con `return_exceptions=False` (el valor por defecto), la primera excepción se propaga al código que hizo `await gather(...)`. Las demás tareas **no se cancelan**: siguen ejecutándose en segundo plano, aunque ya nadie espera su resultado.
- Con `return_exceptions=True`, las excepciones no se propagan: aparecen como un elemento más de la lista de resultados, en la posición de la tarea que falló. Así se pueden procesar los resultados exitosos y registrar los errores.

```python
resultados = await asyncio.gather(tarea_a(), tarea_b(), return_exceptions=True)
for r in resultados:
    if isinstance(r, Exception):
        print("falló:", r)
```

## asyncio.TaskGroup

Desde Python 3.11, `asyncio.TaskGroup` es la forma recomendada de lanzar un grupo de tareas relacionadas. Implementa la **concurrencia estructurada**: el bloque `async with` no termina hasta que terminan todas las tareas del grupo.

```python
async with asyncio.TaskGroup() as tg:
    t1 = tg.create_task(tarea_a())
    t2 = tg.create_task(tarea_b())
print(t1.result(), t2.result())
```

La diferencia clave con `gather` está en el manejo de errores. Si una tarea del grupo lanza una excepción, el `TaskGroup` **cancela automáticamente las demás tareas** y, al salir del bloque, lanza un `ExceptionGroup` con todas las excepciones ocurridas. Para manejarlas se usa la sintaxis `except*`:

```python
try:
    async with asyncio.TaskGroup() as tg:
        tg.create_task(tarea_a())
        tg.create_task(tarea_b())
except* ValueError as grupo:
    for error in grupo.exceptions:
        print("error de valor:", error)
```

En resumen: `gather` con `return_exceptions=True` sirve cuando cada tarea es independiente y conviene aprovechar los resultados parciales; `TaskGroup` sirve cuando las tareas forman una unidad y, si una falla, no tiene sentido que las otras sigan.

## asyncio.as_completed y asyncio.wait

`asyncio.as_completed(awaitables)` permite procesar los resultados **a medida que terminan**, en lugar de esperar a todos:

```python
for proxima in asyncio.as_completed([tarea_a(), tarea_b()]):
    resultado = await proxima
```

`asyncio.wait(tareas, return_when=...)` da más control. Con `return_when=asyncio.FIRST_COMPLETED` vuelve apenas termina la primera tarea; con `FIRST_EXCEPTION`, cuando una falla; y con `ALL_COMPLETED`, cuando terminan todas. Devuelve dos conjuntos, `done` y `pending`, y no cancela las tareas pendientes: eso queda a cargo de quien la llama.
