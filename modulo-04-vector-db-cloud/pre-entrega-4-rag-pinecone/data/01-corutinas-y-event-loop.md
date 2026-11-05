---
category: fundamentos
tags: [corutinas, event loop, asyncio.run, await]
---
# Corutinas y event loop

> Guía técnica de la librería estándar `asyncio` de Python (3.11 o superior), escrita para este ejercicio.

`asyncio` es la librería de Python para escribir código concurrente con la sintaxis `async` / `await`. Está pensada sobre todo para programas que pasan mucho tiempo esperando operaciones de entrada y salida: pedidos HTTP, consultas a una base de datos, lectura de sockets o llamadas a la API de un modelo de lenguaje.

## Qué es una corutina

Una función definida con `async def` es una **función corutina**. Llamarla no ejecuta su código: devuelve un **objeto corutina**, que recién se ejecuta cuando alguien lo espera con `await` o lo programa como tarea.

```python
async def saludar():
    return "hola"

c = saludar()        # no se ejecutó nada todavía
resultado = await c  # recién acá corre el cuerpo de la función
```

Un error muy común es llamar a una corutina sin `await`. Python no ejecuta el código y, cuando el objeto se descarta, muestra la advertencia `RuntimeWarning: coroutine 'saludar' was never awaited`. Si aparece ese mensaje, casi siempre falta un `await`.

## El event loop

El **event loop** (bucle de eventos) es el motor de `asyncio`. Mantiene una lista de tareas listas para ejecutarse y las va corriendo de a una. Cuando una tarea llega a un `await` sobre algo que todavía no terminó, por ejemplo una respuesta de red, la tarea se suspende y el loop pasa a ejecutar otra. Cuando la operación termina, la tarea suspendida vuelve a la lista de listas para continuar.

Todo esto ocurre en **un solo hilo**. Por eso se habla de **concurrencia cooperativa**: cada tarea cede el control voluntariamente en cada `await`. No hay dos corutinas ejecutando código Python al mismo tiempo, así que `asyncio` mejora los programas que esperan mucho, pero no acelera los cálculos que usan la CPU de forma intensiva.

## asyncio.run: el punto de entrada

La forma recomendada de arrancar un programa asíncrono es `asyncio.run(main())`. Esta función crea un event loop nuevo, ejecuta la corutina `main` hasta que termina, cancela las tareas que hayan quedado pendientes y cierra el loop.

```python
import asyncio

async def main():
    await asyncio.sleep(1)
    print("listo")

asyncio.run(main())
```

`asyncio.run()` no se puede llamar si ya hay un event loop corriendo en el mismo hilo: lanza `RuntimeError: asyncio.run() cannot be called from a running event loop`. Ese error es típico en Jupyter Notebook, que ya tiene un loop activo. Ahí la solución es escribir directamente `await main()` en la celda.

## Objetos awaitables

Se puede usar `await` con tres tipos de objetos, llamados **awaitables**:

- **Corutinas**: el resultado de llamar a una función `async def`.
- **Tasks**: corutinas programadas para ejecutarse en el loop, creadas con `asyncio.create_task()`.
- **Futures**: objetos de bajo nivel que representan un resultado que va a estar disponible más adelante. En el código de aplicación casi nunca se crean a mano.

## Ceder el control con asyncio.sleep

`await asyncio.sleep(segundos)` suspende la tarea actual durante ese tiempo y deja que el loop ejecute otras tareas mientras tanto. Un caso especial es `await asyncio.sleep(0)`: no espera nada, pero cede el control una vez, lo que permite que otras tareas avancen dentro de un bucle largo.

No hay que confundirlo con `time.sleep()`, que es bloqueante y detiene todo el event loop (ver la guía de código bloqueante).
