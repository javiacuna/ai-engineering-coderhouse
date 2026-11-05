---
category: rendimiento
tags: [bloqueante, to_thread, run_in_executor, debug, time.sleep]
---
# Código bloqueante y depuración

> Guía técnica de la librería estándar `asyncio` de Python (3.11 o superior), escrita para este ejercicio.

El event loop ejecuta todas las tareas en un solo hilo. Si una tarea ejecuta una operación **bloqueante**, es decir, una que no cede el control con `await`, el loop entero se detiene: ninguna otra tarea avanza, no se procesan respuestas de red y los timeouts no se disparan hasta que esa operación termina. El síntoma típico es un programa asíncrono que "se congela" o que tarda lo mismo que si fuera secuencial.

## Operaciones bloqueantes comunes

- `time.sleep()`: hay que reemplazarlo por `await asyncio.sleep()`.
- Clientes HTTP síncronos como `requests.get()`: conviene usar uno asíncrono, como `httpx.AsyncClient` o `aiohttp`.
- Drivers de bases de datos síncronos.
- Lectura y escritura de archivos grandes.
- Cálculos pesados de CPU: procesar imágenes, entrenar un modelo o comprimir datos.

## asyncio.to_thread

Cuando no hay una alternativa asíncrona, `asyncio.to_thread(funcion, *args)` (Python 3.9 o superior) ejecuta una función bloqueante en un **hilo aparte** del pool por defecto y devuelve una corutina que se puede esperar. Mientras tanto, el event loop sigue atendiendo a las demás tareas.

```python
import requests

async def obtener(url):
    respuesta = await asyncio.to_thread(requests.get, url, timeout=10)
    return respuesta.json()
```

`to_thread` es la solución adecuada para operaciones bloqueantes de **entrada y salida**. Para cálculos pesados de CPU no alcanza: por el GIL de Python, un hilo que calcula sigue compitiendo con el resto. En ese caso hay que usar procesos.

## loop.run_in_executor

`loop.run_in_executor(executor, funcion, *args)` es la herramienta de más bajo nivel. Permite elegir el executor, por ejemplo un `concurrent.futures.ProcessPoolExecutor` para ejecutar cálculos de CPU en otros procesos:

```python
from concurrent.futures import ProcessPoolExecutor

async def main():
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor() as pool:
        resultado = await loop.run_in_executor(pool, calcular_estadisticas, datos)
```

Con `executor=None` usa el pool de hilos por defecto, igual que `to_thread`. La función y sus argumentos tienen que poder serializarse (*pickle*) para enviarse a otro proceso.

## Modo debug

`asyncio` tiene un **modo debug** que ayuda a encontrar estos problemas. Se activa con `asyncio.run(main(), debug=True)` o con la variable de entorno `PYTHONASYNCIODEBUG=1`. En modo debug:

- Se registra una advertencia cada vez que un callback o un paso de una tarea tarda más de `loop.slow_callback_duration`, que por defecto es **100 milisegundos**. Es la forma más directa de encontrar el código que bloquea el loop.
- Se detectan corutinas que nunca se esperaron y se muestra dónde se crearon.
- Se verifica que las APIs que no son seguras entre hilos se llamen desde el hilo correcto.

## Otros mensajes frecuentes

- `Task was destroyed but it is pending!`: una tarea quedó sin terminar cuando se cerró el loop o fue eliminada por el recolector de basura porque nadie guardaba una referencia a ella.
- `Task exception was never retrieved`: una tarea falló y nadie consultó su resultado con `await` ni con `result()`, así que la excepción se perdió.

En los dos casos, la solución es guardar las tareas creadas y esperarlas, o usar un `TaskGroup`, que se ocupa de eso automáticamente.
