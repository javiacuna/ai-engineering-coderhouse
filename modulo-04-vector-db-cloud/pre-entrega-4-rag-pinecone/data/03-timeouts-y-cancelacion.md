---
category: control-de-tiempo
tags: [timeout, wait_for, cancelacion, cancellederror, shield]
---
# Timeouts y cancelación

> Guía técnica de la librería estándar `asyncio` de Python (3.11 o superior), escrita para este ejercicio.

En un programa que depende de servicios externos, ninguna espera debería ser infinita. `asyncio` ofrece herramientas para limitar el tiempo de una operación y para cancelar tareas de forma ordenada.

## asyncio.timeout

Desde Python 3.11, la forma recomendada de poner un límite de tiempo es el administrador de contexto `asyncio.timeout(segundos)`:

```python
try:
    async with asyncio.timeout(5):
        respuesta = await llamar_api()
except TimeoutError:
    print("la API tardó más de 5 segundos")
```

Cuando vence el plazo, `asyncio` cancela la tarea que está dentro del bloque. Dentro de la corutina eso se ve como un `CancelledError`, pero al salir del bloque `async with` se convierte en un **`TimeoutError`**, que es la excepción que hay que capturar. Desde Python 3.11, `asyncio.TimeoutError` es simplemente un alias de la excepción incorporada `TimeoutError`.

El plazo también se puede expresar como un momento absoluto con `asyncio.timeout_at(cuando)`, usando el reloj del loop (`loop.time()`). Además, el objeto que devuelve `asyncio.timeout()` tiene un método `reschedule()` para extender o acortar el plazo mientras el bloque está corriendo.

## asyncio.wait_for

`asyncio.wait_for(awaitable, timeout)` es la alternativa más antigua: espera el awaitable durante como máximo `timeout` segundos. Si el tiempo se agota, cancela la tarea y lanza `TimeoutError`. Sigue siendo útil cuando se quiere limitar una única llamada en una sola línea:

```python
resultado = await asyncio.wait_for(llamar_api(), timeout=5)
```

## Cancelar una tarea

`tarea.cancel()` pide la cancelación de una tarea. La cancelación no es inmediata: en el próximo `await` de esa corutina se lanza un `asyncio.CancelledError` dentro de ella. La tarea puede atraparlo para liberar recursos, pero **debe volver a lanzarlo**. Si lo traga y sigue como si nada, la cancelación no se completa y las herramientas que dependen de ella, como `TaskGroup` o `asyncio.timeout`, dejan de funcionar correctamente.

```python
async def trabajador():
    try:
        while True:
            await procesar_siguiente()
    except asyncio.CancelledError:
        await cerrar_conexiones()
        raise   # siempre volver a lanzarla
```

Desde Python 3.8, `CancelledError` hereda de `BaseException` y no de `Exception`. Por eso un `except Exception:` genérico no la atrapa por accidente, lo que evita que una cancelación se pierda. Para limpiar recursos tanto si la tarea termina bien como si la cancelan, lo más simple es un bloque `finally`.

## asyncio.shield

`asyncio.shield(awaitable)` protege una operación de la cancelación. Si la tarea que hace `await asyncio.shield(operacion())` es cancelada, la espera se interrumpe, pero la operación protegida **sigue ejecutándose** hasta terminar. Sirve para operaciones que no conviene dejar a la mitad, como confirmar una transacción o terminar de escribir un archivo.

## Buenas prácticas

- Poner un timeout a cada llamada de red, en especial a las APIs de modelos de lenguaje, que pueden tardar mucho bajo carga.
- Preferir `asyncio.timeout()` cuando el límite abarca varias operaciones juntas.
- No atrapar `CancelledError` sin volver a lanzarlo.
- Combinar timeouts con reintentos y espera exponencial para los errores temporales.
