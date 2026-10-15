"""
Ejercicio Módulo 1 - Orquestador concurrente de modelos (Coderhouse AI Engineering)

Simula un backend que consulta varios modelos de IA a la vez:
  - Parte A: dispara 3 modelos al mismo tiempo con asyncio.gather
  - Parte B: dispara 10 llamadas, pero un Semaphore deja correr solo 2 a la vez
En ambas, cada llamada tiene un timeout de 2 segundos.
"""

import asyncio
import time

TIMEOUT_SEGUNDOS = 2.0
MAX_CONCURRENTES = 2

# Guardamos el momento en que arranca el programa para mostrar tiempos relativos en los logs
INICIO = time.perf_counter()


def log(mensaje: str) -> None:
    """Imprime un mensaje con los segundos transcurridos desde el inicio."""
    transcurrido = time.perf_counter() - INICIO
    print(f"[{transcurrido:5.2f}s] {mensaje}")


# ---------------------------------------------------------------------------
# 1. Simulación de llamadas a modelos
#    Cada una "espera" un tiempo distinto, como si fuera la latencia de red.
# ---------------------------------------------------------------------------

async def gpt_4_call(prompt: str) -> str:
    log("gpt-4     -> inicia")
    await asyncio.sleep(1.0)  # await: cede el control mientras "espera la red"
    log("gpt-4     -> terminó")
    return f"gpt-4 respondió a '{prompt}'"


async def claude_3_call(prompt: str) -> str:
    log("claude-3  -> inicia")
    await asyncio.sleep(1.5)
    log("claude-3  -> terminó")
    return f"claude-3 respondió a '{prompt}'"


async def local_llama_call(prompt: str) -> str:
    log("llama     -> inicia")
    await asyncio.sleep(3.0)  # tarda más que el timeout: va a fallar a propósito
    log("llama     -> terminó")  # esta línea nunca se imprime
    return f"llama respondió a '{prompt}'"


# ---------------------------------------------------------------------------
# 2. Envoltorio con timeout: si la llamada tarda más de 2s, la cortamos
#    y devolvemos un mensaje de error en vez de romper el programa.
# ---------------------------------------------------------------------------

async def llamar_con_timeout(nombre: str, llamada) -> str:
    try:
        async with asyncio.timeout(TIMEOUT_SEGUNDOS):
            return await llamada
    except TimeoutError:
        log(f"{nombre:<9} -> TIMEOUT: superó {TIMEOUT_SEGUNDOS}s, se cancela")
        return f"ERROR: {nombre} no respondió a tiempo"


# ---------------------------------------------------------------------------
# 3. Parte A: los tres modelos al mismo tiempo
# ---------------------------------------------------------------------------

async def parte_a(prompt: str) -> None:
    log("=== PARTE A: 3 modelos en simultáneo con gather ===")
    resultados = await asyncio.gather(
        llamar_con_timeout("gpt-4", gpt_4_call(prompt)),
        llamar_con_timeout("claude-3", claude_3_call(prompt)),
        llamar_con_timeout("llama", local_llama_call(prompt)),
        return_exceptions=True,  # si algo explota igual, no tumba a los demás
    )
    for r in resultados:
        log(f"resultado: {r}")


# ---------------------------------------------------------------------------
# 4. Parte B: 10 llamadas, pero solo 2 a la vez (Semaphore)
# ---------------------------------------------------------------------------

async def llamada_limitada(numero: int, sem: asyncio.Semaphore) -> str:
    log(f"llamada {numero:>2} -> en la fila")
    async with sem:  # solo entran MAX_CONCURRENTES a la vez
        log(f"llamada {numero:>2} -> ENTRA (ejecutando)")
        # la llamada 7 la hacemos lenta para ver el timeout también acá
        latencia = 2.5 if numero == 7 else 0.5
        try:
            async with asyncio.timeout(TIMEOUT_SEGUNDOS):
                await asyncio.sleep(latencia)
        except TimeoutError:
            log(f"llamada {numero:>2} -> TIMEOUT")
            return f"llamada {numero}: ERROR timeout"
        log(f"llamada {numero:>2} -> sale")
        return f"llamada {numero}: OK"


async def parte_b() -> None:
    log(f"=== PARTE B: 10 llamadas, máximo {MAX_CONCURRENTES} a la vez ===")
    sem = asyncio.Semaphore(MAX_CONCURRENTES)
    tareas = [llamada_limitada(i, sem) for i in range(1, 11)]
    resultados = await asyncio.gather(*tareas, return_exceptions=True)
    ok = sum(1 for r in resultados if isinstance(r, str) and r.endswith("OK"))
    log(f"Resumen: {ok}/10 OK, {10 - ok} con error")


# ---------------------------------------------------------------------------
# 5. Punto de entrada
# ---------------------------------------------------------------------------

async def main() -> None:
    await parte_a("¿Qué es la entropía?")
    print()
    await parte_b()


if __name__ == "__main__":
    asyncio.run(main())
