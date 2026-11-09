# Módulo 6 · Sistemas multi-agente

## Ejercicio: Topología jerárquica para respuesta automatizada a quejas

📁 [`ejercicio-topologia-jerarquica/`](ejercicio-topologia-jerarquica/)

Propuesta arquitectónica de un sistema que recibe quejas de clientes por Twitter, analiza el sentimiento, verifica el pedido en la base de datos y responde de forma personalizada. Lo resuelven un **supervisor** y cuatro especialistas: Sentimiento, Pedidos, Redacción y Cumplimiento. El documento incluye:

- las responsabilidades, herramientas y contratos de cada agente;
- el estado compartido;
- el flujo paso a paso, con un ejemplo;
- la justificación de la topología jerárquica frente a la cooperativa, y cómo se mitigan sus riesgos.

![Diagrama](ejercicio-topologia-jerarquica/diagrama.png)

## Ejercicio: Patrón Supervisor con delegación dinámica

📁 [`ejercicio-patron-supervisor/`](ejercicio-patron-supervisor/)

Un supervisor (LLM) decide en cada paso qué especialista trabaja, el Analista o el Escritor, y cuándo terminar. El supervisor **coordina**: no analiza ni escribe.

```
START → supervisor ──(Analista)──► Analista ──┐
            ▲       ──(Escritor)──► Escritor ──┤
            └──────────────────────────────────┘
        supervisor ──(FINALIZAR)──► END
```

- **Estado:** `messages` (con el reducer `add_messages`), `next` (la última decisión del supervisor) y `reason` (su motivo, para la traza).
- **Decisión estructurada:** el supervisor usa `with_structured_output(Route)`, un `TypedDict` con `next_agent: Literal["Analista", "Escritor", "FINALIZAR"]` y un `reason`. El modelo solo puede devolver una de esas tres rutas.
- **Prompt de sistema:** reglas en orden, que el supervisor verifica en el historial (busca los mensajes con nombre `Analista` o `Escritor`):
  1. **Fuera de alcance** (un saludo, una pregunta general) → `FINALIZAR` sin llamar a nadie.
  2. **Ambiguo**: pide un informe pero no dice sobre qué tema, o el tema no es medible → `FINALIZAR` sin llamar a nadie, con la pregunta de aclaración en `reason`. Ante la duda, aclarar: es más barato preguntar que analizar lo que no se pidió.
  3. Si el Analista no respondió → `Analista`.
  4. Si respondió el Analista pero no el Escritor → `Escritor`.
  5. Si el Escritor entregó → `FINALIZAR`, sin volver a llamar a nadie (esto evita el bucle).
- **Respuesta al usuario sin delegar:** cuando el supervisor termina sin que trabaje ningún especialista (casos 1 y 2), le responde al usuario con el motivo o la pregunta, en lugar de cortar en silencio.
- **Especialistas que dependen del pedido:**
  - El **Analista** lee el pedido del usuario (el último mensaje que no viene de un agente del equipo), detecta el tema (ventas, clientes, inventario o costos) y devuelve los datos de ese tema.
  - El **Escritor** arma el informe a partir del resultado del Analista.
  - Si el tema no existe, lo dicen en lugar de inventar.
  - El tema se detecta por **palabras completas**: comparando pedazos de texto, "inventario" se confundía con "venta" (in-**venta**-rio). Lo detectó un test.
- **Resguardos:** si el LLM devuelve una ruta inválida, el grafo termina en lugar de romperse; y `recursion_limit=10` corta cualquier bucle.
- **Proveedor:** la consigna usa OpenAI. Esta versión usa Gemini por defecto, con un modelo de respaldo; con `LLM_PROVIDER=openai` usa `gpt-4o-mini`.

```bash
cd ejercicio-patron-supervisor
python supervisor.py
python -m pytest -q
```

Ejemplo de salida (los textos del `reason` los escribe el LLM, así que varían):

```
Usuario: Necesito un reporte sobre ventas.
[supervisor] decide: Analista (...)
[Analista] Datos procesados (ventas): +20% respecto del período anterior; ingresos de $12,4 M contra $10,3 M del período anterior.
[supervisor] decide: Escritor (...)
[Escritor] El informe de ventas está listo: Crecimiento positivo. +20% respecto del período anterior; [...]
[supervisor] decide: FINALIZAR (...)
Secuencia: supervisor → Analista → supervisor → Escritor → supervisor → fin

Usuario: Armame un informe de inventario.
[Analista] Datos procesados (inventario): -12% respecto del período anterior; rotación de 34 días, 6 productos con quiebre de stock.
[Escritor] El informe de inventario está listo: Requiere atención. [...]

Usuario: Hacé un reporte.
[supervisor] decide: FINALIZAR (¿Sobre qué tema querés el reporte: ventas, clientes, inventario o costos?)
[supervisor → usuario] ¿Sobre qué tema querés el reporte: ventas, clientes, inventario o costos?
Secuencia: supervisor → fin
```

**Tests** (11, con un router simulado, sin API):

- **Flujo:** la secuencia esperada, también con otro tema; que el supervisor recibe su prompt y todo el historial; que el estado final tiene los aportes de los dos especialistas.
- **Pedidos especiales:** uno fuera de alcance termina sin especialistas, y uno ambiguo recibe una pregunta de aclaración.
- **Especialistas:** el Analista responde según el tema pedido; el Escritor usa el análisis; la detección del tema compara palabras completas.
- **Resguardos:** una ruta inválida termina sin romper el grafo, y un supervisor que nunca dice `FINALIZAR` se corta por el `recursion_limit`.

## Ejercicio: Estado compartido y comunicación asíncrona

📁 [`ejercicio-estado-compartido/`](ejercicio-estado-compartido/)

Dos agentes asíncronos, `investigador` y `revisor`, se comunican **solo a través del estado del grafo**, con artefactos validados por Pydantic.

```
START → investigador → revisor → END
```

- **Contratos:** `ResearchArtifact` (`topic`, `findings` con al menos un hallazgo no vacío y `confidence_score` entre 0 y 1) y `ReviewArtifact` (`is_approved`, `feedback`). Los datos se validan **antes** de entrar al estado.
- **Aislamiento de escritura:** cada nodo devuelve solo su parte del estado (`researcher_output` o `reviewer_output`). Además, el decorador `@owns(...)` declara qué claves puede escribir cada nodo, y lanza `StateOwnershipError` si un nodo intenta pisar la parte de otro: el aislamiento queda **garantizado**, no solo documentado.
- **Errores compartidos sin pisarse:** `errors: Annotated[list[str], operator.add]`. Cualquier nodo puede **agregar** su error y el reducer concatena. Es la alternativa segura a una lista global con `.append()` (el "Error 1" de la unidad).
- **La llamada externa, protegida:** la investigación está separada en `fetch_research()`, que simula la llamada a un LLM o a una API de búsqueda. Corre dentro de `asyncio.timeout(5)` y de un `try/except` que cubre **la llamada completa**, no solo la validación:

  | Falla | Qué pasa |
  |---|---|
  | `ValidationError` (por ejemplo, `confidence_score=1.5`) | El artefacto inválido nunca entra al estado |
  | `TimeoutError` (la fuente tarda más de 5 s) | Se registra que la fuente no respondió |
  | Error de red o del proveedor (`ResearchSourceError`, `ConnectionError`, errores de la API) | Se registra el tipo y el mensaje |

  En todos los casos, el error queda en `errors`, el investigador devuelve `researcher_output=None` y el revisor igual se ejecuta y rechaza por falta de datos válidos. **Un fallo de red no detiene el orquestador** ni deja el grafo "en el limbo" (el "Error 2").
- **Mensaje de error defensivo:** `describe_validation_error()` no asume que el error tiene una ubicación simple. Si `loc` viene vacío (un error de validación de todo el modelo), usa "artefacto"; si es anidado (`("findings", 1)`), lo muestra como `findings.1`.
- **Concurrencia real:** `main()` corre los 4 escenarios con `asyncio.gather(..., return_exceptions=True)`. Las esperas de 1 segundo se **superponen**: el total es ~1 s en lugar de ~4 s. Cada línea lleva el nombre de su escenario (`[alta]`, `[red caída]`) para distinguir las salidas intercaladas, y una falla inesperada en un escenario no cancela los demás.
- **Dependencias:** la arista `investigador → revisor` garantiza que el revisor lee el estado recién cuando el investigador terminó de escribir.
- **Configuración fuera del estado:** el `confidence_score` simulado y la falla de red de cada escenario se pasan por `config["configurable"]`, no por el estado, porque son parámetros de prueba y no datos del dominio.

```bash
cd ejercicio-estado-compartido
python shared_state.py
python -m pytest -q
```

Salida (las líneas de los 4 escenarios se intercalan porque corren a la vez):

```
[alta] --- Investigando: Sistemas Multi-Agente ---
[baja] --- Investigando: Sistemas Multi-Agente ---
[inválido] --- Investigando: Sistemas Multi-Agente ---
[red caída] --- Investigando: Sistemas Multi-Agente ---
[inválido]     Error manejado → investigador: artefacto inválido (confidence_score: Input should be less than or equal to 1)
[red caída]     Error manejado → investigador: falló la fuente externa (ResearchSourceError: la API de búsqueda no respondió (conexión rechazada))
[alta] Nodo 'investigador' finalizado.
[...]
[alta] Revisión: APROBADA — Aprobado: 3 hallazgos con confianza 0.92.
[baja] Revisión: RECHAZADA — Rechazado: la confianza 0.60 no supera 0.8. Se necesitan más fuentes que respalden los hallazgos.
[inválido] Revisión: RECHAZADA — No hay resultados de investigación válidos para revisar.
[red caída] Revisión: RECHAZADA — No hay resultados de investigación válidos para revisar.

===== Resumen (1.0 s en total; en secuencia serían ~4 s) =====
[alta] APROBADA | errores: 0
[baja] RECHAZADA | errores: 0
[inválido] RECHAZADA | errores: 1
[red caída] RECHAZADA | errores: 1
```

**Tests** (16):

- **Contratos:** el score fuera de rango y los hallazgos vacíos se rechazan.
- **Escenarios:** aprobado, rechazado y el umbral estricto (0.8 no alcanza); un score inválido se maneja sin cortar.
- **Llamada externa:** una caída de red, un timeout y una excepción inesperada se manejan, y el revisor igual se ejecuta.
- **Mensaje defensivo:** funciona con `loc` vacío y con `loc` anidado.
- **Escritura:** cada nodo devuelve solo su clave; un nodo que escribe la clave de otro es rechazado; los errores se agregan sin pisarse.
- **Concurrencia:** `main()` corre los escenarios en paralelo (tarda menos que la suma de las esperas y las salidas se intercalan), y también se superponen tres corridas independientes del grafo.
