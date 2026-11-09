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
