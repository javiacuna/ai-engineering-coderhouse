# Pre-entrega 5 · Agente de razonamiento cíclico con memoria persistente

Agente **ReAct** con **LangGraph** que responde consultas sobre clientes y pedidos de una tienda. El LLM decide por sí mismo qué herramientas usar y en qué orden, puede encadenar varias para llegar a una respuesta, se corrige cuando una herramienta devuelve un error, y recuerda la conversación de cada sesión (`thread_id`) en una base **SQLite**, incluso después de reiniciar el script.

| Criterio / checklist | Dónde está |
|---|---|
| **Autonomía**: el LLM decide cuándo usar una herramienta, sin `if/else` manuales | `agent.py` → `bind_tools(TOOLS)` + `tools_condition` |
| **Ciclo de retorno**: ante un error o un dato incompleto, reintenta o pide aclaración | Herramientas que devuelven `{"error", "sugerencia"}`, `ToolNode(handle_tool_errors=True)` y el prompt de sistema; se ve en las sesiones 2 y 3 de la traza |
| **Resiliencia de estado**: recuerda la sesión por `thread_id` | `AsyncSqliteSaver` → `data/checkpoints.sqlite`; `ask.py` lo prueba desde otro proceso |
| **Código limpio**: Python 3.12, type hints, asyncio | Todo el proyecto es asíncrono (`ainvoke`, `astream`, herramientas `async def`) |
| `StateGraph` que hereda de `MessagesState` | `agent.py` → `class AgentState(MessagesState)` |
| Nodo de modelo + nodo de herramientas + arista condicional `tools_condition` | `agent.py` → `build_graph()` |
| Al menos 1 herramienta con `@tool` y docstring descriptivo | `tools.py`: 3 herramientas con contrato Pydantic |
| LLM vinculado con `bind_tools()` | `agent.py` → `build_model()` |
| Persistencia con `SqliteSaver` + `thread_id` | `agent.py` → `sqlite_checkpointer()` |
| Razonamiento multi-paso (≥2 llamadas a herramientas) y `recursion_limit` | Sesiones 1 y 3 de `main.py`; `RECURSION_LIMIT = 10` |
| Traza de ejecución (.json o log) | `traces/react_trace.json` y `traces/react_trace.log` |

## Estructura

```
pre-entrega-5-agente-react/
├── database.py           # Base de datos simulada de clientes y pedidos (consultas async)
├── tools.py              # Herramientas (@tool + contrato Pydantic + docstrings para el LLM)
├── agent.py              # Estado, modelo, grafo, checkpointer SQLite y ejecución con traza
├── main.py               # Demo: 3 sesiones, 6 turnos → genera la traza
├── ask.py                # Una pregunta sobre un thread existente (prueba de persistencia)
├── traces/               # Traza ReAct de la corrida real (.json y .log)
├── data/                 # checkpoints.sqlite (se genera al correr; excluido del repo)
├── tests/                # 13 tests con un modelo guionado (sin API keys)
├── requirements.txt
└── .env.example
```

## Cómo levantar el entorno

Requiere **Python 3.12**.

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # y completá GOOGLE_API_KEY
```

**Proveedor del LLM:** la consigna menciona OpenAI o Anthropic. El agente soporta los dos (`LLM_PROVIDER=openai` o `anthropic`, con su key), y por defecto usa **Gemini**, que tiene plan gratuito. El grafo, las herramientas y la persistencia son los mismos con cualquiera de los tres.

```bash
python main.py                                                          # demo completa + traza
python ask.py --thread sesion-1 "¿Cuál era el total del primer cliente?"  # memoria desde otro proceso
python -m pytest -q                                                     # tests (sin API)
```

Si el modelo principal está saturado y tarda, se puede usar el liviano para una corrida:

```bash
GEMINI_MODEL=gemini-flash-lite-latest python main.py
```

## Arquitectura

```
START → agent ──(tools_condition: el LLM pidió herramientas)──► tools ──┐
          ▲                                                             │
          └──────────────────── observación ───────────────────────────┘
        agent ──(sin tool_calls: respuesta final)──► END
```

### Estado

```python
class AgentState(MessagesState):
    """Hereda messages: Annotated[list[AnyMessage], add_messages]"""
```

`add_messages` es el **reducer**: cada nodo devuelve solo sus mensajes nuevos y se agregan al historial. El estado nunca se modifica en el lugar: los nodos devuelven actualizaciones y LangGraph las fusiona. La consigna menciona `operator.add`; `add_messages` es su versión para mensajes: además de concatenar, si llega un mensaje con un `id` que ya existe, lo reemplaza en lugar de duplicarlo.

### Herramientas

| Herramienta | Qué hace | Cuándo la usa el LLM (según su docstring) |
|---|---|---|
| `search_customers(name)` | Busca clientes por nombre (sin importar tildes ni mayúsculas) | Cuando el usuario nombra a un cliente y hace falta su ID. Si hay varios, debe preguntar cuál. |
| `get_customer_orders(customer_id)` | Cantidad de pedidos, total gastado (sin los cancelados) y lista de pedidos | Para preguntas de cuántos pedidos, cuánto gastó o cuál fue el último |
| `get_order_detail(order_id)` | Productos, estado y monto de un pedido | Para saber qué se compró o en qué estado está un pedido |

- **Docstrings:** las descripciones dicen **cuándo** usar cada herramienta y **cómo se encadenan** ("si solo tenés el nombre, primero usá `search_customers`"). El LLM decide solo con esa información.
- **Contratos:** cada herramienta tiene un contrato Pydantic (`customer_id: int >= 1`, `name` de 2 a 100 caracteres).
- **Errores como datos:** un ID inexistente o una búsqueda sin resultados no lanzan una excepción. Devuelven `{"error": "...", "sugerencia": "..."}`, que el agente lee como observación y usa para corregirse.
- **Errores inesperados:** `ToolNode(handle_tool_errors=True)` convierte cualquier excepción, o argumentos inválidos, en un mensaje de error para el LLM, en lugar de romper el grafo.

### Ciclo de retorno ante errores

El prompt de sistema (`SYSTEM_PROMPT`) fija las reglas: nunca inventar datos, leer la `sugerencia` cuando una herramienta falla y reintentar con otros argumentos o con otra herramienta, y si no se puede resolver, o si hay varios clientes posibles, **preguntarle al usuario**.

### Persistencia

`AsyncSqliteSaver` es la versión asíncrona de `SqliteSaver`, necesaria porque el grafo corre con `ainvoke` y `astream`. Guarda un checkpoint después de cada nodo, por `thread_id`, en `data/checkpoints.sqlite`:

- Un segundo mensaje en el mismo thread recibe toda la conversación anterior.
- Un thread nuevo empieza de cero.
- Como está en un archivo, la memoria **sobrevive a un reinicio**: `ask.py` es otro proceso y continúa la conversación que creó `main.py`.

`main.py` borra sus tres sesiones al empezar (`adelete_thread`), así cada corrida genera una traza reproducible.

### Límites

- **`recursion_limit = 10`** pasos por turno: si el modelo entra en un loop de llamadas a herramientas, el turno se corta con un mensaje claro en lugar de seguir gastando tokens.
- **Estado sucio:** el checkpoint guarda el historial completo, pero al LLM solo se le envía la parte más reciente que entra en `MAX_CONTEXT_TOKENS = 4000` (`trim_messages`). El recorte siempre empieza en un mensaje del usuario, para no separar el resultado de una herramienta de la llamada que lo generó (las APIs rechazan eso).
- **Timeouts:** 60 segundos y 2 reintentos por llamada al LLM, con el modelo de respaldo si el principal falla.
- **Diseño:** el modelo y el checkpointer se pasan a `build_graph()` en lugar de guardarse en el estado. El estado solo tiene datos.

## La demo (`main.py`)

| Sesión | Turno | Qué demuestra |
|---|---|---|
| `sesion-1` | "¿Cuántos pedidos tuvo el cliente 102 y cuál fue el total?" | El ejemplo de la consigna: 3 pedidos, $14.500 |
| `sesion-1` | "¿Y el último?" | Memoria: entiende que se habla del cliente 102 y busca el detalle del pedido más reciente |
| `sesion-1` | "¿Quién gastó más en total: ese cliente o Diego Pereyra?" | Multi-paso: "ese cliente" sale de la memoria; busca a Diego y sus pedidos, y compara |
| `sesion-2` | "¿Cuánto gastó García en total?" | Información incompleta: hay dos García, así que pide aclaración |
| `sesion-2` | "El de Rosario" | La aclaración se entiende gracias a la memoria del turno anterior |
| `sesion-3` | "Mostrame los pedidos del cliente 999, es Sofía García" | Error y segundo intento: el cliente 999 no existe, busca a Sofía García por nombre y reintenta con el ID correcto |

## Resultados de la corrida real

Corrida con `gemini-flash-lite-latest`: **6 turnos, 9 llamadas a herramientas, 2 turnos multi-paso, 0 errores**. La traza completa está en [`traces/react_trace.log`](traces/react_trace.log) y [`traces/react_trace.json`](traces/react_trace.json).

**Sesión 1: el ejemplo de la consigna, memoria y razonamiento multi-paso**

```
usuario: ¿Cuántos pedidos tuvo el cliente 102 y cuál fue el total?
  → acción: get_customer_orders({'customer_id': 102})
  ← observación: {"pedidos": 3, "total": 14500, ...}
agente: El cliente 102 (Lucía Fernández) tuvo 3 pedidos y el monto total gastado es de $14.500.

usuario: ¿Y el último?
  → acción: get_order_detail({'order_id': 5012})
  ← observación: {"fecha": "2026-10-22", "estado": "en_camino", "monto": 5000, ...}
agente: El último pedido (ID 5012) fue realizado el 22/10/2026, está en estado en camino y su monto total es de $5.000. [...]

usuario: ¿Quién gastó más en total: ese cliente o Diego Pereyra?
  → acción: search_customers({'name': 'Diego Pereyra'})
  ← observación: {"coincidencias": 1, "clientes": [{"id": 104, ...}]}
  → acción: get_customer_orders({'customer_id': 104})
  ← observación: {"pedidos": 1, "total": 850000, ...}
agente: Gastó más Diego Pereyra, con un total de $850.000 en su pedido (el cliente 102, Lucía Fernández, gastó $14.500).
```

- **"¿Y el último?"**: el agente sacó el ID del pedido más reciente (5012) de la observación del turno anterior, guardada en el checkpoint, sin volver a consultar los pedidos.
- **La comparación**: "ese cliente" se resolvió con la memoria, y el total de Lucía no se volvió a pedir. Para Diego Pereyra encadenó dos herramientas: buscó su ID y después sus pedidos.

**Sesión 2: información incompleta → pide aclaración**

```
usuario: ¿Cuánto gastó García en total?
  → acción: search_customers({'name': 'García'})
  ← observación: {"coincidencias": 2, "clientes": [Martín García (Rosario), Sofía García (Mendoza)], "aviso": "..."}
agente: Hay más de un cliente con ese apellido. ¿Te referís a Martín García (de Rosario) o a Sofía García (de Mendoza)?

usuario: El de Rosario
  → acción: get_customer_orders({'customer_id': 101})
  ← observación: {"pedidos": 2, "pedidos_cancelados": 1, "total": 98000, ...}
agente: Martín García gastó un total de $98.000 (tiene un pedido entregado de $98.000 y otro de $120.000 que fue cancelado).
```

El agente no eligió un García al azar: preguntó. Y "El de Rosario" se entendió porque la lista de candidatos estaba en el historial del thread.

**Sesión 3: error de la herramienta → segundo intento**

```
usuario: Mostrame los pedidos del cliente 999, es Sofía García
  → acción: get_customer_orders({'customer_id': 999})
  ← observación: {"error": "El cliente 999 no existe.", "sugerencia": "Si el usuario mencionó un nombre, buscá el ID con search_customers; [...]"}
  → acción: search_customers({'name': 'Sofía García'})
  ← observación: {"coincidencias": 1, "clientes": [{"id": 103, ...}]}
  → acción: get_customer_orders({'customer_id': 103})
  ← observación: {"pedidos": 1, "total": 39500, ...}
agente: Sofía García (cliente ID 103) tiene 1 pedido registrado por un total de $39.500 [...]
```

El primer intento falló, el agente leyó la `sugerencia`, buscó el ID correcto por nombre y reintentó: **3 llamadas** para llegar a la respuesta.

**Persistencia entre procesos (`ask.py`)**

```
$ python ask.py --thread sesion-1 "¿Cuál era el total del primer cliente?"
Thread 'sesion-1': 14 mensajes guardados antes de esta pregunta
agente: El total del primer cliente (Lucía Fernández, ID 102) es de $14.500.
(0 llamadas a herramientas, 16 mensajes en el thread)
```

`ask.py` es otro proceso: la conversación de `main.py` ya había terminado, y la memoria se cargó desde `data/checkpoints.sqlite` (14 mensajes). El agente respondió **sin llamar a ninguna herramienta**, porque el dato ya estaba en la conversación.

## Tests

```bash
python -m pytest -q
```

Usan un modelo guionado que devuelve llamadas a herramientas predefinidas, así que corren sin API keys. Cubren:

- **Herramientas:** tildes, ambigüedad, el total de la consigna, los cancelados y los errores con sugerencia.
- **Ciclo ReAct:** dos llamadas encadenadas, y que la observación vuelve al agente.
- **Recuperación:** error y segundo intento; argumentos inválidos devueltos como observación.
- **Memoria:** el mismo thread recuerda y otro no; la memoria **sobrevive a un reinicio**, con un grafo y un checkpointer nuevos sobre el mismo archivo.
- **Límites:** el `recursion_limit` frena un loop infinito, y el recorte del historial deja pares válidos.
- **Base nueva:** se puede borrar un thread antes de cualquier escritura.
