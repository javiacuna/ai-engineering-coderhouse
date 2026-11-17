# Capturas del dashboard de Phoenix

Sacalas **después de correr la prueba de carga**:

```bash
python scripts/load_test.py
```

Abrí <http://localhost:6006> → proyecto **`pe7-api-agente`**. Arriba a la derecha, poné el rango de tiempo en **Last 15 Min** (o **Last Hour**): con el rango de 7 días, la corrida entera queda en un solo punto del gráfico.

| Archivo | Pantalla | Qué tiene que verse |
|---|---|---|
| `01-trazas.png` | Pestaña **Traces** | Las 5 trazas `job.start`, con las columnas **latency**, **total tokens** y **total cost** (si no aparecen, activalas en **Columns**) |
| `02-arbol-de-una-traza.png` | Click en una traza | El árbol: `job.start → LangGraph → assess_risk → run_orchestrator → supervisor / researcher / analyst / synthesizer → ChatGoogleGenerativeAI`, las herramientas y los spans `db.*.search`; arriba, **Total Cost** y **Latency** |
| `03-latencia-p95.png` | Pestaña **Metrics** → gráfico **Trace latency** | Pasá el mouse sobre el punto de la corrida: aparece un recuadro con p50, p75, p90, **p95**, p99 y max |
| `04-costo.png` | Pestaña **Metrics** → gráficos **Cost** y **Top models by cost** | El costo estimado en USD, separado en prompt y completion, y por modelo |
| `05-tokens-por-nodo.png` | Pestaña **Spans**, filtro `span_kind == 'LLM'` | Las llamadas al LLM con su estado, latencia y tokens. En **Columns**, destildá input, output y las de anotaciones para que entren **latency**, **total tokens** y **total cost**. El agente de cada llamada no aparece en la tabla: el reparto por agente lo calcula `python scripts/token_breakdown.py` |
| `06-hitl-sesion.png` (opcional) | Pestaña **Sessions** → un job que pasó por aprobación | Las dos trazas del mismo job: `job.start`, que termina en `WAITING_APPROVAL`, y `job.resume` |

Los mismos números (costo por ejecución, p50 y p95) quedan en `results/load_test.md`: el script los pide a Phoenix por GraphQL, así que tienen que coincidir con las capturas.
