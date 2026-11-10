# Ejercicio: Instrumentación y análisis de trazas con Arize Phoenix

Instrumentación del **orquestador multi-agente de la pre-entrega 6** (supervisor, investigador, analista y sintetizador) con **Arize Phoenix** y **OpenInference**, más un análisis automático de las trazas: latencia, tokens, costos, errores y groundedness.

| Archivo | Qué hace |
|---|---|
| `tracing.py` | Conecta el exportador de OpenTelemetry al Phoenix local e instrumenta LangChain/LangGraph y las **fuentes de datos** |
| `run_traffic.py` | Genera el tráfico: 5 consultas complejas y 2 fallas provocadas, cada una con metadatos y la versión del prompt |
| `analyze_traces.py` | Lee las trazas desde Phoenix y calcula el análisis que pide el ejercicio |
| `evaluate_faithfulness.py` | Evaluador **LLM-as-a-judge** de fidelidad: compara cada respuesta con lo que devolvieron las herramientas y carga el veredicto en Phoenix como anotación |
| `results/` | `traffic_run.json` (las corridas), `trace_analysis.md` y `trace_analysis.json` (el análisis) |
| `tests/` | Tests de las funciones del análisis (no necesitan Phoenix) |

**Documento de hallazgos:** [Hallazgos de observabilidad: orquestador multi-agente en Phoenix](https://claude.ai/artifact/FQZMgzh54kRjNs64RrqhAd): latencia, groundedness, errores, tokens y recomendaciones, con los datos de la corrida `orquestador-m7-20261109-2002`. Copia en PDF: [`hallazgos-observabilidad.pdf`](hallazgos-observabilidad.pdf).

El orquestador **no se modifica**: los scripts lo importan desde `modulo-06-sistemas-multiagente/pre-entrega-6-orquestador-multiagente/` y lo instrumentan desde afuera.

## Cómo ejecutarlo

**1. Instalar** (en el `.venv` del repo, Python 3.12):

```bash
pip install arize-phoenix arize-phoenix-otel openinference-instrumentation-langchain
```

**2. Levantar Phoenix** en una terminal aparte (queda corriendo):

```bash
python -m phoenix.server.main serve
```

El dashboard queda en <http://localhost:6006>.

**3. Generar el tráfico** en otra terminal:

```bash
cd modulo-07-observabilidad/ejercicio-phoenix-trazas
GEMINI_MODEL=gemini-flash-lite-latest python run_traffic.py
```

Tarda unos 5 minutos: entre consulta y consulta espera 30 segundos, por el límite de pedidos por minuto del plan gratuito de Gemini. Al terminar, imprime el nombre del proyecto de Phoenix.

**4. Analizar:**

```bash
python analyze_traces.py
```

Imprime las tablas y las guarda en `results/trace_analysis.md`.

**5. Evaluar la fidelidad con un juez LLM** (Phoenix tiene que seguir corriendo):

```bash
python evaluate_faithfulness.py
```

Usa `gemini-3.8-flash` como juez (se cambia con `GEMINI_JUDGE_MODEL`), un modelo distinto del que escribió las respuestas, para que no se juzgue a sí mismo. Guarda `results/faithfulness_eval.md` y carga dos anotaciones en el span raíz de cada consulta: `faithfulness` (del juez, `annotator_kind="LLM"`) y `numeric_grounding` (del chequeo numérico, `annotator_kind="CODE"`). En Phoenix se ven en la columna de anotaciones de cada traza.

## Qué se instrumenta

| Componente | Cómo | Tipo de span |
|---|---|---|
| Grafo de LangGraph, nodos (supervisor, researcher, analyst, synthesizer) y agentes ReAct | `LangChainInstrumentor` de OpenInference | `CHAIN`, `AGENT` |
| Llamadas a Gemini (con tokens de entrada y salida, y modelo) | `LangChainInstrumentor` | `LLM` |
| Herramientas (`search_reviews`, `search_prices`, `analyze_sentiment`, `compute_statistics`) | `LangChainInstrumentor` | `TOOL` |
| **Fuentes de datos** (reseñas y precios) | Spans manuales en `tracing.py`, con la consulta y los documentos devueltos | `RETRIEVER` |
| Cada consulta completa | Span raíz `consulta:<escenario>` en `run_traffic.py` | `CHAIN` |

**La "mancha ciega" de la base de datos:** las fuentes no son objetos de LangChain, así que el instrumentador no las ve. Por eso `tracing.py` las envuelve en spans `RETRIEVER`. Para que esos spans queden **dentro de la herramienta que los llamó**, y no colgando de la raíz, el span padre se toma con `get_current_span()` de OpenInference: el instrumentador de LangChain crea sus spans desde callbacks y no los marca como "actuales" en el contexto de OpenTelemetry. Así, la jerarquía queda completa:

```
consulta:q1-comparacion
└── LangGraph
    ├── supervisor → ChatGoogleGenerativeAI (LLM)
    ├── researcher
    │   └── LangGraph (agente ReAct)
    │       ├── agent → ChatGoogleGenerativeAI (LLM)
    │       └── tools → search_reviews (TOOL) → db.reviews.search (RETRIEVER)
    ├── analyst → ... → analyze_sentiment / compute_statistics (TOOL)
    └── synthesizer → ChatGoogleGenerativeAI (LLM)
```

**Atributos en todos los spans** (con `using_attributes` de OpenInference):

- `session_id`: el nombre del proyecto, uno por sesión;
- `tags`: el tipo de consulta (`compleja`, `falla: fuera de alcance`, `falla: herramienta desconectada`);
- `metadata.scenario`: el id del escenario;
- `prompt_template_version`: la versión de los prompts. Es la buena práctica del "Error 2" de la unidad: si una alucinación aparece después de cambiar un prompt, las trazas muestran qué versión la produjo.

**Sesiones separadas:** cada corrida de `run_traffic.py` crea un **proyecto nuevo de Phoenix** (`orquestador-m7-AAAAMMDD-HHMM`), así las métricas de una sesión, como el p95, no se mezclan con corridas anteriores.

## El tráfico

| Escenario | Tipo | Qué prueba |
|---|---|---|
| `q1-comparacion` | compleja | Dos productos: sentimiento, precio promedio y recomendación |
| `q2-ranking` | compleja | Ranking de tres productos por sentimiento y relación precio-valoración |
| `q3-quejas` | compleja | Principales quejas y la tienda más barata |
| `q4-presupuesto` | compleja | Un presupuesto que el modelo tiene que contrastar con los precios reales |
| `q5-uso` | compleja | Recomendación según un uso concreto (comodidad, durabilidad, precio) |
| `f1-fuera-de-alcance` | falla | Un producto que no está en las fuentes: la respuesta correcta es decir que no hay datos |
| `f2-fuente-caida` | falla | La fuente de reseñas lanza `ConnectionError`: el error tiene que verse en el span de la base y en cómo se propaga |

## Qué calcula `analyze_traces.py`

| Pedido del ejercicio | Cálculo |
|---|---|
| **Span de mayor latencia** | El span `LLM`, `TOOL` o `RETRIEVER` más largo de cada consulta, con el agente al que pertenece |
| **Latencia de red vs. de procesamiento** | **Externa**: llamadas a la API del LLM (red + generación) más fuentes de datos. **Espera**: huecos de más de 1 s sin ningún span trabajando, como el backoff después de un 429. **Local**: el resto, o sea el cálculo de las herramientas más la orquestación. Se usa la **unión de intervalos**, para no contar dos veces las llamadas en paralelo |
| **p50 y p95** | De las consultas completas y de las llamadas individuales al LLM |
| **Tokens y costo** | Tokens de entrada y salida por consulta y por agente, y el detalle del **razonamiento más largo**. El costo se estima con precios del plan pago: el plan gratuito no cobra |
| **Groundedness** | Cada número de la respuesta final tiene que aparecer en las salidas de las herramientas de **esa** traza, o en la pregunta del usuario (se admite redondeo). Un número sin fuente es evidencia de alucinación. En las consultas sin datos, la respuesta tiene que reconocerlo |
| **Errores** | Dónde se originó cada error (`TOOL`, `RETRIEVER` o `LLM`) y a cuántos spans superiores se propagó |

**Supuesto de precios** (USD por millón de tokens, plan pago, según [morphllm.com](https://www.morphllm.com/gemini-api-pricing), actualizado al 25/10/2026): `gemini-3.8-flash` cuesta $0,75 de entrada y $3,75 de salida. `gemini-flash-lite-latest` se valúa como `gemini-3.5-flash-lite`, a $0,30 y $2,50, porque el alias no dice a qué versión apunta. Los precios se pueden cambiar en `PRICES`.

**Limitación del chequeo de groundedness:** es determinístico, así que solo verifica **números**. Una afirmación inventada sin números, como "el iPhone 17 todavía no salió", no la detecta. Para eso está `evaluate_faithfulness.py`, el juez LLM. Pero el juez también puede equivocarse: es el "Error 3" de la unidad, ningún evaluador automático es una verdad absoluta. Por eso se guardan los dos veredictos lado a lado, y los casos en que no coinciden son los que conviene revisar a mano.

## Tests

```bash
python -m pytest -q
```

Verifican:
- la unión de intervalos;
- la lectura de números en notación argentina e inglesa ($182.000, $88.666,67, 4,2, 0.33);
- la separación de las esperas (backoff) del procesamiento local;
- la detección de números inventados, con tolerancia al redondeo;
- el desempaquetado de la salida de las herramientas;
- la asignación de cada span a su agente.
