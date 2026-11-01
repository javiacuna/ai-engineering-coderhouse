# Módulo 2 — Orquestación con LangChain (LCEL)

## Ejercicio: Refactorización a LCEL asíncrono

📁 [`ejercicio-lcel-async/`](ejercicio-lcel-async/)

Migración de la llamada "cruda" al SDK (Módulo 1) a una cadena declarativa con LCEL:

```
{"question": "..."} → ChatPromptTemplate → ChatGoogleGenerativeAI → StrOutputParser → str
```

- `ChatPromptTemplate.from_messages` con roles **system** y **human** y la variable `{question}`.
- Modelo de chat de LangChain configurado con `model` y `temperature`.
- Cadena compuesta con el operador `|`: `prompt | modelo | parser`.
- Ejecución asíncrona con `await chain.ainvoke({"question": ...})`.
- `StrOutputParser` devuelve texto plano en lugar del `AIMessage` completo.

### Proveedor del modelo

Se usa **Google Gemini** (`gemini-3.8-flash`) por su plan gratuito. Requiere `GOOGLE_API_KEY`
en el `.env` de la raíz (se obtiene gratis en [aistudio.google.com](https://aistudio.google.com)).

```bash
python ejercicio-lcel-async/lcel_async.py
```

Gracias a LCEL, cambiar de proveedor solo modifica el bloque del modelo; la cadena no cambia:

```python
# OpenAI  (requiere OPENAI_API_KEY)
from langchain_openai import ChatOpenAI
modelo = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# Anthropic  (requiere ANTHROPIC_API_KEY)
from langchain_anthropic import ChatAnthropic
modelo = ChatAnthropic(model="claude-sonnet-5-5", temperature=0.3)
```

## Ejercicio: Validación estructurada y resiliencia en cadenas

📁 [`ejercicio-validacion-estructurada/`](ejercicio-validacion-estructurada/)

Extrae información de un texto y la devuelve como un objeto Pydantic validado, no como texto libre.

**Esquema `EntityExtraction` (Pydantic v2)**
- `topic: str` (no vacío), `entities: List[str]`, `sentiment_score: float` entre 0 y 1 (`Field(ge=0, le=1)`).
- `language: Optional[str]` con `pattern=r"^[a-z]{2}$"` (código ISO de idioma, opcional).
- `@field_validator("entities")`: limpia espacios, descarta vacías, deduplica sin importar mayúsculas y rechaza la lista si queda vacía.

**Resiliencia en capas**
1. `.with_structured_output(EntityExtraction)`: el modelo responde con una instancia del esquema.
2. `.with_retry(stop_after_attempt=3, wait_exponential_jitter=True)`: hasta 3 intentos con backoff.
3. `.with_fallbacks([...])`: si se agotan los reintentos, usa un modelo alternativo (`GEMINI_FALLBACK_MODEL`).
4. Si también falla, el `except` devuelve un **resultado degradado** (`topic="unknown"`, `entities=[]`, sentimiento neutral) en vez de cortar el programa.

Usa `logging` (con hora y nivel) en lugar de `print` para dejar trazabilidad de errores y degradaciones.

Variables necesarias en `.env`: `GOOGLE_API_KEY`, `GEMINI_MODEL` y `GEMINI_FALLBACK_MODEL` (ver `.env.example`).

```bash
python ejercicio-validacion-estructurada/structured_validation.py
```
