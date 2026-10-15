# Módulo 2 — Orquestación con LangChain (LCEL)

## Ejercicio: Refactorización a LCEL asíncrono

📁 [`ejercicio-lcel-async/`](ejercicio-lcel-async/)

Migración de la llamada "cruda" al SDK (Módulo 1) a una cadena declarativa con LCEL:

```
{"pregunta": "..."} → ChatPromptTemplate → ChatGoogleGenerativeAI → StrOutputParser → str
```

- `ChatPromptTemplate.from_messages` con roles **system** y **human** y la variable `{pregunta}`.
- Modelo de chat de LangChain configurado con `model` y `temperature`.
- Cadena compuesta con el operador `|`: `prompt | modelo | parser`.
- Ejecución asíncrona con `await chain.ainvoke({"pregunta": ...})`.
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
