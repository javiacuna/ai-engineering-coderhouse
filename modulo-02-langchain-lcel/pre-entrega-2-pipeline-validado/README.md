# Pre-entrega 2 · Pipeline de extracción de entidades técnicas

Pipeline con **LangChain (LCEL)** que recibe un texto sin procesar (una descripción de arquitectura o un log de error) y devuelve un **objeto validado con Pydantic** con las tecnologías mencionadas, el nivel de criticidad y un resumen técnico.

| Requisito | Dónde está |
|---|---|
| Esquema Pydantic (`tecnologias`, `nivel_de_criticidad`, `resumen_tecnico`) | `schemas.py` → `TechnicalExtraction` |
| Prompt template modular (texto + instrucciones de formato) | `chain.py` → `PROMPT` (`ChatPromptTemplate`, sin f-strings) |
| Cadena LCEL `prompt | model.with_structured_output(Schema)` | `chain.py` → `build_chain()` |
| Reintento ante JSON mal formado o incompleto | `.with_retry()` + validación de `finish_reason` en `check_output()` |
| Función asíncrona `process_text()` con `.ainvoke()` y logs | `chain.py` → `process_text()` |
| Mini-script de prueba asíncrono | `main.py` |

## Estructura

```
pre-entrega-2-pipeline-validado/
├── schemas.py        # Contrato de salida (Pydantic)
├── chain.py          # Configuración, prompt, cadena LCEL, validación y process_text()
├── main.py           # Script de prueba: 3 textos, incluido uno ambiguo
├── tests/            # 22 tests con un modelo simulado (no necesitan API keys)
├── requirements.txt
└── .env.example
```

## Instalación

Requiere **Python 3.12+**.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # y completá la key del proveedor que uses
```

## Variables de entorno

| Variable | Obligatoria | Descripción |
|---|---|---|
| `LLM_PROVIDER` | No (default `gemini`) | `openai`, `anthropic` o `gemini` |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY` | La del proveedor activo | Con Gemini también se acepta `GEMINI_API_KEY` |
| `LLM_MODEL` | No | Por defecto `gpt-4o-mini`, `claude-haiku-4-5` o `gemini-3.8-flash` |
| `LLM_FALLBACK_MODEL` | No | Modelo de respaldo si el principal falla (por ejemplo, saturado con un 503) |
| `LLM_MAX_TOKENS` | No (1024) | Límite de tokens de la respuesta |
| `LLM_MAX_ATTEMPTS` | No (3) | Intentos totales cuando la respuesta llega cortada o mal formada |

Con Gemini también se leen `GEMINI_MODEL` y `GEMINI_FALLBACK_MODEL`, las mismas variables que usan los ejercicios del módulo 2.

## Uso

```bash
python main.py
```

El script procesa tres textos: una arquitectura con un cuello de botella, un log de error de Kafka y Kubernetes, y un **texto ambiguo** como prueba de estrés.

Desde código:

```python
from chain import process_text

resultado = await process_text("Nuestra API en FastAPI usa Redis como caché delante de PostgreSQL...")
print(resultado.to_contract_json())
```

### Ejemplo de salida (corrida real con `gemini-3.8-flash`)

Entrada: *"Nuestra API está hecha con FastAPI y usa Redis como caché delante de PostgreSQL. Desde el lunes, en horario pico, las respuestas pasan de 80 ms a 4 segundos y el pool de conexiones de Postgres se agota; varios clientes reportan timeouts al pagar."*

```json
{
  "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "Se registra una degradación severa de rendimiento y timeouts en pagos durante el horario pico debido al agotamiento del pool de conexiones en la base de datos PostgreSQL, posiblemente por un fallo o alta presión en la caché de Redis."
}
```

Logs del proceso de validación:

```
01:54:06 | INFO    | extraction_pipeline | Processing text (246 characters)
01:54:45 | INFO    | extraction_pipeline | Valid output (finish_reason=stop, 3 technologies)
01:54:45 | INFO    | extraction_pipeline | Extraction OK: criticality=alta, technologies=['FastAPI', 'Redis', 'PostgreSQL']
```

Con el log de error de Kafka/Kubernetes devolvió `["Apache Kafka", "Kubernetes", "Google Kubernetes Engine", "Java"]` con criticidad `alta`: reconoció GKE a partir del nombre del node pool (`gke-n2-standard-4`).

## Prueba de estrés: lo que encontró

Con el texto ambiguo (*"Anda raro desde ayer. Creo que es la base o algo del server…"*), la primera versión devolvió:

```json
{ "tecnologias": ["No especificada"], "nivel_de_criticidad": "media", "resumen_tecnico": "..." }
```

Como el esquema exige una lista **no vacía**, el modelo **inventó un valor de relleno** para cumplir la regla, y Pydantic lo aceptó porque era un string válido. Es un dato falso que pasaba la validación. Se corrigió en dos capas:

1. **Validador:** `clean_technologies` rechaza valores de relleno (`"No especificada"`, `"N/A"`, `"desconocido"`, etc.). Esa respuesta pasa a ser inválida y dispara un **reintento**.
2. **Prompt:** si el texto no nombra tecnologías concretas, el modelo debe listar los componentes genéricos que sí menciona (por ejemplo `"base de datos"`, `"servidor"`) y nunca usar valores de relleno.

Resultado con la corrección (corrida real):

```json
{
  "tecnologias": ["base de datos", "servidor"],
  "nivel_de_criticidad": "media",
  "resumen_tecnico": "El sistema presenta problemas intermitentes de rendimiento y disponibilidad que afectan la carga, probablemente originados en el servidor o en la base de datos."
}
```

Los tests `test_placeholders_are_rejected` y `test_placeholder_answer_triggers_a_retry` cubren este caso.

## Cómo funciona la resiliencia

```
PROMPT | model.with_structured_output(TechnicalExtraction, include_raw=True) | check_output
        └──────────────────────── .with_retry(3 intentos) ───────────────────────┘
                                   .with_fallbacks([modelo de respaldo])   (opcional)
```

1. **Validación del esquema.** `with_structured_output` le pasa el esquema al modelo y Pydantic valida la respuesta: la lista de tecnologías no puede quedar vacía ni tener valores de relleno como "No especificada" (y se limpia y deduplica), la criticidad tiene que ser `baja`, `media` o `alta`, y el resumen debe tener entre 20 y 400 caracteres.
2. **Detección de respuestas incompletas (`finish_reason`).** Con `include_raw=True`, `check_output()` recibe también la respuesta cruda del modelo. Si el `finish_reason` indica que se cortó por el límite de tokens (`length` en OpenAI, `max_tokens` en Anthropic, `MAX_TOKENS` en Gemini), la descarta **antes** de usarla, aunque el JSON parcial pareciera válido.
3. **Reintentos.** Una respuesta cortada, mal formada o que no cumple el esquema lanza `IncompleteOutputError`, y `.with_retry()` vuelve a llamar al modelo. Solo se reintentan esos errores. Una API key inválida no se reintenta, porque no tiene sentido.
4. **Fallback (opcional).** Si se agotan los intentos o el proveedor falla, se usa el modelo de `LLM_FALLBACK_MODEL`.
5. **Error controlado.** Si todo falla, `process_text()` lanza una única excepción, `ExtractionError`, con un mensaje claro. Cada paso queda registrado en los logs.

## Tests

```bash
python -m pytest -q
```

Los tests reemplazan el LLM por un modelo simulado, así que corren sin keys ni internet. Cubren las validaciones del esquema, el prompt modular, la detección de respuestas cortadas en los tres proveedores, los reintentos (respuesta cortada → mal formada → válida), el rechazo de valores de relleno, el abandono tras el máximo de intentos y la configuración.

## Decisiones de diseño

- **Código en inglés, contrato en español.** Los atributos de Python se llaman `technologies`, `criticality` y `technical_summary`, pero con **alias** de Pydantic el esquema que ve el modelo y el JSON de salida usan las claves que pide la consigna (`tecnologias`, `nivel_de_criticidad`, `resumen_tecnico`).
- **`with_structured_output` en lugar de `PydanticOutputParser`.** Usa la salida estructurada nativa de cada proveedor, que es más confiable que pedir JSON en el texto. Con OpenAI se usa `method="function_calling"`, porque el modo `json_schema` estricto no admite restricciones como "lista no vacía".
- **Prompt modular.** Las instrucciones de formato se inyectan con `.partial(format_instructions=...)`, así el prompt solo espera `{text}` y las instrucciones se pueden cambiar sin tocar la cadena.
- **Mismo criterio que el cliente del Módulo 1.** El proveedor se elige por variable de entorno y las keys se guardan como `SecretStr`, para que no aparezcan en logs.
