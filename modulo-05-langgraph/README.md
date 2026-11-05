# Módulo 5 · Agentes con LangGraph

## Ejercicio: Diseño de un grafo de soporte técnico multinivel

📁 [`ejercicio-diseno-grafo/`](ejercicio-diseno-grafo/)

Diagrama y esquema del estado (`TypedDict`) de un asistente de soporte. Clasifica la duda (Triage), busca una solución (Search_Knowledge), la valida (Evaluator) y, si no sirve, vuelve a buscar: ese es el **ciclo**. Después de 3 intentos fallidos, o si el caso es crítico, lo escala a una persona (Human_Support): el grafo se pausa con `interrupt()` hasta que esa persona responde (*human-in-the-loop* con checkpoints), y le pasa el historial de por qué fallaron los intentos. Incluye la implementación ejecutable con LangGraph, que recorre los tres caminos posibles.

![Diagrama del grafo](ejercicio-diseno-grafo/diagrama.png)

## Ejercicio: Tool Calling con contrato validado

📁 [`ejercicio-tool-calling/`](ejercicio-tool-calling/)

`search_knowledge_base` es una herramienta asíncrona de LangChain (`@tool(args_schema=SearchInput)`) que un agente puede invocar para buscar en la base de conocimiento.

- **Contrato de entrada (`SearchInput`):** `query` (de 3 a 500 caracteres; se limpian los espacios de más) y `limit` (de 1 a 10, por defecto 3). Pydantic valida los argumentos que manda el LLM **antes** de que la función toque la base de datos: un `limit` de 50 se rechaza.
- **Descripción para el LLM:** el docstring explica qué busca, cuándo conviene usarla (preguntas sobre reglas y procesos propios de la empresa), cuándo no (saludos, cálculos, temas generales) y qué devuelve. Es lo que el modelo lee para decidir si la llama.
- **Errores:** la búsqueda corre dentro de un `try` con un timeout de 10 segundos. Si la base falla o tarda demasiado, la herramienta no corta la ejecución: devuelve un texto que empieza con `"Error:"`, para que el LLM le explique al usuario qué pasó en lugar de inventar una respuesta.

**Estructura:**

| Archivo | Contenido |
|---|---|
| `search_tool.py` | Solo la definición de la herramienta (`SearchInput` y `search_knowledge_base`): es lo que importa un agente |
| `demo.py` | Demostración manual: contrato, llamada válida, `limit=50` rechazado y base caída |
| `tests/test_search_tool.py` | 12 tests con pytest que automatizan esas mismas verificaciones |

```bash
cd modulo-05-langgraph/ejercicio-tool-calling
python demo.py
python -m pytest -q
```

Salida de `demo.py`:

```
search_knowledge_base {'query': {'description': "Pregunta o tema a buscar, ...", 'maxLength': 500, 'minLength': 3, ...}, 'limit': {'default': 3, 'description': 'Cantidad máxima de fragmentos a devolver, entre 1 y 10. Por defecto, 3.', 'maximum': 10, 'minimum': 1, ...}}
['Resultado para políticas de vacaciones']
Validation error: ('limit',) -> Input should be less than or equal to 10
Search failed for 'políticas de vacaciones': ConnectionError: vector store unreachable
Error: no se pudo consultar la base de conocimiento (ConnectionError).
```

**Tests** (reemplazan la base por versiones simuladas con `monkeypatch`):

| Grupo | Qué verifican |
|---|---|
| Contrato | Nombre, descripción, campos `query` y `limit` con sus descripciones y límites |
| Éxito | La búsqueda devuelve resultados y recibe los argumentos correctos; `limit` vale 3 por defecto; la `query` llega sin espacios de más; sin resultados, devuelve un mensaje |
| Entrada inválida | `limit` 50 o 0, `query` muy corta, solo espacios o ausente: se rechaza con `ValidationError` y **la base nunca se consulta** |
| Base caída | Un `ConnectionError` o un error inesperado vuelve como texto `"Error: ..."` en lugar de cortar la ejecución; una base lenta corta por timeout |
