# AI Engineering — Coderhouse

> **English summary.** Exercises and project milestones from Coderhouse's *AI Engineering* course, in Python 3.12+: async orchestration with `asyncio`, a unified async LLM client for OpenAI / Anthropic / Gemini with Pydantic validation and streaming, LangChain LCEL chains with structured output, retries and fallbacks, an evaluation of embeddings vs. keyword search with an LLM re-ranker, and token-based document chunking. Code and comments are in English; the documentation is in Spanish because the course is taught in Spanish.

Ejercicios y pre-entregas del curso **AI Engineering** de Coderhouse, en Python 3.12+.

## Contenido

| Módulo | Tema | Trabajos |
|---|---|---|
| [01](modulo-01-asyncio-y-clientes/) | Asincronía con asyncio y abstracción de clientes LLM | Orquestador concurrente · Cliente LLM (ejercicio) · **Pre-entrega 1: Unified Async LLM Client** |
| [02](modulo-02-langchain-lcel/) | Orquestación con LangChain (LCEL) | LCEL asíncrono · Validación estructurada con retry y fallback |
| [03](modulo-03-embeddings-vector-db/) | Embeddings y similitud semántica | Embeddings y similitud (informe PDF) · Limpieza y chunking por tokens |

## Cómo correr los ejemplos

```bash
# 1. Clonar el repo
git clone https://github.com/javiacuna/ai-engineering-coderhouse.git
cd ai-engineering-coderhouse

# 2. Crear y activar un entorno virtual
python3 -m venv .venv
source .venv/bin/activate      # en Windows: .venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar las API keys
cp .env.example .env           # y completar las keys en .env
```

Después, cada carpeta de ejercicio explica cómo ejecutarlo en el README de su módulo.

> Las API keys se leen desde variables de entorno o desde el archivo `.env`, que está excluido en `.gitignore`.
