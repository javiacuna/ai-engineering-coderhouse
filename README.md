# AI Engineering — Coderhouse

Ejercicios y pre-entregas del curso **AI Engineering** de Coderhouse, en Python 3.12+.

## Contenido

| Módulo | Tema | Trabajos |
|---|---|---|
| [01](modulo-01-asyncio-y-clientes/) | Asincronía con asyncio y abstracción de clientes LLM | Orquestador concurrente · Cliente LLM unificado |
| [02](modulo-02-langchain-lcel/) | Orquestación con LangChain (LCEL) | Refactorización a LCEL asíncrono |

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
