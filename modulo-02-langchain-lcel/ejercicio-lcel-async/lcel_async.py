"""
Ejercicio Módulo 2 - Refactorización a LCEL Asíncrono (Coderhouse AI Engineering)

Flujo:  {"pregunta": "..."} -> ChatPromptTemplate -> ChatGoogleGenerativeAI -> StrOutputParser -> str

Se usa Google Gemini por su plan gratuito. Para usar OpenAI o Anthropic solo cambia
el bloque del modelo (ver README del módulo); la cadena queda igual.
"""

import asyncio
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Carga las variables del archivo .env (ej. GOOGLE_API_KEY) al entorno
load_dotenv()

# 1. Modelo: reemplaza al cliente "crudo" del SDK del Módulo 1
modelo = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
    temperature=0.3,
)

# 2. Prompt con roles System / Human y una variable: {pregunta}
prompt = ChatPromptTemplate.from_messages([
    ("system", "Sos un profesor de física. Respondé en español, claro y en no más de 3 oraciones."),
    ("human", "{pregunta}"),
])

# 3. Parser: extrae solo el texto del AIMessage
parser = StrOutputParser()

# 4. La cadena LCEL: cada | pasa la salida de un paso como entrada del siguiente
chain = prompt | modelo | parser


async def main() -> None:
    # 5. Ejecución asíncrona: las llaves del dict coinciden con {pregunta} del prompt
    respuesta = await chain.ainvoke({"pregunta": "¿Qué es la entropía?"})

    print("¿Es texto plano?", isinstance(respuesta, str))  # True -> no es un AIMessage
    print(respuesta)


if __name__ == "__main__":
    asyncio.run(main())
