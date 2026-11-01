"""
Module 2 exercise - Refactoring to async LCEL (Coderhouse AI Engineering)

Flow:  {"question": "..."} -> ChatPromptTemplate -> ChatGoogleGenerativeAI -> StrOutputParser -> str

Google Gemini is used because of its free tier. To use OpenAI or Anthropic, only the
model block changes (see the module README); the chain stays the same.
"""

import asyncio
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Load variables from the .env file (e.g. GOOGLE_API_KEY) into the environment
load_dotenv()

# 1. Model: replaces the raw SDK client from Module 1
model = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
    temperature=0.3,
)

# 2. Prompt with System / Human roles and one variable: {question}
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a physics teacher. Answer in Spanish, clearly, in at most 3 sentences."),
    ("human", "{question}"),
])

# 3. Parser: extracts only the text from the AIMessage
parser = StrOutputParser()

# 4. The LCEL chain: each | passes the output of one step as the input of the next
chain = prompt | model | parser


async def main() -> None:
    # 5. Async execution: the dict keys must match {question} in the prompt
    answer = await chain.ainvoke({"question": "¿Qué es la entropía?"})

    print("Is it plain text?", isinstance(answer, str))  # True -> not an AIMessage
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
