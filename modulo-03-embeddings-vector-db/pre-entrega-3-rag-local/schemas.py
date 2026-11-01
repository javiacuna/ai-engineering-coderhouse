"""Pydantic models of the RAG answer.

LLMAnswer is what the model must return (parsed with PydanticOutputParser). The model only cites
fragment numbers; the code turns them into full references, so a reference can never be invented.
"""

import re
from typing import List

from pydantic import BaseModel, Field, field_validator, model_validator

NOT_FOUND_ANSWER = "No lo sé"


def is_not_found(text: str) -> bool:
    """True for "No lo sé", "no lo se.", "No lo sé, el contexto no ...", etc."""
    normalized = re.sub(r"[^\w\s]", "", text.lower()).replace("é", "e").strip()
    return normalized.startswith("no lo se")


class LLMAnswer(BaseModel):
    answer: str = Field(
        description=(
            "Answer in Spanish, based ONLY on the context fragments. "
            f"If the context does not contain the answer, exactly: \"{NOT_FOUND_ANSWER}\""
        ),
    )
    fragment_ids: List[int] = Field(
        default_factory=list,
        description="Numbers [n] of the context fragments that support the answer. Empty list if the answer is \"No lo sé\".",
    )

    @field_validator("answer")
    @classmethod
    def clean_answer(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("answer cannot be empty")
        return value

    @model_validator(mode="after")
    def normalize_not_found(self) -> "LLMAnswer":
        # "I don't know" never carries sources, whatever the model wrote
        if is_not_found(self.answer):
            self.answer, self.fragment_ids = NOT_FOUND_ANSWER, []
        return self


class Reference(BaseModel):
    fragment: int = Field(description="Number [n] the fragment had in the prompt")
    source: str = Field(description="File in data/")
    title: str
    chunk_id: str
    distance: float = Field(description="Cosine distance to the question (0 = identical)")
    excerpt: str


class RAGResponse(BaseModel):
    question: str
    answer: str
    answered: bool = Field(description="False when the answer was not in the documents (\"No lo sé\")")
    references: List[Reference] = []
