"""Output contract of the technical entity extraction pipeline.

Python attributes are in English; the JSON keys and enum values follow the course
contract in Spanish (tecnologias, nivel_de_criticidad, resumen_tecnico) through aliases.
"""

from enum import Enum
from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Criticality(str, Enum):
    LOW = "baja"
    MEDIUM = "media"
    HIGH = "alta"


# Filler values a model may invent to satisfy "non-empty list" (found in the stress test)
PLACEHOLDERS = {"no especificada", "no especificado", "no especificadas", "desconocida", "desconocido",
                "ninguna", "ninguno", "n/a", "na", "none", "unknown", "not specified", "-"}


class TechnicalExtraction(BaseModel):
    """Technical entities extracted from a software architecture description or an error log."""

    model_config = ConfigDict(populate_by_name=True)

    technologies: List[str] = Field(
        alias="tecnologias",
        min_length=1,
        description=(
            "Concrete technologies mentioned in the text: languages, frameworks, databases, "
            "message brokers, cloud services, tools. Use their official names (e.g. 'PostgreSQL', not 'postgres')."
        ),
    )
    criticality: Criticality = Field(
        alias="nivel_de_criticidad",
        description=(
            "Operational impact: 'alta' = outage, data loss, security issue or users blocked; "
            "'media' = degraded performance or partial errors; 'baja' = informative, no visible impact."
        ),
    )
    technical_summary: str = Field(
        alias="resumen_tecnico",
        min_length=20,
        max_length=400,
        description="One or two sentences in Spanish summarizing the architecture or the problem and its likely cause.",
    )

    @field_validator("technologies")
    @classmethod
    def clean_technologies(cls, values: List[str]) -> List[str]:
        """Trim, drop empty items, reject placeholders and deduplicate (case-insensitive), keeping the order."""
        seen, cleaned = set(), []
        for value in values:
            value = value.strip()
            if value.lower() in PLACEHOLDERS:
                raise ValueError(f"'{value}' is a placeholder, not a technology mentioned in the text")
            if value and value.lower() not in seen:
                seen.add(value.lower())
                cleaned.append(value)
        if not cleaned:
            raise ValueError("tecnologias must contain at least one technology")
        return cleaned

    @field_validator("technical_summary")
    @classmethod
    def clean_summary(cls, value: str) -> str:
        return " ".join(value.split())

    def to_contract_json(self) -> str:
        """JSON with the keys defined by the contract (tecnologias, nivel_de_criticidad, resumen_tecnico)."""
        return self.model_dump_json(by_alias=True, indent=2)
