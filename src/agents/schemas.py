"""Common structured result schema for agents and workflow stages."""
from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator


class AgentResult(BaseModel):
    """Agent response with an answer, evidence references, and explicit limits."""

    agent: str
    status: Literal["success", "partial", "error"]
    summary: str
    answer: str | None = None
    data: Any = None
    sources: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def default_answer_to_summary(self) -> "AgentResult":
        """Keep older callers compatible while exposing an explicit answer field."""
        if self.answer is None:
            self.answer = self.summary
        return self
