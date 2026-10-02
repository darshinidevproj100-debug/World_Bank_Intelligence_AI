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


class EvidenceValidationResult(BaseModel):
    """Deterministic source consistency findings, not semantic entailment proof."""
    is_valid: bool
    groundedness_status: Literal["supported_references", "insufficient_evidence", "invalid_citations", "not_semantically_verified"]
    supported_claims: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    citation_errors: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    recommended_action: str = "continue_with_limitations"
    validation_notes: list[str] = Field(default_factory=list)
