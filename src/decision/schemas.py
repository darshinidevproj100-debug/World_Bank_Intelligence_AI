"""Typed contracts for the provisional decision-policy interface."""
from typing import Any, Literal
from pydantic import BaseModel, Field

Action = Literal["route_to_agent", "retrieve_more_evidence", "request_clarification", "request_review", "retry", "continue", "stop", "return_partial_result"]

class DecisionInput(BaseModel):
    execution_id: str
    question: str
    detected_intent: str | None = None
    candidate_agents: list[str] = Field(default_factory=list)
    available_evidence: list[dict[str, Any]] = Field(default_factory=list)
    evidence_coverage: float = 0.0
    confidence: float | None = None
    confidence_source: str = "heuristic"
    risk_level: Literal["low", "medium", "high"] = "low"
    task_status: str = "pending"
    missing_information: list[str] = Field(default_factory=list)
    remaining_steps: int = 8
    retrieval_attempts: int = 0
    max_retrieval_attempts: int = 2
    agent_calls: int = 0
    max_agent_calls: int = 8
    previous_actions: list[str] = Field(default_factory=list)
    tool_errors: list[str] = Field(default_factory=list)
    allowed_agents: list[str] = Field(default_factory=lambda: ["data_agent", "financial_agent", "project_agent", "research_agent"])
    ambiguous: bool = False
    conflicting_evidence: bool = False

class DecisionOutput(BaseModel):
    action: Action
    selected_agent: str | None = None
    reason: str
    evidence_required: bool = True
    confidence: float | None = None
    risk_level: Literal["low", "medium", "high"]
    stop: bool = False
    next_step: str
    decision_metadata: dict[str, Any] = Field(default_factory=dict)
