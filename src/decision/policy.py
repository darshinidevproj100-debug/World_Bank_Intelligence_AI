"""Deterministic provisional policy pending owner-supplied formal JEV rules."""
from __future__ import annotations

from datetime import datetime, timezone
import uuid
from typing import Any
from src.decision.rules import evidence_is_sufficient
from src.decision.schemas import DecisionInput, DecisionOutput

class PrototypeDecisionPolicy:
    """Explainable bounded policy. Thresholds and retry limits are configurable."""
    def __init__(self, *, evidence_threshold: float = .6, high_risk_review: bool = True):
        if not 0 <= evidence_threshold <= 1: raise ValueError("evidence_threshold must be between 0 and 1")
        self.evidence_threshold, self.high_risk_review = evidence_threshold, high_risk_review
    def decide(self, data: DecisionInput) -> DecisionOutput:
        agent = next((a for a in data.candidate_agents if a in data.allowed_agents), None)
        base = {"risk_level": data.risk_level, "confidence": data.confidence,
                "decision_metadata": {"policy": "provisional", "confidence_source": data.confidence_source,
                    "confidence_derivation": data.confidence_source,
                    "confidence_is_measured": data.confidence is not None and data.confidence_source in {"measured", "calibrated", "human-verified"},
                    "timestamp": datetime.now(timezone.utc).isoformat(), "decision_id": str(uuid.uuid4()),
                    "execution_id": data.execution_id, "candidate_agents": data.candidate_agents,
                    "evidence_count": len(data.available_evidence),
                    "evidence_references": [e.get("chunk_id") or e.get("source_url") or e.get("document_id") for e in data.available_evidence],
                    "tool_error_count": len(data.tool_errors),
                    "remaining_steps": data.remaining_steps}}
        if data.ambiguous:
            return DecisionOutput(action="request_clarification", reason="The request lacks a material detail needed to select evidence safely.", evidence_required=False, stop=True, next_step="ask_user_to_clarify", **base)
        if data.remaining_steps <= 0 or data.agent_calls >= data.max_agent_calls:
            return DecisionOutput(action="return_partial_result" if data.available_evidence else "stop", reason="Execution budget is exhausted; further agent calls are prohibited.", stop=True, next_step="return_partial_result", **base)
        if data.candidate_agents and agent is None:
            return DecisionOutput(action="stop", reason="No candidate agent is permitted for this task.", stop=True, next_step="stop", **base)
        if data.risk_level == "high" and self.high_risk_review:
            return DecisionOutput(action="request_review", reason="High-risk requests require human review under the configured prototype policy.", stop=True, next_step="human_review", **base)
        if data.conflicting_evidence:
            return DecisionOutput(action="request_review", reason="Supplied evidence is marked contradictory and needs review.", stop=True, next_step="review_conflicting_evidence", **base)
        if data.task_status in {"partial", "success", "completed", "error"} and not evidence_is_sufficient(data.evidence_coverage, len(data.available_evidence), self.evidence_threshold):
            return DecisionOutput(action="return_partial_result" if data.available_evidence else "stop", reason="The configured workflow pass ended without sufficient evidence; return a bounded result.", stop=True, next_step="return_partial_result" if data.available_evidence else "stop", **base)
        if data.tool_errors and data.retrieval_attempts < data.max_retrieval_attempts and "retry" not in data.previous_actions:
            return DecisionOutput(action="retry", selected_agent=agent, reason="A tool failed and a bounded retry remains.", next_step="retry_failed_tool", **base)
        if data.task_status == "pending" and agent and data.agent_calls == 0:
            return DecisionOutput(action="route_to_agent", selected_agent=agent, reason="The highest-priority permitted candidate is selected for evidence gathering.", next_step=f"invoke_{agent}", **base)
        if not evidence_is_sufficient(data.evidence_coverage, len(data.available_evidence), self.evidence_threshold):
            if data.retrieval_attempts < data.max_retrieval_attempts:
                return DecisionOutput(action="retrieve_more_evidence", selected_agent="research_agent" if "research_agent" in data.allowed_agents else None, reason="Available evidence is absent or below the configured coverage threshold.", next_step="retrieve_evidence", **base)
            return DecisionOutput(action="return_partial_result" if data.available_evidence else "stop", reason="Evidence remains insufficient after the configured retrieval limit.", stop=True, next_step="return_partial_result" if data.available_evidence else "stop", **base)
        if data.task_status in {"success", "completed"}:
            return DecisionOutput(action="continue", reason="Evidence threshold is met; finalize the validated result.", stop=True, next_step="finalize_response", **base)
        if data.task_status == "partial":
            return DecisionOutput(action="return_partial_result", reason="Some evidence is available, but validation or source coverage remains incomplete.", stop=True, next_step="return_partial_result", **base)
        if agent:
            return DecisionOutput(action="route_to_agent", selected_agent=agent, reason="The highest-priority permitted candidate is selected for evidence gathering.", next_step=f"invoke_{agent}", **base)
        return DecisionOutput(action="stop", reason="No candidate agent is permitted for this task.", stop=True, next_step="stop", **base)
