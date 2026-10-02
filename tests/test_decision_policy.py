"""Decision contract and bounded prototype policy tests."""
import pytest
from pydantic import ValidationError
from src.decision import DecisionInput, PrototypeDecisionPolicy

def inp(**kwargs):
    return DecisionInput(execution_id="e1", question="GDP?", candidate_agents=["data_agent"], **kwargs)

def test_policy_routes_only_permitted_agents_and_labels_confidence():
    policy = PrototypeDecisionPolicy()
    result = policy.decide(inp(task_status="pending", allowed_agents=["research_agent"]))
    assert result.action == "stop"
    assert result.stop
    result = policy.decide(inp(task_status="pending", agent_calls=0, remaining_steps=3))
    assert result.action == "route_to_agent"
    assert result.decision_metadata["confidence_is_measured"] is False

def test_policy_clarifies_reviews_retries_and_stops_at_limits():
    policy = PrototypeDecisionPolicy()
    assert policy.decide(inp(ambiguous=True)).action == "request_clarification"
    assert policy.decide(inp(risk_level="high")).action == "request_review"
    assert policy.decide(inp(tool_errors=["timeout"], retrieval_attempts=0)).action == "retry"
    assert policy.decide(inp(remaining_steps=0)).stop
    assert policy.decide(inp(conflicting_evidence=True)).action == "request_review"

def test_decision_schema_rejects_invalid_values_and_policy_config():
    with pytest.raises(ValidationError): inp(risk_level="urgent")
    with pytest.raises(ValueError): PrototypeDecisionPolicy(evidence_threshold=2)
