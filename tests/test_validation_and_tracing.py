"""Evidence reference checks, clarification paths, and local trace behavior."""
import json
from src.agents.validation_agent import validate_evidence
from src.utils.workflow_logging import append_jsonl
from src.workflows.agent_graph import run_agent_workflow

def test_validation_requires_real_retrieved_citations_and_discloses_limits():
    result = validate_evidence(claims=["GDP increased"], citations=[{"citation_id": "E1", "source_url": "https://x"}],
        retrieved_chunks=[{"citation_id": "E1", "text": "GDP evidence"}])
    assert result.is_valid
    assert result.groundedness_status == "not_semantically_verified"
    assert "does not prove semantic entailment" in result.validation_notes[0]
    bad = validate_evidence(claims=["report findings"], citations=[], retrieved_chunks=[])
    assert not bad.is_valid and bad.evidence_gaps
    numerical = validate_evidence(claims=["GDP was 42 in 2020"],
        citations=[{"citation_id": "E1", "source_url": "https://x"}],
        retrieved_chunks=[{"citation_id": "E1", "text": "GDP value was 41 in 2020"}])
    assert not numerical.is_valid and numerical.unsupported_claims
    conflict = validate_evidence(claims=[], citations=[], retrieved_chunks=[
        {"country_code": "GBR", "indicator_code": "GDP", "year": 2020, "value": 1},
        {"country_code": "GBR", "indicator_code": "GDP", "year": 2020, "value": 2},
    ])
    assert conflict.contradictions

def test_jsonl_logger_and_ambiguous_workflow_are_reviewable(tmp_path):
    log_path = tmp_path / "traces.jsonl"
    assert append_jsonl(log_path, {"execution_id": "e1", "status": "ok"}) is None
    assert json.loads(log_path.read_text(encoding="utf-8")) == {"execution_id": "e1", "status": "ok"}
    response = run_agent_workflow("Show me the figures", trace_path=tmp_path / "workflow.jsonl")
    assert response.selected_agents == ["clarification_agent"]
    assert response.decision["action"] == "request_clarification"
    assert response.execution_id and response.trace and response.latency_ms >= 0
