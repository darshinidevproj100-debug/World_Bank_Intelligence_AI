"""Unit tests for deterministic prototype evaluation metrics."""
import pytest

from evaluation.evaluation_metrics import (
    agent_execution_success_rate,
    citation_coverage,
    data_validation_pass_rate,
    factual_consistency_rate,
    faithfulness_rate,
    ingestion_success_rate,
    mean_latency_ms,
    precision_at_k,
    recall_at_k,
    routing_accuracy,
    success_rate,
    hit_rate_at_k,
    mean_reciprocal_rank,
    average_precision,
    ndcg_at_k,
    evidence_coverage,
    unsupported_claim_rate,
    clarification_precision_recall,
)


def test_retrieval_and_routing_metrics_are_deterministic():
    assert precision_at_k({"a", "c"}, ["a", "b", "c"], 2) == 0.5
    assert recall_at_k({"a", "c"}, ["a", "b", "c"], 2) == 0.5
    assert routing_accuracy({"data_agent", "research_agent"}, {"data_agent"}) == 0.0
    assert routing_accuracy({"data_agent"}, {"data_agent"}) == 1.0


def test_workflow_metrics_have_explicit_denominators():
    assert citation_coverage(2, 4) == 0.5
    assert citation_coverage(0, 0) == 1.0
    assert success_rate(3, 4) == 0.75
    assert ingestion_success_rate(3, 4) == 0.75
    assert data_validation_pass_rate(8, 10) == 0.8
    assert agent_execution_success_rate(4, 5) == 0.8
    assert faithfulness_rate(7, 10) == 0.7
    assert factual_consistency_rate(8, 10) == 0.8
    assert mean_latency_ms([10, 30]) == 20


@pytest.mark.parametrize(
    "call",
    [
        lambda: precision_at_k(set(), [], 0),
        lambda: recall_at_k(set(), [], 0),
        lambda: citation_coverage(3, 2),
        lambda: success_rate(2, 1),
        lambda: mean_latency_ms([-1]),
        lambda: hit_rate_at_k(set(), [], 0),
        lambda: ndcg_at_k([-1], 2),
    ],
)
def test_metrics_reject_invalid_inputs(call):
    with pytest.raises(ValueError):
        call()


def test_ranked_and_grounding_metrics_handle_empty_and_missing_labels():
    assert hit_rate_at_k({"a"}, ["b", "a"], 2) == 1.0
    assert mean_reciprocal_rank({"a"}, ["b", "a"]) == 0.5
    assert average_precision({"a", "c"}, ["a", "b", "c"]) == pytest.approx(5 / 6)
    assert ndcg_at_k([3, 0, 1], 3) < 1
    assert ndcg_at_k([], 3) == 0
    assert evidence_coverage(0, 0) == 0
    assert unsupported_claim_rate(0, 0) == 0
    assert clarification_precision_recall([True, False], [True, True]) == (0.5, 1.0)
