"""Small, transparent metrics with explicit denominators and limitations."""
from collections.abc import Sequence


def precision_at_k(relevant_ids: set[str], retrieved_ids: Sequence[str], k: int) -> float:
    """Return the fraction of the top-k retrieved identifiers that are relevant."""
    if k <= 0:
        raise ValueError("k must be positive")
    top_retrieved_ids = list(retrieved_ids[:k])
    if not top_retrieved_ids:
        return 0.0
    relevant_retrieved_count = sum(identifier in relevant_ids for identifier in top_retrieved_ids)
    return relevant_retrieved_count / len(top_retrieved_ids)


def routing_accuracy(expected: set[str], actual: set[str]) -> float:
    """Return exact-match accuracy for one expected/actual routing decision."""
    return 1.0 if expected == actual else 0.0


def recall_at_k(relevant_ids: set[str], retrieved_ids: Sequence[str], k: int) -> float:
    """Return relevant retrieved IDs / all known relevant IDs (zero if none)."""
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant_ids:
        return 0.0
    return len(relevant_ids & set(retrieved_ids[:k])) / len(relevant_ids)


def citation_coverage(cited_claims: int, factual_claims: int) -> float:
    """Return the share of factual claims carrying citations; semantic support is unscored."""
    if factual_claims < 0 or cited_claims < 0 or cited_claims > factual_claims:
        raise ValueError("claim counts must satisfy 0 <= cited_claims <= factual_claims")
    return cited_claims / factual_claims if factual_claims else 1.0


def success_rate(successful_runs: int, total_runs: int) -> float:
    """Return successful workflow runs / attempted runs."""
    if total_runs < 0 or successful_runs < 0 or successful_runs > total_runs:
        raise ValueError("run counts must satisfy 0 <= successful_runs <= total_runs")
    return successful_runs / total_runs if total_runs else 0.0


def ingestion_success_rate(successful_ingestions: int, attempted_ingestions: int) -> float:
    """Return completed source ingestions / attempted source ingestions."""
    return success_rate(successful_ingestions, attempted_ingestions)


def data_validation_pass_rate(passing_records: int, validated_records: int) -> float:
    """Return records passing validation / records checked."""
    return success_rate(passing_records, validated_records)


def agent_execution_success_rate(successful_agents: int, attempted_agents: int) -> float:
    """Return successful agent calls / attempted agent calls."""
    return success_rate(successful_agents, attempted_agents)


def faithfulness_rate(supported_claims: int, assessed_claims: int) -> float:
    """Return human-labelled claims supported by evidence / claims assessed."""
    return success_rate(supported_claims, assessed_claims)


def factual_consistency_rate(consistent_claims: int, assessed_claims: int) -> float:
    """Return human-labelled consistent source/claim pairs / pairs assessed."""
    return success_rate(consistent_claims, assessed_claims)


def mean_latency_ms(latencies_ms: Sequence[float]) -> float:
    """Return mean measured latency; callers must use monotonic elapsed times."""
    if any(latency < 0 for latency in latencies_ms):
        raise ValueError("latencies must be non-negative")
    return sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
