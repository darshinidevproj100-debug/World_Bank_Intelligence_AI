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


def hit_rate_at_k(relevant_ids: set[str], retrieved_ids: Sequence[str], k: int) -> float:
    """Return 1 when any labelled relevant ID appears in the top k, else 0."""
    _validate_k(k)
    return float(bool(relevant_ids.intersection(retrieved_ids[:k])))


def mean_reciprocal_rank(relevant_ids: set[str], retrieved_ids: Sequence[str]) -> float:
    """Reciprocal rank of the first relevant result; zero when no hit/labels."""
    for rank, identifier in enumerate(retrieved_ids, start=1):
        if identifier in relevant_ids:
            return 1.0 / rank
    return 0.0


def average_precision(relevant_ids: set[str], retrieved_ids: Sequence[str], k: int | None = None) -> float:
    """Average precision over ranked results, normalized by known relevant IDs."""
    if k is not None: _validate_k(k)
    if not relevant_ids: return 0.0
    results = retrieved_ids[:k] if k else retrieved_ids
    hits = 0; total = 0.0
    for rank, identifier in enumerate(results, start=1):
        if identifier in relevant_ids:
            hits += 1; total += hits / rank
    return total / len(relevant_ids)


def ndcg_at_k(relevance: Sequence[float], k: int) -> float:
    """Normalized discounted cumulative gain for ordered non-negative grades."""
    _validate_k(k)
    if any(value < 0 for value in relevance): raise ValueError("relevance grades must be non-negative")
    import math
    def dcg(values): return sum((2 ** value - 1) / math.log2(i + 2) for i, value in enumerate(values[:k]))
    actual = dcg(relevance)
    ideal = dcg(sorted(relevance, reverse=True))
    return actual / ideal if ideal else 0.0


def _validate_k(k: int) -> None:
    if k <= 0: raise ValueError("k must be positive")


def evidence_coverage(supported_evidence: int, required_evidence: int) -> float:
    """Coverage over explicitly labelled expected evidence references."""
    if required_evidence < 0 or supported_evidence < 0 or supported_evidence > required_evidence:
        raise ValueError("evidence counts must satisfy 0 <= supported <= required")
    return supported_evidence / required_evidence if required_evidence else 0.0


def unsupported_claim_rate(unsupported_claims: int, assessed_claims: int) -> float:
    """Share of reviewed claims labelled unsupported by a human/evaluator."""
    if assessed_claims < 0 or unsupported_claims < 0 or unsupported_claims > assessed_claims:
        raise ValueError("claim counts must satisfy 0 <= unsupported <= assessed")
    return unsupported_claims / assessed_claims if assessed_claims else 0.0


def clarification_precision_recall(expected: Sequence[bool], actual: Sequence[bool]) -> tuple[float, float]:
    """Return precision and recall for clarification labels; empty denominators yield zero."""
    if len(expected) != len(actual): raise ValueError("expected and actual labels must align")
    tp = sum(e and a for e, a in zip(expected, actual))
    fp = sum((not e) and a for e, a in zip(expected, actual))
    fn = sum(e and (not a) for e, a in zip(expected, actual))
    return (tp / (tp + fp) if tp + fp else 0.0, tp / (tp + fn) if tp + fn else 0.0)


def decision_agreement(expected: Sequence[str], actual: Sequence[str]) -> float:
    """Exact agreement on labelled actions; this evaluates policy agreement only."""
    if len(expected) != len(actual): raise ValueError("expected and actual actions must align")
    return sum(e == a for e, a in zip(expected, actual)) / len(expected) if expected else 0.0


def retry_success_rate(retries_succeeded: int, retries_attempted: int) -> float:
    """Share of attempted retries that recovered successfully."""
    return success_rate(retries_succeeded, retries_attempted)


def policy_violation_count(decisions: Sequence[dict]) -> int:
    """Count decisions that select a disallowed agent or exceed a step budget."""
    return sum(1 for d in decisions if (d.get("selected_agent") and d.get("selected_agent") not in d.get("allowed_agents", []))
               or d.get("remaining_steps", 1) < 0)


def average_agent_calls(call_counts: Sequence[int]) -> float:
    """Mean measured agent calls per task, or zero for no observations."""
    if any(count < 0 for count in call_counts): raise ValueError("agent call counts must be non-negative")
    return sum(call_counts) / len(call_counts) if call_counts else 0.0
