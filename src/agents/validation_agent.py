"""Deterministic evidence and citation checks with explicit semantic limits."""
import re
from src.agents.schemas import AgentResult, EvidenceValidationResult


def validate_evidence(*, claims: list[str], citations: list[dict], retrieved_chunks: list[dict],
                      metadata_only: bool = False) -> EvidenceValidationResult:
    """Check that citation IDs resolve to retrieved sources and flag unlinked claims."""
    valid = {str(c.get("citation_id")): c for c in citations if c.get("citation_id") and (c.get("source_url") or c.get("document_id"))}
    chunks = {str(c.get("citation_id")): c for c in retrieved_chunks if c.get("citation_id")}
    errors = [f"Citation {key} has no retrieved chunk." for key in valid if key not in chunks]
    if not valid and claims:
        errors.append("Claims were returned without any source citation.")
    gaps = [claim for claim in claims if not valid]
    unsupported = []
    numeric_pattern = re.compile(r"(?<!\w)\d+(?:[,.]\d+)*(?:%|\b)")
    cited_text = " ".join(str(row.get("text", "")) for row in retrieved_chunks
                           if str(row.get("citation_id")) in valid)
    for claim in claims:
        claim_numbers = {value.replace(",", "") for value in numeric_pattern.findall(claim)}
        if claim_numbers and not claim_numbers.issubset({value.replace(",", "") for value in numeric_pattern.findall(cited_text)}):
            unsupported.append(claim)
    values_by_key: dict[tuple, set[str]] = {}
    for row in retrieved_chunks:
        metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
        key = tuple(row.get(k, metadata.get(k)) for k in ("country_code", "indicator_code", "year"))
        value = row.get("value", metadata.get("value"))
        if all(part is not None for part in key) and value is not None:
            values_by_key.setdefault(key, set()).add(str(value))
    contradictions = [f"Conflicting values were supplied for {key}: {sorted(values)}"
                      for key, values in values_by_key.items() if len(values) > 1]
    notes = ["Checks citation presence and source consistency only; citation presence does not prove semantic entailment."]
    if metadata_only:
        notes.append("Metadata search results are not full-text evidence and cannot support report-content claims.")
        gaps.extend(claims)
    unsupported.extend(gaps)
    is_valid = bool(valid) and not errors and not gaps and not contradictions and not unsupported
    status = "invalid_citations" if errors else "insufficient_evidence" if gaps or unsupported else "not_semantically_verified"
    return EvidenceValidationResult(is_valid=is_valid, groundedness_status=status,
        supported_claims=[], unsupported_claims=list(dict.fromkeys(unsupported)),
        citation_errors=errors, contradictions=contradictions,
        evidence_gaps=([] if valid and not gaps else ["No citable or complete evidence for one or more claims."]),
        recommended_action="request_review" if contradictions else "continue_with_limitations" if is_valid else "retrieve_more_evidence",
        validation_notes=notes)


def validate_results(results: list[AgentResult]) -> AgentResult:
    """Check source presence and status without claiming claim-level verification."""
    successful = [r for r in results if r.status == "success"]
    sources = sorted({source for r in results for source in r.sources})
    limitations = [item for r in results for item in r.limitations]
    errors = [item for r in results for item in r.errors]
    citation_gaps = [f"{r.agent} returned success without source reference." for r in successful if not r.sources]
    claims = [r.summary for r in successful if r.agent != "documents_metadata_search"]
    citations: list[dict] = []
    retrieved_chunks: list[dict] = []
    for result in successful:
        if result.agent == "research_agent" and isinstance(result.data, dict):
            citations.extend(result.data.get("citations", []))
            retrieved_chunks.extend(result.data.get("retrieved_chunks", []))
        elif result.agent == "data_agent" and isinstance(result.data, list):
            for index, row in enumerate(result.data):
                citation_id = f"D{index + 1}"
                source = row.get("source_url")
                if source:
                    citations.append({"citation_id": citation_id, "source_url": source})
                    retrieved_chunks.append({"citation_id": citation_id, "text": str(row),
                                             **{key: row.get(key) for key in ("country_code", "indicator_code", "year", "value")}})
    finding = validate_evidence(claims=claims, citations=citations, retrieved_chunks=retrieved_chunks)
    if any(r.agent == "documents_metadata_search" for r in successful):
        finding.validation_notes.append("Metadata search records were kept separate from full-text RAG evidence.")
    if not successful:
        return AgentResult(agent="validation_agent", status="partial" if results else "error",
            summary="No agent returned a successful, source-backed result.",
            data={"findings": [], "citation_gaps": citation_gaps, "validation": finding.model_dump()},
            sources=sources, limitations=limitations, errors=errors)
    status = "partial" if limitations or errors or citation_gaps else "success"
    return AgentResult(agent="validation_agent", status=status,
        summary="Source references were checked; semantic claim verification is unavailable.",
        data={"findings": [r.model_dump() for r in results], "citation_gaps": citation_gaps,
              "validation": finding.model_dump()}, sources=sources,
        limitations=limitations + ["Citation presence alone does not prove claim entailment."], errors=errors)
