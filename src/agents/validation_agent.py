"""Evidence presence validation with explicit limits on automated checking."""
from src.agents.schemas import AgentResult


def validate_results(results: list[AgentResult]) -> AgentResult:
    """Check outcome status and source presence without pretending to prove claims."""
    successful = [r for r in results if r.status == "success"]
    sources = sorted({source for r in results for source in r.sources})
    limitations = [item for r in results for item in r.limitations]
    errors = [item for r in results for item in r.errors]
    citation_gaps = [
        f"{result.agent} returned a successful result without a source reference."
        for result in successful if not result.sources
    ]

    if not successful:
        return AgentResult(
            agent="validation_agent", status="partial" if results else "error",
            summary="No agent returned a successful, source-backed result.",
            data={"findings": [], "citation_gaps": citation_gaps},
            sources=sources, limitations=limitations, errors=errors,
        )

    validation_status = "partial" if limitations or errors or citation_gaps else "success"
    return AgentResult(
        agent="validation_agent",
        status=validation_status,
        summary="Evidence references were checked; automated claim-level factual verification is not implemented.",
        data={
            "findings": [result.model_dump() for result in results],
            "citation_gaps": citation_gaps,
        },
        sources=sources,
        limitations=limitations + ["This workflow does not prove that every generated claim is entailed by its sources."],
        errors=errors,
    )
