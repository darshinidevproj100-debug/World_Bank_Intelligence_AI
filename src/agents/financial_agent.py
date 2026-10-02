"""Explicit Finances One specialist placeholder pending dataset selection."""
from src.agents.schemas import AgentResult


def run_financial_agent(question: str) -> AgentResult:
    """Return a limitation until a verified dataset and safe query schema exist."""
    del question  # The question is not sent to an unspecified query service.
    return AgentResult(
        agent="financial_agent", status="partial",
        summary="No Finances One records are configured for this workflow.",
        limitations=[
            "Select a public Finances One dataset and confirm its fields before enabling queries.",
            "Financial facts and derived calculations are not available in this run.",
        ],
    )
