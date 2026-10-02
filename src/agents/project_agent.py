"""Explicit Projects & Operations specialist placeholder pending schema selection."""
from src.agents.schemas import AgentResult


def run_project_agent(question: str) -> AgentResult:
    """Return a limitation until verified project fields and filters are configured."""
    del question  # Do not pass natural-language input to an unspecified query tool.
    return AgentResult(
        agent="project_agent", status="partial",
        summary="No Projects & Operations records are configured for this workflow.",
        limitations=[
            "Select a Projects & Operations API/export and verify its fields before enabling queries.",
            "No project attributes or status details have been inferred.",
        ],
    )
