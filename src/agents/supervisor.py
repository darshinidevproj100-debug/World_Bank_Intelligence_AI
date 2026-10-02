"""Transparent keyword routing baseline pending authoritative JEV rules."""


def route_question(question: str) -> list[str]:
    """Return the agents selected by explicit keyword rules, in stable order."""
    normalized_question = question.casefold()
    selected_agents = ["data_agent"]

    if any(term in normalized_question for term in ("finance", "funding", "loan", "disbursement", "commitment")):
        selected_agents.append("financial_agent")
    if any(term in normalized_question for term in ("project", "operation", "portfolio")):
        selected_agents.append("project_agent")
    if any(term in normalized_question for term in ("report", "document", "explain", "evidence", "why")):
        selected_agents.append("research_agent")

    return list(dict.fromkeys(selected_agents))
