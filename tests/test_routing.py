"""Tests for predictable baseline supervisor routing."""
from src.agents.supervisor import route_question


def test_router_selects_financial_and_project_agents():
    agents = route_question("Show project funding and loan details")
    assert "financial_agent" in agents
    assert "project_agent" in agents


def test_router_defaults_to_data_agent():
    assert route_question("Show GDP trends") == ["data_agent"]
