"""Bounded local agent orchestration with an optional LangGraph wrapper."""
from __future__ import annotations

from pathlib import Path
import logging
from typing import Any, TypedDict

import pandas as pd
from pydantic import BaseModel, Field

from src.agents.data_agent import run_data_agent
from src.agents.financial_agent import run_financial_agent
from src.agents.project_agent import run_project_agent
from src.agents.research_agent import run_research_agent
from src.agents.schemas import AgentResult
from src.agents.supervisor import route_question
from src.agents.validation_agent import validate_results
from src.data.documents_reports import fetch_document_metadata
from src.data.query_tools import WDIQueryTool
from src.retrieval.document_loader import load_document_chunks, retrieve_local_documents

LOGGER = logging.getLogger(__name__)


class WorkflowResponse(BaseModel):
    """Stable response contract returned for each user question."""

    question: str
    selected_agents: list[str]
    status: str
    answer: str
    sources: list[str] = Field(default_factory=list)
    agent_results: list[AgentResult] = Field(default_factory=list)
    validation: AgentResult
    limitations: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class WorkflowState(TypedDict, total=False):
    """Shared workflow state passed from routing through answer validation."""

    question: str
    selected_agents: list[str]
    agent_results: list[AgentResult]
    validation: AgentResult
    final_answer: str
    sources: list[str]


def route_node(state: WorkflowState) -> dict[str, Any]:
    """Classify the query using the transparent keyword routing baseline."""
    return {"selected_agents": route_question(state["question"])}


def _run_selected_agents(
    question: str,
    selected_agents: list[str],
    *,
    query_tool: WDIQueryTool | None,
    document_chunks: list[dict[str, Any]],
    remote_document_metadata: list[dict[str, Any]],
) -> list[AgentResult]:
    """Run each selected specialist once; agent calls have no retry loop."""
    run_functions = {
        "data_agent": lambda: run_data_agent(question, query_tool=query_tool),
        "financial_agent": lambda: run_financial_agent(question),
        "project_agent": lambda: run_project_agent(question),
        "research_agent": lambda: run_research_agent(
            question,
            retriever=lambda query: (
                retrieve_local_documents(query, document_chunks)
                + [
                    {
                        "document_id": document["document_key"],
                        "title": document["title"],
                        "chunk_index": 0,
                        "page_number": None,
                        "text": document.get("abstract") or "No abstract was returned by the API.",
                        "source_url": document["source_url"],
                        "pdf_url": document.get("pdf_url"),
                        "evidence_type": "World Bank D&R metadata/abstract",
                        "publication_date": document.get("publication_date"),
                    }
                    for document in remote_document_metadata
                ]
            ),
        ),
    }
    agent_results: list[AgentResult] = []
    for agent_name in selected_agents:
        try:
            agent_results.append(run_functions[agent_name]())
        except Exception as agent_error:
            LOGGER.warning("Agent %s failed (%s)", agent_name, type(agent_error).__name__)
            agent_results.append(
                AgentResult(
                    agent=agent_name, status="error",
                    summary="The selected agent failed; no answer was produced by that agent.",
                    errors=[f"{type(agent_error).__name__}: {agent_error}"],
                )
            )
    return agent_results


def _synthesize_answer(question: str, agent_results: list[AgentResult]) -> str:
    """Assemble only explicit agent findings and literal retrieved evidence."""
    answer_sections: list[str] = []
    for agent_result in agent_results:
        if agent_result.agent == "data_agent" and agent_result.status == "success":
            answer_sections.append(agent_result.answer or agent_result.summary)
        elif agent_result.agent == "research_agent" and agent_result.status == "success":
            for passage in agent_result.data:
                passage_excerpt = str(passage.get("text", ""))[:600]
                evidence_label = passage.get("evidence_type", "local document passage")
                answer_sections.append(
                    f"Document evidence ({evidence_label}) — {passage.get('title', 'Untitled')}: "
                    f"{passage_excerpt} [source: {passage.get('source_url', 'missing source')}]"
                )
    if answer_sections:
        return "\n\n".join(answer_sections)
    return (
        f"The configured sources do not provide enough verified evidence to answer: {question} "
        "Add the relevant WDI observations or local source documents and try again."
    )


def run_agent_workflow(
    question: str,
    *,
    wdi_csv_path: str | Path | None = None,
    documents_folder: str | Path | None = None,
    world_bank_document_query: str | None = None,
) -> WorkflowResponse:
    """Run routing, selected agents, validation, and evidence-only synthesis.

    Local mode works without Databricks or an LLM. The keyword router chooses
    specialists; WDI queries are parameterized and documents use deterministic
    local lexical retrieval. Each agent runs once, so the workflow is bounded.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    dependency_errors: list[AgentResult] = []
    wdi_query_tool: WDIQueryTool | None = None
    document_chunks: list[dict[str, Any]] = []
    remote_document_metadata: list[dict[str, Any]] = []
    if wdi_csv_path:
        try:
            wdi_query_tool = WDIQueryTool(pd.read_csv(wdi_csv_path))
        except Exception as source_error:
            dependency_errors.append(AgentResult(
                agent="wdi_loader", status="error",
                summary="The local WDI CSV could not be loaded or validated.",
                errors=[f"{type(source_error).__name__}: {source_error}"],
            ))
    if documents_folder:
        try:
            document_chunks = load_document_chunks(documents_folder)
        except Exception as source_error:
            dependency_errors.append(AgentResult(
                agent="document_loader", status="error",
                summary="Local source documents could not be loaded.",
                errors=[f"{type(source_error).__name__}: {source_error}"],
            ))
    if world_bank_document_query:
        try:
            remote_document_metadata = fetch_document_metadata(world_bank_document_query)
        except Exception as source_error:
            dependency_errors.append(AgentResult(
                agent="documents_reports_api", status="error",
                summary="World Bank Documents & Reports metadata search failed.",
                errors=[f"{type(source_error).__name__}: {source_error}"],
            ))
    selected_agents = route_question(question.strip())
    agent_results = dependency_errors + _run_selected_agents(
        question.strip(), selected_agents,
        query_tool=wdi_query_tool,
        document_chunks=document_chunks,
        remote_document_metadata=remote_document_metadata,
    )
    validation_result = validate_results(agent_results)
    final_answer = _synthesize_answer(question.strip(), agent_results)
    all_sources = sorted({source for result in agent_results for source in result.sources})
    workflow_errors = [error for result in agent_results for error in result.errors]
    workflow_limitations = list(dict.fromkeys(
        limitation for result in agent_results + [validation_result]
        for limitation in result.limitations
    ))
    has_successful_evidence = any(result.status == "success" and result.sources for result in agent_results)
    response_status = "success" if has_successful_evidence and validation_result.status == "success" else "partial"

    return WorkflowResponse(
        question=question.strip(), selected_agents=selected_agents, status=response_status,
        answer=final_answer, sources=all_sources, agent_results=agent_results,
        validation=validation_result, limitations=workflow_limitations, errors=workflow_errors,
    )


def build_graph(
    *, wdi_csv_path: str | Path | None = None,
    documents_folder: str | Path | None = None,
    world_bank_document_query: str | None = None,
):
    """Build an optional LangGraph wrapper with configured local evidence tools."""
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as import_error:
        raise RuntimeError("Install the optional agents dependencies first.") from import_error

    wdi_query_tool = WDIQueryTool(pd.read_csv(wdi_csv_path)) if wdi_csv_path else None
    local_document_chunks = load_document_chunks(documents_folder) if documents_folder else []
    remote_document_metadata = (
        fetch_document_metadata(world_bank_document_query) if world_bank_document_query else []
    )

    def execute_agents_node(state: WorkflowState) -> dict[str, Any]:
        """Invoke selected agents once with the configured local tools."""
        return {
            "agent_results": _run_selected_agents(
                state["question"], state["selected_agents"],
                query_tool=wdi_query_tool, document_chunks=local_document_chunks,
                remote_document_metadata=remote_document_metadata,
            )
        }

    graph = StateGraph(WorkflowState)
    graph.add_node("route", route_node)
    graph.add_node("execute", execute_agents_node)
    graph.add_node("validate", lambda state: {
        "validation": validate_results(state.get("agent_results", []))
    })
    graph.add_node("synthesize", lambda state: {
        "final_answer": _synthesize_answer(state["question"], state.get("agent_results", [])),
        "sources": sorted({
            source for result in state.get("agent_results", []) for source in result.sources
        }),
    })
    graph.set_entry_point("route")
    graph.add_edge("route", "execute")
    graph.add_edge("execute", "validate")
    graph.add_edge("validate", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()
