"""Bounded local agent orchestration with an optional LangGraph wrapper."""
from __future__ import annotations

from pathlib import Path
import logging
import os
import time
import uuid
from datetime import datetime, timezone
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
from src.decision import DecisionInput, PrototypeDecisionPolicy
from src.data.documents_reports import fetch_document_metadata
from src.data.query_tools import WDIQueryTool
from src.retrieval.document_loader import load_document_chunks
from src.retrieval.retrievers import infer_metadata_filters, make_retriever
from src.utils.config import get_runtime_config
from src.utils.workflow_logging import append_jsonl

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
    execution_id: str = ""
    decision: dict[str, Any] = Field(default_factory=dict)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    latency_ms: float = 0.0
    intent: str = "unknown"
    current_agent: str | None = None
    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    confidence: float | None = None
    retrieval_attempts: int = 0
    risk_level: str = "low"
    iteration_count: int = 0
    remaining_steps: int = 0
    validation_status: str = "unknown"
    citations: list[dict[str, Any]] = Field(default_factory=list)
    started_at: str = ""
    completed_at: str = ""


class WorkflowState(TypedDict, total=False):
    """Shared workflow state passed from routing through answer validation."""

    question: str
    intent: str
    selected_agents: list[str]
    current_agent: str | None
    agent_outputs: list[AgentResult]
    agent_results: list[AgentResult]
    validation: AgentResult
    final_answer: str
    started_at: str
    completed_at: str
    sources: list[str]
    execution_id: str
    retrieved_chunks: list[dict[str, Any]]
    evidence_gaps: list[str]
    confidence: float | None
    retrieval_attempts: int
    risk_level: str
    decision: dict[str, Any]
    iteration_count: int
    remaining_steps: int
    tool_errors: list[str]
    validation_status: str
    answer: str
    citations: list[dict[str, Any]]
    status: str
    started_at: str
    completed_at: str
    latency_ms: float


def route_node(state: WorkflowState) -> dict[str, Any]:
    """Classify the query using the transparent keyword routing baseline."""
    question = state["question"].strip()
    task_terms = ("gdp", "population", "unemployment", "economic", "indicator", "finance", "loan",
        "funding", "project", "operation", "portfolio", "report", "document", "research", "evidence", "compare")
    agents = route_question(question)
    if not any(term in question.casefold() for term in task_terms):
        agents = ["clarification_agent"]
    return {"selected_agents": agents}


def _run_selected_agents(
    question: str,
    selected_agents: list[str],
    *,
    query_tool: WDIQueryTool | None,
    document_chunks: list[dict[str, Any]],
    remote_document_metadata: list[dict[str, Any]],
    trace: list[dict[str, Any]] | None = None,
) -> list[AgentResult]:
    """Run each selected specialist once; agent calls have no retry loop."""
    runtime = get_runtime_config()
    research_index = None
    research_setup_error = None
    if "research_agent" in selected_agents:
        try:
            research_index = make_retriever(
                document_chunks, method=str(runtime["retrieval_method"]),
                model_id=os.getenv("EMBEDDING_MODEL") or None,
                lexical_weight=float(runtime["hybrid_lexical_weight"]),
                semantic_weight=float(runtime["hybrid_semantic_weight"]),
            )
        except Exception as setup_error:
            research_setup_error = setup_error
    def retrieve_research(query: str):
        if research_setup_error:
            raise RuntimeError(str(research_setup_error)) from research_setup_error
        filters = infer_metadata_filters(query, document_chunks)
        return research_index.search(query, top_k=int(runtime["retrieval_top_k"]), filters=filters) if research_index else []
    run_functions = {
        "data_agent": lambda: run_data_agent(question, query_tool=query_tool),
        "financial_agent": lambda: run_financial_agent(question),
        "project_agent": lambda: run_project_agent(question),
        "research_agent": lambda: run_research_agent(
            question,
            retriever=retrieve_research,
        ),
        "clarification_agent": lambda: AgentResult(
            agent="clarification_agent", status="partial",
            summary="Please clarify the country, topic, or evidence source you want to examine.",
            limitations=["The question does not identify a supported data or research task."],
        ),
        "review_agent": lambda: AgentResult(
            agent="review_agent", status="partial",
            summary="This request was held for human review by the configured decision policy.",
            limitations=["No high-impact recommendation was produced automatically."],
        ),
    }
    agent_results: list[AgentResult] = []
    for agent_name in selected_agents:
        agent_started = time.perf_counter()
        agent_started_at = datetime.now(timezone.utc).isoformat()
        if trace is not None:
            trace.append({"stage": "agent_start", "agent": agent_name, "timestamp": agent_started_at})
        try:
            agent_result = run_functions[agent_name]()
            agent_results.append(agent_result)
        except Exception as agent_error:
            LOGGER.warning("Agent %s failed (%s)", agent_name, type(agent_error).__name__)
            agent_results.append(
                AgentResult(
                    agent=agent_name, status="error",
                    summary="The selected agent failed; no answer was produced by that agent.",
                    errors=[f"{type(agent_error).__name__}: {agent_error}"],
                )
            )
            agent_result = agent_results[-1]
        if trace is not None:
            trace.append({"stage": "agent_end", "agent": agent_name, "status": agent_result.status,
                          "duration_ms": round((time.perf_counter() - agent_started) * 1000, 3),
                          "timestamp": datetime.now(timezone.utc).isoformat()})
    if remote_document_metadata:
        agent_results.append(AgentResult(
            agent="documents_metadata_search", status="success",
            summary=f"World Bank Documents & Reports metadata search returned {len(remote_document_metadata)} record(s).",
            data=remote_document_metadata,
            sources=sorted({str(row.get("source_url")) for row in remote_document_metadata if row.get("source_url")}),
            limitations=["These are Documents & Reports metadata/abstract results, not retrieved or analyzed full report text."],
        ))
    return agent_results


def _synthesize_answer(question: str, agent_results: list[AgentResult]) -> str:
    """Assemble only explicit agent findings and literal retrieved evidence."""
    answer_sections: list[str] = []
    for agent_result in agent_results:
        if agent_result.agent == "clarification_agent":
            answer_sections.append(agent_result.answer or agent_result.summary)
        elif agent_result.agent == "review_agent":
            answer_sections.append(agent_result.answer or agent_result.summary)
        elif agent_result.agent in {"financial_agent", "project_agent"}:
            answer_sections.append(agent_result.answer or agent_result.summary)
        elif agent_result.agent == "data_agent" and agent_result.status == "success":
            answer_sections.append(agent_result.answer or agent_result.summary)
        elif agent_result.agent == "research_agent" and agent_result.status == "success":
            data = agent_result.data or {}
            passages = data.get("retrieved_chunks", []) if isinstance(data, dict) else data
            for passage in passages:
                passage_excerpt = str(passage.get("text", ""))[:600]
                evidence_label = passage.get("evidence_type", "local document passage")
                answer_sections.append(
                    f"Document evidence ({evidence_label}) — {passage.get('title', 'Untitled')}: "
                    f"{passage_excerpt} [source: {passage.get('source_url', 'missing source')}]"
                )
        elif agent_result.agent == "documents_metadata_search" and agent_result.status == "success":
            for record in agent_result.data or []:
                abstract = record.get("abstract")
                detail = f" Abstract: {abstract}" if abstract else " No abstract was provided by the search API."
                answer_sections.append(
                    f"World Bank D&R metadata/abstract — {record.get('title') or 'Untitled'}"
                    f"{detail} [metadata source: {record.get('source_url')}]"
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
    trace_path: str | Path | None = None,
    preloaded_document_chunks: list[dict[str, Any]] | None = None,
) -> WorkflowResponse:
    """Run routing, selected agents, validation, and evidence-only synthesis.

    Local mode works without Databricks or an LLM. The keyword router chooses
    specialists; WDI queries are parameterized and documents use deterministic
    local lexical retrieval. Each agent runs once, so the workflow is bounded.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    started = time.perf_counter()
    execution_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc).isoformat()
    trace = [{"stage": "input_validation", "status": "passed", "timestamp": started_at}]
    dependency_errors: list[AgentResult] = []
    wdi_query_tool: WDIQueryTool | None = None
    document_chunks: list[dict[str, Any]] = list(preloaded_document_chunks or [])
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
    if documents_folder and not document_chunks:
        try:
            runtime = get_runtime_config()
            document_chunks = load_document_chunks(documents_folder,
                chunk_size=int(runtime["chunk_size"]), overlap=int(runtime["chunk_overlap"]))
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
    broad_task_terms = ("gdp", "population", "unemployment", "economic", "indicator", "finance", "loan",
        "funding", "project", "operation", "portfolio", "report", "document", "research", "evidence", "compare")
    ambiguous = not any(term in question.casefold() for term in broad_task_terms)
    if ambiguous:
        selected_agents = ["clarification_agent"]
    runtime = get_runtime_config()
    max_agent_steps = int(runtime["max_agent_steps"])
    elevated_risk = any(term in question.casefold() for term in ("recommend", "should we", "approve", "eligibility", "sanction"))
    selected_agents = selected_agents[:max_agent_steps] if max_agent_steps else []
    trace.append({"stage": "intent_analysis", "agents": selected_agents, "timestamp": datetime.now(timezone.utc).isoformat()})
    policy = PrototypeDecisionPolicy()
    initial_decision_started = time.perf_counter()
    initial_decision = policy.decide(DecisionInput(
        execution_id=execution_id, question=question.strip(), candidate_agents=selected_agents,
        task_status="pending", remaining_steps=max_agent_steps, max_agent_calls=max_agent_steps,
        ambiguous=ambiguous, risk_level="high" if elevated_risk else "low",
    ))
    initial_decision_data = initial_decision.model_dump()
    initial_decision_data["decision_metadata"]["duration_ms"] = round((time.perf_counter() - initial_decision_started) * 1000, 3)
    trace.append({"stage": "jev_decision_policy", **initial_decision_data,
                  "timestamp": datetime.now(timezone.utc).isoformat()})
    if initial_decision.action == "request_clarification":
        selected_agents = ["clarification_agent"]
    elif initial_decision.action == "request_review":
        selected_agents = ["review_agent"]
    elif initial_decision.action == "route_to_agent" and initial_decision.selected_agent:
        selected_agents = [initial_decision.selected_agent] + [a for a in selected_agents if a != initial_decision.selected_agent]
    elif initial_decision.stop:
        selected_agents = ["review_agent"]
    agent_results = dependency_errors + _run_selected_agents(
        question.strip(), selected_agents,
        query_tool=wdi_query_tool,
        document_chunks=document_chunks,
        remote_document_metadata=remote_document_metadata,
        trace=trace,
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

    # The post-agent decision records bounded retry/partial/termination outcome.
    evidence_refs = [{"source_url": source} for source in all_sources]
    decision_started = time.perf_counter()
    policy_decision = policy.decide(DecisionInput(
        execution_id=execution_id, question=question.strip(), candidate_agents=selected_agents,
        available_evidence=evidence_refs, evidence_coverage=1.0 if evidence_refs else 0.0,
        confidence=None, confidence_source="not_estimated", task_status=response_status,
        retrieval_attempts=sum(1 for result in agent_results if result.agent == "research_agent"),
        max_retrieval_attempts=int(runtime["max_retrieval_attempts"]), agent_calls=len(selected_agents),
        max_agent_calls=max_agent_steps, remaining_steps=max(0, max_agent_steps - len(selected_agents)),
        ambiguous=ambiguous,
        risk_level="high" if elevated_risk else "low",
        tool_errors=workflow_errors,
    ))
    decision_data = policy_decision.model_dump()
    decision_data["decision_metadata"]["duration_ms"] = round((time.perf_counter() - decision_started) * 1000, 3)
    trace.append({"stage": "supervisor_decision", "agents": selected_agents, "timestamp": datetime.now(timezone.utc).isoformat()})
    trace.append({"stage": "provisional_decision_policy", **decision_data, "timestamp": datetime.now(timezone.utc).isoformat()})
    trace.append({"stage": "evidence_validation", "status": validation_result.status, "evidence_count": len(all_sources), "timestamp": datetime.now(timezone.utc).isoformat()})
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    trace.append({"stage": "final_response", "status": response_status, "latency_ms": elapsed_ms, "timestamp": datetime.now(timezone.utc).isoformat()})
    log_path = trace_path or os.getenv("WORKFLOW_LOG_PATH", "data/processed/workflow_traces.jsonl")
    logging_error = append_jsonl(log_path, {"execution_id": execution_id, "started_at": started_at,
        "question": question.strip(), "selected_agents": selected_agents, "decision": decision_data,
        "evidence_count": len(all_sources), "validation_status": validation_result.status,
        "status": response_status, "latency_ms": elapsed_ms, "trace": trace})
    if logging_error:
        workflow_errors.append(f"Trace logging failed: {logging_error}")

    retrieved_chunks = [
        chunk for result in agent_results
        if result.agent == "research_agent" and isinstance(result.data, dict)
        for chunk in result.data.get("retrieved_chunks", [])
    ]
    citations = [
        citation for result in agent_results
        if result.agent == "research_agent" and isinstance(result.data, dict)
        for citation in result.data.get("citations", [])
    ]
    cited_sources = {item.get("source_url") for item in citations}
    for source in all_sources:
        if source not in cited_sources:
            citations.append({"citation_id": f"S{len(citations) + 1}", "source_url": source,
                              "evidence_type": "metadata_or_agent_source"})
    evidence_gaps = [] if has_successful_evidence else ["No citable source evidence was returned."]
    completed_at = datetime.now(timezone.utc).isoformat()
    intent = next((agent.removesuffix("_agent") for agent in selected_agents), "unknown")

    return WorkflowResponse(
        question=question.strip(), selected_agents=selected_agents, status=response_status,
        answer=final_answer, sources=all_sources, agent_results=agent_results,
        validation=validation_result, limitations=workflow_limitations, errors=workflow_errors,
        execution_id=execution_id, decision=decision_data, trace=trace, latency_ms=elapsed_ms,
        intent=intent, current_agent=selected_agents[-1] if selected_agents else None,
        retrieved_chunks=retrieved_chunks, evidence=citations, evidence_gaps=evidence_gaps,
        confidence=None, risk_level="high" if elevated_risk else "low",
        retrieval_attempts=sum(1 for result in agent_results if result.agent == "research_agent"),
        iteration_count=len(selected_agents), remaining_steps=max(0, 8 - len(selected_agents)),
        validation_status=validation_result.status, citations=citations,
        started_at=started_at, completed_at=completed_at,
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

    def decision_node(state: WorkflowState) -> dict[str, Any]:
        """Record the bounded decision policy outcome after evidence validation."""
        results = state.get("agent_results", [])
        sources = sorted({src for result in results for src in result.sources})
        decision = PrototypeDecisionPolicy().decide(DecisionInput(
            execution_id=state.get("execution_id", "langgraph-run"),
            question=state["question"], candidate_agents=state.get("selected_agents", []),
            allowed_agents=["data_agent", "financial_agent", "project_agent", "research_agent"],
            available_evidence=[{"source_url": src} for src in sources],
            evidence_coverage=1.0 if sources else 0.0,
            task_status="completed", agent_calls=len(results), max_agent_calls=8,
            remaining_steps=max(0, 8 - len(results)),
            ambiguous=state.get("selected_agents") == ["clarification_agent"],
        ))
        return {"decision": decision.model_dump(), "sources": sources}

    def predecision_node(state: WorkflowState) -> dict[str, Any]:
        """Apply permission and clarification rules before any specialist tool runs."""
        question = state["question"]
        high_risk = any(term in question.casefold() for term in ("recommend", "should we", "approve", "eligibility", "sanction"))
        decision = PrototypeDecisionPolicy().decide(DecisionInput(
            execution_id=state.get("execution_id", "langgraph-run"), question=question,
            candidate_agents=state.get("selected_agents", []), task_status="pending",
            remaining_steps=8, max_agent_calls=8, risk_level="high" if high_risk else "low",
            ambiguous=state.get("selected_agents") == ["clarification_agent"],
        ))
        agents = state.get("selected_agents", [])
        if decision.action == "request_clarification": agents = ["clarification_agent"]
        elif decision.action == "request_review": agents = ["review_agent"]
        elif decision.action == "route_to_agent" and decision.selected_agent:
            agents = [decision.selected_agent] + [a for a in agents if a != decision.selected_agent]
        return {"selected_agents": agents, "decision": decision.model_dump()}

    graph = StateGraph(WorkflowState)
    graph.add_node("input_validation", lambda state: {
        "question": state["question"].strip() if state.get("question", "").strip() else (_ for _ in ()).throw(ValueError("question must be a non-empty string"))
    })
    graph.add_node("intent_analysis_and_supervisor", route_node)
    graph.add_node("jev_decision_policy", predecision_node)
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
    graph.add_node("post_evidence_decision", decision_node)
    graph.set_entry_point("input_validation")
    graph.add_edge("input_validation", "intent_analysis_and_supervisor")
    graph.add_edge("intent_analysis_and_supervisor", "jev_decision_policy")
    graph.add_edge("jev_decision_policy", "execute")
    graph.add_edge("execute", "validate")
    graph.add_edge("validate", "post_evidence_decision")
    graph.add_edge("post_evidence_decision", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()
