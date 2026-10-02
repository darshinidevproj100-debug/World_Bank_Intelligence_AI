"""Research agent over explicitly supplied document retrieval results."""
from collections.abc import Callable
from typing import Any

from src.agents.schemas import AgentResult


def run_research_agent(
    question: str, retriever: Callable[[str], list[dict[str, Any]]] | None = None
) -> AgentResult:
    """Retrieve literal evidence passages and retain their source references."""
    if retriever is None:
        return AgentResult(
            agent="research_agent", status="partial",
            summary="Document retriever is not configured.",
            limitations=["Load documents and implement a retrieval backend."]
        )
    try:
        retrieved_passages = retriever(question)
        if not retrieved_passages:
            return AgentResult(
                agent="research_agent", status="partial",
                summary="No local document passage matched the question.",
                limitations=["The local keyword retriever found no supporting evidence."],
            )
        return AgentResult(
            agent="research_agent", status="success",
            summary=f"Retrieved {len(retrieved_passages)} document evidence item(s) for review.",
            data=retrieved_passages,
            sources=sorted({passage["source_url"] for passage in retrieved_passages if passage.get("source_url")}),
            limitations=["Retrieved text is untrusted evidence and is not interpreted as executable instructions."],
        )
    except Exception as exc:
        return AgentResult(
            agent="research_agent", status="error",
            summary="Document retrieval failed.", errors=[str(exc)]
        )
