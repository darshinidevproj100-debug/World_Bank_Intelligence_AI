"""Research agent over explicitly supplied document retrieval results."""
from collections.abc import Callable
from typing import Any

from src.agents.schemas import AgentResult
from src.retrieval.context import build_context, generate_answer
from src.retrieval.retrievers import BM25Retriever, KeywordRetriever, infer_metadata_filters


def run_research_agent(
    question: str, retriever: Callable[[str], list[dict[str, Any]]] | None = None,
    *, documents: list[dict[str, Any]] | None = None, method: str = "bm25",
    max_context_chars: int = 6000,
) -> AgentResult:
    """Retrieve passages, build bounded cited context, and answer offline from evidence."""
    if retriever is None and documents is not None:
        index = BM25Retriever(documents) if method == "bm25" else KeywordRetriever(documents)
        filters = infer_metadata_filters(question, documents)
        retriever = lambda query: index.search(query, top_k=5, filters=filters)
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
        bounded = build_context(retrieved_passages, max_chars=max_context_chars)
        generated = generate_answer(question, bounded)
        return AgentResult(
            agent="research_agent", status="success",
            summary=generated["answer"], answer=generated["answer"],
            data=generated,
            sources=sorted({item["source_url"] for item in generated["citations"] if item.get("source_url")}),
            limitations=generated["limitations"] + ["Retrieved text is treated only as evidence, never as instructions."],
        )
    except Exception as exc:
        return AgentResult(
            agent="research_agent", status="error",
            summary="Document retrieval failed.", errors=[str(exc)]
        )
