"""Run repeatable local retrieval baselines and the configured workflow."""
from __future__ import annotations
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.evaluation_metrics import hit_rate_at_k, mean_reciprocal_rank, precision_at_k, recall_at_k
from src.retrieval.document_loader import load_document_chunks
from src.retrieval.retrievers import BM25Retriever, KeywordRetriever
from src.workflows.agent_graph import run_agent_workflow

def run_evaluation(*, dataset_path: str | Path = "evaluation/test_questions.csv",
                   documents_folder: str | Path | None = None,
                   wdi_csv_path: str | Path | None = None, top_k: int = 5,
                   documents: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Evaluate keyword, BM25 and full local workflow using one labelled set.

    Empty expected-document labels are deliberately excluded from retrieval metrics.
    No generation-quality score is inferred without human reference judgments.
    """
    with Path(dataset_path).open(encoding="utf-8-sig", newline="") as stream:
        cases = list(csv.DictReader(stream))
    document_records = documents if documents is not None else (load_document_chunks(documents_folder) if documents_folder else [])
    indexes = [("baseline_keyword", KeywordRetriever(document_records)), ("baseline_bm25", BM25Retriever(document_records))]
    run_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for case in cases:
        expected = {x for x in case.get("expected_document_ids", "").split("|") if x}
        for name, retriever in indexes:
            result = retriever.search(case["question"], top_k=top_k)
            ids = [r.get("document_id", "") for r in result]
            rows.append({"run_at": run_at, "test_id": case["test_id"], "configuration": name,
                "question": case["question"], "expected_agent": case["expected_agent"],
                "retrieved_ids": "|".join(ids), "retrieval_method": name.removeprefix("baseline_"),
                "hit_rate_at_k": hit_rate_at_k(expected, ids, top_k) if expected else None,
                "precision_at_k": precision_at_k(expected, ids, top_k) if expected else None,
                "recall_at_k": recall_at_k(expected, ids, top_k) if expected else None,
                "mrr": mean_reciprocal_rank(expected, ids) if expected else None,
                "status": "evaluated", "latency_ms": None, "generation_quality": "not_evaluated"})
        response = run_agent_workflow(case["question"], wdi_csv_path=wdi_csv_path,
            documents_folder=documents_folder, preloaded_document_chunks=document_records)
        expected_agents = set(case.get("expected_agent", "").split("|")) - {""}
        actual_agents = set(response.selected_agents)
        expected_action = case.get("expected_decision", "")
        rows.append({"run_at": run_at, "test_id": case["test_id"], "configuration": "proposed_local_workflow",
            "question": case["question"], "expected_agent": case["expected_agent"],
            "actual_agents": "|".join(response.selected_agents), "retrieved_ids": "|".join(response.sources),
            "retrieval_method": "configured_local", "status": response.status,
            "routing_correct": actual_agents == expected_agents,
            "expected_decision": expected_action, "decision_action": response.decision.get("action"),
            "decision_agreement": response.decision.get("action") == expected_action if expected_action else None,
            "validation_status": response.validation.status, "iteration_count": response.iteration_count,
            "latency_ms": response.latency_ms, "generation_quality": "not_evaluated"})
    return rows

def write_results(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """Write records as portable CSV, including run timestamp/configuration."""
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader(); writer.writerows(rows)
    return destination

def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline World Bank workflow evaluation")
    parser.add_argument("--dataset", default="evaluation/test_questions.csv")
    parser.add_argument("--documents-folder")
    parser.add_argument("--wdi-csv")
    parser.add_argument("--output", default="evaluation/results.csv")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    rows = run_evaluation(dataset_path=args.dataset, documents_folder=args.documents_folder,
                          wdi_csv_path=args.wdi_csv, top_k=args.top_k)
    result_path = write_results(rows, args.output)
    print(json.dumps({"runs": len(rows), "output": str(result_path),
                      "config": {"top_k": args.top_k, "documents_folder": args.documents_folder,
                                 "wdi_csv": args.wdi_csv}}, indent=2))

if __name__ == "__main__": main()
