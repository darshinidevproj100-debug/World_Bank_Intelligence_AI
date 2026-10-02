# Evaluation Guide

`evaluation/test_questions.csv` is a seed dataset with explicit labels for intent, expected agent/document/evidence/decision, clarification, risk, and category. Synthetic fixture IDs are labeled as such. Empty document labels are unjudged and are excluded from retrieval scores. The dataset is not a gold-standard correctness benchmark.

Run the portable baselines and workflow:

```powershell
python -m evaluation.run_evaluation --documents-folder data/documents --wdi-csv data/processed/wdi_sample.csv --output evaluation/results.csv --top-k 5
```

Baseline A uses query-term coverage. Baseline B uses BM25. The proposed local workflow uses its supervisor, configured local data/research sources, evidence validation, and provisional decision layer. Results share test IDs and run timestamps. Without reference judgments the evaluator records retrieval candidates but does not claim semantic answer correctness; generation quality is marked not evaluated. With no LLM configured, there is no LLM generation benchmark.

Metrics in `evaluation/evaluation_metrics.py` include precision/recall/hit rate at K, MRR, average precision, nDCG, routing and decision agreement, clarification precision/recall, evidence/citation coverage, workflow/source success helpers, human-labeled faithfulness/consistency, latency, and agent-call averages. Empty/missing labels yield defined zero or null results. `nDCG` needs relevance grades; faithfulness and unsupported-claim rates need reviewed claims. Citation presence is not entailment. No measured performance or time-saved outcome is asserted.

The Databricks notebook runs the same CSV labels against available local documents and Gold WDI export, saves a driver CSV, and appends result rows to the configured `evaluation_results` Delta table. Its actual values are workspace-dependent.
