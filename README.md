# World Bank Intelligence AI

A configurable project for World Bank indicator queries, local report retrieval, bounded specialist-agent workflows, and reproducible evaluation. The local path works without Databricks, an LLM, or a managed vector database.

## Components

- WDI ingestion, cleaning, safe query tools, and the existing Bronze/Silver/Gold Delta notebook sequence.
- TXT, Markdown, CSV, and optional PDF document loading; deterministic chunking; keyword and BM25 retrieval; optional injected embeddings and hybrid ranking; JSON local vector storage.
- Evidence-bounded offline response generation and structured source validation.
- A keyword supervisor and a replaceable **provisional decision-policy interface**. The project has no supplied formal JEV definition; see [JEV_DECISION_LAYER.md](JEV_DECISION_LAYER.md).
- Local JSONL execution traces, CLI application, and portable evaluation metrics.

See [ARCHITECTURE.md](ARCHITECTURE.md), [RAG_GUIDE.md](RAG_GUIDE.md), [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md), [DATABRICKS_RUNBOOK.md](DATABRICKS_RUNBOOK.md), and [REQUIREMENTS_TRACEABILITY.md](REQUIREMENTS_TRACEABILITY.md) for implementation detail and limitations.

## Local setup

Python 3.10 or newer:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional dependencies:

```powershell
pip install ".[documents]"  # PDF extraction
pip install ".[agents]"     # LangGraph adapter
```

Configure through environment variables or a private `.env` file. `.env.example` has safe defaults and no credentials. Default Databricks identifiers are `workspace.default`; the configured Gold table is `workspace.default.indicators_gold`.

## Ingest WDI

```powershell
python -m src.data.world_bank_ingestion --country GBR --indicators NY.GDP.MKTP.CD,SP.POP.TOTL --start-year 2020 --end-year 2022 --output data/processed/wdi_sample.csv
```

The public Indicators API needs no key. This command writes a local CSV and archives raw API pages under `data/raw/wdi/`.

## Ask a question

```powershell
python -m app.app --question "Show GDP trends for GBR from 2020 to 2022" --wdi-csv data/processed/wdi_sample.csv --json
python -m app.app --question "Summarize rural water evidence" --documents-folder data/documents
python -m app.app --question "Find reports about education" --search-world-bank-documents
```

The app reports selected agents, provisional decision and reason, evidence validation, citations/sources, limitations, errors, execution ID, latency, and traceable JSON output. The Documents & Reports search requires internet access and returns metadata/abstracts, not downloaded full text. Finance/project agent results remain explicitly unavailable without configured, verified datasets.

## Retrieval and evaluation

Keyword and BM25 work offline. Semantic retrieval requires an explicitly injected embedding model; no model is called or silently substituted by default. Hybrid retrieval combines normalized lexical and semantic rankings. Read [RAG_GUIDE.md](RAG_GUIDE.md).

```powershell
python -m evaluation.run_evaluation --documents-folder data/documents --wdi-csv data/processed/wdi_sample.csv --output evaluation/results.csv --top-k 5
```

Evaluation labels are a seed set, not a validated benchmark. Empty document labels are unjudged, and no generation quality is claimed without reviewed references. See [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md).

## Tests

```powershell
python -m pytest -q
```

Tests use mocked APIs and synthetic fixtures. They do not need internet, credentials, Spark, or Databricks.

## Databricks notebooks

Run in this order: `01_wdi_ingestion.py` → `02_delta_medallion.py` → `03_data_quality.py` → `04_first_agent_test.py` → `04b_document_ingestion.py` → `05_supervisor_workflow.py` → `06_workflow_evaluation.py`.

The notebooks are configuration-dependent examples; they have not been run in the target workspace. Workspace entitlements, Unity Catalog access, optional LangGraph/PDF dependencies, and managed AI services must be verified separately. The WDI Medallion notebook performs a full refresh; document indexing MERGEs chunks separately and does not modify Gold. See [DATABRICKS_RUNBOOK.md](DATABRICKS_RUNBOOK.md).

## Current limitations

- No formal JEV definition was found or implemented; decision behavior is provisional and configurable.
- Semantic retrieval and hosted LLM generation require external adapters that are not configured here.
- Deterministic citation/source checks do not prove claim entailment or contradiction absence.
- No OCR, remote report-PDF download, managed vector backend, or deployed app is included.
- Finance and project source schemas remain to be selected and connected.
