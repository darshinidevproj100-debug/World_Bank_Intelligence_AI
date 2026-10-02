# Databricks Runbook

## Configuration and smoke checks

Defaults are `DATABRICKS_CATALOG=workspace`, `DATABRICKS_SCHEMA=default`, `DEFAULT_COUNTRY=GBR`, and Gold `workspace.default.indicators_gold`. Set these as job/notebook environment values; identifiers are validated and tables are allowlisted. Install the repository package and only needed extras on the cluster (`pip install -e .`; add `.[agents,documents]` if LangGraph/PDF extraction are needed). Do not configure secrets in notebooks or tracked files.

Workspace paths and driver-local paths differ. `DOCUMENT_STORAGE_PATH` must point to files readable by the notebook driver (for example a mounted volume path); `/tmp` exports are ephemeral. For large document corpora, stage files on a supported shared store before driver-side extraction. The JSON vector fallback is local and not a replacement for a managed distributed index.

## Notebook order

1. `notebooks/01_wdi_ingestion.py` fetches configured WDI observations and reports coverage.
2. `notebooks/02_delta_medallion.py` writes configured Bronze, Silver, and Gold tables. It is a full refresh of those WDI tables.
3. `notebooks/03_data_quality.py` summarizes missingness and duplicate keys.
4. `notebooks/04_first_agent_test.py` performs a parameterized Gold query and trend example.
5. `notebooks/04b_document_ingestion.py` chunks PDF/TXT/MD/CSV files and idempotently MERGEs `document_chunks` by stable chunk ID. It never writes Gold.
6. `notebooks/05_supervisor_workflow.py` runs the Databricks-oriented supervisor and logs optional Delta traces.
7. `notebooks/06_workflow_evaluation.py` smoke-checks configured Gold/chunk tables, evaluates keyword/BM25/workflow cases, exports driver CSV, and appends Delta results.

Restartable cells should be run with the repository root/package available on Python's import path. The notebooks have not been executed against this workspace. Unity Catalog permissions, Spark/Delta behavior, LangGraph installation, network access, model serving, Vector Search, MLflow features, and Apps are not verified. A missing source table is reported or causes the relevant query to return a source error; a missing indexing table allows evaluation to fall back to configured local files.

No notebook overwrites the Gold table for document indexing. The Medallion notebook refreshes Gold only as part of the existing WDI ingestion path.
