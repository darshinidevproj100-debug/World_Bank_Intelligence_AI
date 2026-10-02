# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
"""Reproducible keyword/BM25/workflow evaluation with CSV and optional Delta output."""
import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.utils.config import get_table_name
from evaluation.run_evaluation import run_evaluation, write_results

run_id = str(uuid.uuid4())
run_timestamp = datetime.now(timezone.utc).isoformat()
catalog = os.getenv("DATABRICKS_CATALOG", "workspace")
schema = os.getenv("DATABRICKS_SCHEMA", "default")
os.environ["DATABRICKS_CATALOG"] = catalog
os.environ["DATABRICKS_SCHEMA"] = schema

gold_table = get_table_name("indicators_gold")
document_table = get_table_name("document_chunks")
evaluation_table = get_table_name("evaluation_results")

# Smoke check: report missing workspace tables and continue with available local fixtures.
has_gold = spark.catalog.tableExists(gold_table)
has_chunks = spark.catalog.tableExists(document_table)
print(f"Run ID: {run_id}; catalog/schema: {catalog}.{schema}")
print(f"Gold table available: {has_gold}; indexed chunks available: {has_chunks}")

wdi_path = None
temporary_wdi_path = None
if has_gold:
    wdi_frame = spark.table(gold_table).toPandas()
    handle = tempfile.NamedTemporaryFile(prefix="wb_wdi_eval_", suffix=".csv", delete=False)
    temporary_wdi_path = handle.name
    handle.close()
    wdi_frame.to_csv(temporary_wdi_path, index=False)
    wdi_path = temporary_wdi_path

indexed_chunks = None
if has_chunks:
    indexed_chunks = [
        {"chunk_id": row.chunk_id, "document_id": row.document_id,
         "chunk_index": row.chunk_index, "title": row.title,
         "source_url": row.source_url, "page_number": row.page_number,
         "text": row.text, "metadata": json.loads(row.metadata_json or "{}")}
        for row in spark.table(document_table).select(
            "chunk_id", "document_id", "chunk_index", "title", "source_url",
            "page_number", "text", "metadata_json").toLocalIterator()
    ]

dataset = Path("evaluation/test_questions.csv")
documents_folder = os.getenv("DOCUMENT_STORAGE_PATH", "data/documents")
records = run_evaluation(dataset_path=dataset, documents_folder=documents_folder,
    wdi_csv_path=wdi_path, top_k=int(os.getenv("RETRIEVAL_TOP_K", "5")),
    documents=indexed_chunks)
for row in records:
    row["run_id"] = run_id
    row["run_timestamp_utc"] = run_timestamp
    row["catalog"] = catalog
    row["schema"] = schema
    row["llm_generation"] = "disabled_unless_explicitly_configured"

display(spark.createDataFrame(records))

# Always export an inspectable CSV on the driver; this path is ephemeral unless copied to durable storage.
csv_output = f"/tmp/world_bank_evaluation_{run_id}.csv"
write_results(records, csv_output)
print(f"CSV results written to driver path: {csv_output}")

evaluation_df = spark.createDataFrame(records)
evaluation_df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(evaluation_table)
print(f"Appended evaluation records to {evaluation_table}")
