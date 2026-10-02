# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
"""Idempotently load local report files and merge chunk records into Delta."""
import json
import os
from datetime import datetime, timezone

from pyspark.sql import types as T
from src.retrieval.document_loader import load_document_chunks
from src.utils.config import get_table_name

document_path = os.getenv("DOCUMENT_STORAGE_PATH", "data/documents")
chunk_size = int(os.getenv("CHUNK_SIZE", "1200"))
chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "150"))
chunk_table = get_table_name("document_chunks")

chunks = load_document_chunks(document_path, chunk_size=chunk_size, overlap=chunk_overlap)
if not chunks:
    print(f"No supported documents found under {document_path}; no Delta changes made.")
else:
    schema = T.StructType([
        T.StructField("chunk_id", T.StringType(), False),
        T.StructField("document_id", T.StringType(), True),
        T.StructField("chunk_index", T.IntegerType(), False),
        T.StructField("title", T.StringType(), True),
        T.StructField("source_url", T.StringType(), True),
        T.StructField("page_number", T.IntegerType(), True),
        T.StructField("text", T.StringType(), False),
        T.StructField("metadata_json", T.StringType(), False),
        T.StructField("indexed_at_utc", T.StringType(), False),
    ])
    indexed_at = datetime.now(timezone.utc).isoformat()
    rows = [(
        row["chunk_id"], row["document_id"], int(row["chunk_index"]), row.get("title"),
        row.get("source_url"), row.get("page_number"), row["text"],
        json.dumps(row.get("metadata", {}), ensure_ascii=False, default=str), indexed_at,
    ) for row in chunks]
    incoming = spark.createDataFrame(rows, schema=schema)
    incoming.createOrReplaceTempView("incoming_document_chunks")
    spark.sql(f"""CREATE TABLE IF NOT EXISTS {chunk_table} (
        chunk_id STRING, document_id STRING, chunk_index INT, title STRING,
        source_url STRING, page_number INT, text STRING, metadata_json STRING,
        indexed_at_utc STRING) USING DELTA""")
    spark.sql(f"""MERGE INTO {chunk_table} AS target
        USING incoming_document_chunks AS incoming
        ON target.chunk_id = incoming.chunk_id
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *""")
    print(f"Merged {len(rows)} document chunks into {chunk_table}; Gold indicators were not modified.")
    display(spark.table(chunk_table).orderBy("document_id", "chunk_index").limit(20))
