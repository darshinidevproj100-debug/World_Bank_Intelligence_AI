# Databricks notebook source
"""Write a reproducible WDI full refresh to Bronze, Silver, and Gold Delta."""
# Assumes wdi_df is available from the ingestion notebook or is recreated here.
import os

from pyspark.sql import functions as F
from src.utils.config import get_table_name

catalog = os.getenv("DATABRICKS_CATALOG", "main")
schema = os.getenv("DATABRICKS_SCHEMA", "world_bank_ai")
# get_table_name validates these environment-provided identifiers before SQL use.
bronze_table = get_table_name("indicators_bronze")
silver_table = get_table_name("indicators_silver")
gold_table = get_table_name("indicators_gold")

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

# Bronze is a reproducible full refresh of the fetched source observations.
bronze_source_records = spark.createDataFrame(wdi_df)
bronze_source_records.write.format("delta").mode("overwrite").saveAsTable(bronze_table)

silver_indicator_records = (
    spark.table(bronze_table)
    .dropDuplicates(["country_code", "indicator_code", "year"])
    .withColumn("year", F.col("year").cast("int"))
    .withColumn("value", F.col("value").cast("double"))
    .filter(F.col("country_code").rlike("^[A-Z0-9]{3}$"))
    .filter(F.col("indicator_code").isNotNull())
    .filter(F.col("year").between(1800, 2200))
)

silver_indicator_records.write.format("delta").mode("overwrite").saveAsTable(silver_table)

gold_indicator_records = (
    spark.table(silver_table)
    # Keep null measurements visible so missing evidence is not silently hidden.
    .select("country_code", "country", "indicator_code",
            "indicator_name", "year", "value", "source_url", "ingested_at_utc")
)
gold_indicator_records.write.format("delta").mode("overwrite").saveAsTable(gold_table)

display(spark.table(gold_table).limit(20))
