# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///

import os
import re

from pyspark.sql import functions as F
from src.data.world_bank_ingestion import fetch_indicators
from src.utils.config import get_table_name

country_code = os.getenv("DEFAULT_COUNTRY", "GBR").upper()

indicator_codes = [
    code.strip()
    for code in os.getenv(
        "WDI_INDICATORS",
        "NY.GDP.MKTP.CD,SP.POP.TOTL,SL.UEM.TOTL.ZS"
    ).split(",")
    if code.strip()
]

catalog = os.getenv("DATABRICKS_CATALOG", "workspace")
schema = os.getenv("DATABRICKS_SCHEMA", "default")

os.environ["DATABRICKS_CATALOG"] = catalog
os.environ["DATABRICKS_SCHEMA"] = schema

print(f"Using catalog: {catalog}")
print(f"Using schema: {schema}")

if not re.fullmatch(r"[A-Za-z0-9_]+", catalog):
    raise ValueError("Invalid catalog identifier.")

if not re.fullmatch(r"[A-Za-z0-9_]+", schema):
    raise ValueError("Invalid schema identifier.")

os.environ["DATABRICKS_CATALOG"] = catalog
os.environ["DATABRICKS_SCHEMA"] = schema

bronze_table = get_table_name("indicators_bronze")
silver_table = get_table_name("indicators_silver")
gold_table = get_table_name("indicators_gold")

spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{schema}`")

world_bank_indicator_frame = fetch_indicators(
    country_code,
    indicator_codes,
    base_url=os.getenv(
        "WB_API_BASE",
        "https://api.worldbank.org/v2"
    ),
    timeout=30,
    max_retries=3,
)

if world_bank_indicator_frame is None or world_bank_indicator_frame.empty:
    raise ValueError(
        "No WDI observations were returned. "
        "Check the country, indicators, and API response."
    )

print(
    f"Retrieved {len(world_bank_indicator_frame)} observations "
    f"for {country_code}."
)

bronze_source_records = spark.createDataFrame(
    world_bank_indicator_frame
)

(
    bronze_source_records.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(bronze_table)
)

print(f"Bronze table refreshed: {bronze_table}")

silver_indicator_records = (
    spark.table(bronze_table)
    .withColumn("year", F.col("year").cast("int"))
    .withColumn("value", F.col("value").cast("double"))
    .filter(F.col("country_code").rlike("^[A-Z0-9]{3}$"))
    .filter(F.col("indicator_code").isNotNull())
    .filter(F.col("year").between(1800, 2200))
    .dropDuplicates(
        ["country_code", "indicator_code", "year"]
    )
)

(
    silver_indicator_records.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(silver_table)
)

print(f"Silver table refreshed: {silver_table}")

gold_indicator_records = (
    spark.table(silver_table)
    .select(
        "country_code",
        "country",
        "indicator_code",
        "indicator_name",
        "year",
        "value",
        "source_url",
        "ingested_at_utc"
    )
)

(
    gold_indicator_records.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(gold_table)
)

print(f"Gold table refreshed: {gold_table}")

bronze_count = spark.table(bronze_table).count()
silver_count = spark.table(silver_table).count()
gold_count = spark.table(gold_table).count()

print(f"Bronze records: {bronze_count}")
print(f"Silver records: {silver_count}")
print(f"Gold records: {gold_count}")

display(
    spark.table(gold_table)
    .orderBy("indicator_code", "year")
    .limit(20)
)