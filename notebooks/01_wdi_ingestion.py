# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
import os

CATALOG = os.getenv("DATABRICKS_CATALOG", "workspace")
SCHEMA = os.getenv("DATABRICKS_SCHEMA", "default")

os.environ["DATABRICKS_CATALOG"] = CATALOG
os.environ["DATABRICKS_SCHEMA"] = SCHEMA

spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{CATALOG}`.`{SCHEMA}`")

print(f"Using catalog: {CATALOG}")
print(f"Using schema: {SCHEMA}")

# COMMAND ----------


"""Fetch configured WDI indicators for multiple countries."""
import pandas as pd

from src.data.world_bank_ingestion import fetch_indicators

country_codes = [
    code.strip().upper()
    for code in os.getenv("WDI_COUNTRIES", "GBR,IND").split(",")
    if code.strip()
]

indicator_codes = [
    code.strip()
    for code in os.getenv(
        "WDI_INDICATORS",
        "NY.GDP.MKTP.CD,SP.POP.TOTL,SL.UEM.TOTL.ZS"
    ).split(",")
    if code.strip()
]

world_bank_frames = []

for country_code in country_codes:
    print(f"Fetching World Bank data for {country_code}")

    country_indicator_frame = fetch_indicators(
        country_code,
        indicator_codes,
        base_url=os.getenv(
            "WB_API_BASE",
            "https://api.worldbank.org/v2"
        ),
        timeout=30,
        max_retries=3,
    )

    if country_indicator_frame is not None and not country_indicator_frame.empty:
        world_bank_frames.append(country_indicator_frame)
        print(
            f"Retrieved {len(country_indicator_frame)} observations "
            f"for {country_code}"
        )
    else:
        print(f"No observations found for {country_code}")

if not world_bank_frames:
    raise ValueError("No World Bank observations were retrieved.")

world_bank_indicator_frame = pd.concat(
    world_bank_frames,
    ignore_index=True
)

world_bank_spark_frame = spark.createDataFrame(
    world_bank_indicator_frame
)

display(world_bank_spark_frame)

# COMMAND ----------


from pyspark.sql import functions as F

ingestion_check = (
    world_bank_spark_frame
    .groupBy("country_code", "country", "indicator_code")
    .agg(
        F.count("*").alias("total_records"),
        F.count("value").alias("non_null_values"),
        F.min("year").alias("first_year"),
        F.max("year").alias("latest_year")
    )
    .orderBy("country_code", "indicator_code")
)

display(ingestion_check)
