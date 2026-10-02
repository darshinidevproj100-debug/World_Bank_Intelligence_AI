# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
"""Summarize WDI missingness, key duplication, and year coverage."""
from pyspark.sql import functions as F
from src.utils.config import get_table_name

table_name = get_table_name("indicators_silver")
silver_indicator_records = spark.table(table_name)

display(silver_indicator_records.groupBy("indicator_code").agg(
    F.count("*").alias("rows"),
    F.count("value").alias("non_missing_values"),
    F.countDistinct("year").alias("distinct_years"),
    F.sum(F.when(F.col("country_code").isNull(), 1).otherwise(0)).alias("missing_country_codes"),
    F.sum(F.when(F.col("year").isNull(), 1).otherwise(0)).alias("invalid_years"),
    F.countDistinct("country_code", "year").alias("country_year_pairs"),
))

bronze_indicator_records = spark.table(get_table_name("indicators_bronze"))
bronze_key_count = bronze_indicator_records.count()
bronze_distinct_key_count = bronze_indicator_records.dropDuplicates(
    ["country_code", "indicator_code", "year"]
).count()
print(f"Duplicate Bronze observation keys: {bronze_key_count - bronze_distinct_key_count}")