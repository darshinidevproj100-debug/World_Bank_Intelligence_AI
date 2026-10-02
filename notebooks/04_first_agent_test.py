# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
"""Exercise a parameterized, allowlisted Spark query over Gold WDI rows."""
# Initial tool test before adding an LLM endpoint.
import re

from pyspark.sql import functions as F
from src.utils.config import get_table_name

INDICATORS_TABLE = get_table_name("indicators_gold")
ALLOWED_INDICATORS = {
    "NY.GDP.MKTP.CD", "SP.POP.TOTL",
    "SL.UEM.TOTL.ZS", "SP.DYN.LE00.IN",
}

def query_indicator(country_code: str, indicator_code: str, start_year: int | None = None, end_year: int | None = None):
    """Return Gold WDI rows through fixed equality and year-range filters."""
    if not re.fullmatch(r"[A-Z0-9]{3}", country_code):
        raise ValueError("country_code must be a three-character country code")
    if indicator_code not in ALLOWED_INDICATORS:
        raise ValueError("Indicator is not allowlisted")
    if start_year is not None and end_year is not None and start_year > end_year:
        raise ValueError("start_year must be <= end_year")
    indicator_query = spark.table(INDICATORS_TABLE).filter(
        (F.col("country_code") == country_code) &
        (F.col("indicator_code") == indicator_code)
    )
    if start_year is not None:
        indicator_query = indicator_query.filter(F.col("year") >= start_year)
    if end_year is not None:
        indicator_query = indicator_query.filter(F.col("year") <= end_year)
    return indicator_query.orderBy("year")

display(query_indicator("GBR", "NY.GDP.MKTP.CD"))

# COMMAND ----------


from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Read UK GDP observations from the Gold Delta table
gold_table = get_table_name("indicators_gold")

uk_gdp_records = (
    spark.table(gold_table)
    .filter(F.col("country_code") == "GBR")
    .filter(F.col("indicator_code") == "NY.GDP.MKTP.CD")
    .filter(F.col("value").isNotNull())
    .select(
        "country",
        "indicator_name",
        "year",
        "value"
    )
    .withColumn("year", F.col("year").cast("int"))
    .withColumn("value", F.col("value").cast("double"))
)

# Define a window for each country and indicator, ordered by year
gdp_window = (
    Window
    .partitionBy("country", "indicator_name")
    .orderBy("year")
)

# Calculate previous available GDP and annual percentage change
gdp_trend_records = (
    uk_gdp_records
    .withColumn(
        "previous_year_gdp",
        F.lag("value").over(gdp_window)
    )
    .withColumn(
        "annual_change_percent",
        F.when(
            F.col("previous_year_gdp").isNotNull()
            & (F.col("previous_year_gdp") != 0),
            (
                (
                    F.col("value") - F.col("previous_year_gdp")
                ) / F.col("previous_year_gdp")
            ) * 100
        )
    )
    .orderBy("year")
)

# Display the GDP trend results
display(gdp_trend_records)

# COMMAND ----------


from pyspark.sql import functions as F

# Identify the earliest and latest available GDP observations
gdp_summary_records = (
    gdp_trend_records
    .filter(F.col("value").isNotNull())
    .agg(
        F.min("year").alias("first_year"),
        F.max("year").alias("latest_year"),
        F.min_by("value", "year").alias("first_gdp"),
        F.max_by("value", "year").alias("latest_gdp")
    )
)

summary = gdp_summary_records.first()

if summary and summary["first_gdp"] is not None:
    first_gdp = summary["first_gdp"]
    latest_gdp = summary["latest_gdp"]

    total_change_percent = (
        (latest_gdp - first_gdp) / first_gdp * 100
        if first_gdp != 0 else None
    )

    print("United Kingdom GDP Trend Summary")
    print(f"First available year: {summary['first_year']}")
    print(f"Latest available year: {summary['latest_year']}")
    print(f"First GDP value: ${first_gdp:,.2f}")
    print(f"Latest GDP value: ${latest_gdp:,.2f}")

    if total_change_percent is not None:
        print(f"Total percentage change: {total_change_percent:.2f}%")
else:
    print("No valid GDP observations are available.")

# COMMAND ----------


from pyspark.sql import functions as F

def analyse_gdp_trend(country_code="GBR"):
    """Retrieve GDP evidence and produce a concise trend summary."""

    gdp_evidence = (
        spark.table(get_table_name("indicators_gold"))
        .filter(F.col("country_code") == country_code)
        .filter(F.col("indicator_code") == "NY.GDP.MKTP.CD")
        .filter(F.col("value").isNotNull())
        .select("country", "indicator_name", "year", "value", "source_url")
        .orderBy("year")
    )

    evidence_rows = gdp_evidence.collect()

    if not evidence_rows:
        return {
            "status": "no_data",
            "answer": "No GDP observations were found for this country.",
            "evidence": []
        }

    first_record = evidence_rows[0]
    latest_record = evidence_rows[-1]

    first_value = first_record["value"]
    latest_value = latest_record["value"]

    total_change = (
        ((latest_value - first_value) / first_value) * 100
        if first_value != 0 else None
    )

    answer = (
        f"{first_record['country']} GDP (current US$) changed "
        f"from ${first_value:,.2f} in {first_record['year']} "
        f"to ${latest_value:,.2f} in {latest_record['year']}."
    )

    if total_change is not None:
        answer += f" The total nominal change was {total_change:.2f}%."

    return {
        "status": "success",
        "answer": answer,
        "evidence": [
            {
                "year": row["year"],
                "value": row["value"],
                "source_url": row["source_url"]
            }
            for row in [first_record, latest_record]
        ]
    }

# Test the GDP analysis function
gdp_agent_result = analyse_gdp_trend("GBR")

print("Status:", gdp_agent_result["status"])
print("Analysis:", gdp_agent_result["answer"])
print("Evidence:", gdp_agent_result["evidence"])

# COMMAND ----------


# Test the analysis function with different country codes
test_country_codes = ["GBR", "ZZZ"]

for country_code in test_country_codes:
    test_result = analyse_gdp_trend(country_code)

    print(f"\nCountry code: {country_code}")
    print(f"Status: {test_result['status']}")
    print(f"Answer: {test_result['answer']}")

    if test_result["evidence"]:
        print(f"Evidence records: {len(test_result['evidence'])}")
