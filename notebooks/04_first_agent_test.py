# Databricks notebook source
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
