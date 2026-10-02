# Databricks notebook source
"""Fetch configured WDI indicators for a single configured country."""
# Run on Databricks compute. This notebook uses the project source module.
# If the repository root is not on sys.path, configure the project as a package
# or add the repository root using the workspace-supported method.
import os

from src.data.world_bank_ingestion import fetch_indicators

country_code = os.getenv("DEFAULT_COUNTRY", "GBR").upper()
indicator_codes = [
    code.strip() for code in os.getenv(
        "WDI_INDICATORS", "NY.GDP.MKTP.CD,SP.POP.TOTL,SL.UEM.TOTL.ZS"
    ).split(",") if code.strip()
]

world_bank_indicator_frame = fetch_indicators(
    country_code, indicator_codes,
    base_url=os.getenv("WB_API_BASE", "https://api.worldbank.org/v2"),
    timeout=30, max_retries=3,
)
display(spark.createDataFrame(world_bank_indicator_frame))
