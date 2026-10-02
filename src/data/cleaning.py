"""Reusable validation and cleaning functions for indicator data."""
import pandas as pd


def clean_indicators(indicator_records: pd.DataFrame) -> pd.DataFrame:
    """Validate WDI columns, normalize numeric types, and remove duplicate keys."""
    required = {
        "country_code", "country", "indicator_code",
        "indicator_name", "year", "value"
    }
    missing = required.difference(indicator_records.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    cleaned_indicator_records = indicator_records.copy()
    cleaned_indicator_records["year"] = pd.to_numeric(
        cleaned_indicator_records["year"], errors="coerce"
    ).astype("Int64")
    cleaned_indicator_records["value"] = pd.to_numeric(
        cleaned_indicator_records["value"], errors="coerce"
    )
    cleaned_indicator_records = cleaned_indicator_records.drop_duplicates(
        subset=["country_code", "indicator_code", "year"]
    )
    cleaned_indicator_records = cleaned_indicator_records[
        cleaned_indicator_records["country_code"].notna()
    ]
    cleaned_indicator_records = cleaned_indicator_records[
        cleaned_indicator_records["indicator_code"].notna()
    ]
    return cleaned_indicator_records.reset_index(drop=True)
