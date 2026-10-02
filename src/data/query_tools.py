"""Allowlisted local query functions for cleaned World Bank indicator rows."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.cleaning import clean_indicators


@dataclass
class WDIQueryTool:
    """Query WDI records using explicit equality and year-range filters.

    The tool accepts structured filter values, never SQL or model-generated code.
    Values must already occur in the loaded dataset, which keeps the query scope
    limited to the supplied and validated data.
    """

    indicator_records: pd.DataFrame

    def __post_init__(self) -> None:
        """Validate and clean source rows when the query tool is created."""
        self.indicator_records = clean_indicators(self.indicator_records)

    def query_indicator(
        self,
        *,
        country_code: str,
        indicator_code: str,
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> pd.DataFrame:
        """Return matching records; raise ValueError for unsupported filters."""
        if country_code not in set(self.indicator_records["country_code"].dropna().astype(str)):
            raise ValueError("country_code is not present in the loaded indicator data")
        if indicator_code not in set(self.indicator_records["indicator_code"].dropna().astype(str)):
            raise ValueError("indicator_code is not present in the loaded indicator data")
        if start_year is not None and end_year is not None and start_year > end_year:
            raise ValueError("start_year must be <= end_year")

        matching_records = self.indicator_records.loc[
            (self.indicator_records["country_code"].astype(str) == country_code)
            & (self.indicator_records["indicator_code"].astype(str) == indicator_code)
        ].copy()
        if start_year is not None:
            matching_records = matching_records.loc[matching_records["year"] >= start_year]
        if end_year is not None:
            matching_records = matching_records.loc[matching_records["year"] <= end_year]
        return matching_records.sort_values("year", na_position="last").reset_index(drop=True)


def load_wdi_query_tool(csv_path: str) -> WDIQueryTool:
    """Load a local WDI CSV and return a validated query tool."""
    indicator_records = pd.read_csv(csv_path)
    return WDIQueryTool(indicator_records)
