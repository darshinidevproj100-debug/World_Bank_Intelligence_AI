"""WDI data analysis agent using an explicit local or Databricks query tool."""
from __future__ import annotations

import re
from typing import Protocol

import pandas as pd

from src.agents.schemas import AgentResult


class IndicatorQueryTool(Protocol):
    """Interface implemented by safe, parameterized indicator query tools."""

    def query_indicator(
        self, *, country_code: str, indicator_code: str,
        start_year: int | None = None, end_year: int | None = None,
    ) -> pd.DataFrame:
        """Return matching indicator rows for explicit equality/date filters."""
        ...


INDICATOR_TERMS = {
    "NY.GDP.MKTP.CD": ("gdp", "gross domestic product"),
    "SP.POP.TOTL": ("population",),
    "SL.UEM.TOTL.ZS": ("unemployment",),
    "SP.DYN.LE00.IN": ("life expectancy", "life-expectancy"),
}
COUNTRY_CODE_PATTERN = re.compile(r"\b[A-Z0-9]{3}\b")
YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")


def _choose_indicator(question: str, available_codes: set[str]) -> str | None:
    """Choose a uniquely named indicator from explicit question terms."""
    normalized_question = question.casefold()
    for indicator_code, search_terms in INDICATOR_TERMS.items():
        if indicator_code in available_codes and any(term in normalized_question for term in search_terms):
            return indicator_code
    return next(iter(available_codes)) if len(available_codes) == 1 else None


def _records_with_json_nulls(query_records: pd.DataFrame) -> list[dict]:
    """Serialize pandas missing values as Python None for structured output."""
    json_safe_records = query_records.astype(object).where(pd.notna(query_records), None)
    return json_safe_records.to_dict(orient="records")


def run_data_agent(question: str, query_tool: IndicatorQueryTool | None = None) -> AgentResult:
    """Summarize one indicator series using only available WDI observations."""
    if query_tool is None:
        return AgentResult(
            agent="data_agent",
            status="partial",
            summary="Data query tool is not configured yet.",
            limitations=["Connect an approved query tool to the Gold Delta tables."]
        )
    try:
        available_country_codes = set(query_tool.indicator_records["country_code"].dropna().astype(str))
        available_indicator_codes = set(query_tool.indicator_records["indicator_code"].dropna().astype(str))
        requested_country = next(
            (code for code in COUNTRY_CODE_PATTERN.findall(question.upper()) if code in available_country_codes),
            None,
        )
        country_rows = query_tool.indicator_records[["country_code", "country"]].dropna().drop_duplicates()
        named_countries = [
            (str(row.country_code), str(row.country)) for row in country_rows.itertuples(index=False)
            if str(row.country).casefold() in question.casefold()
        ]
        selected_countries = list(dict.fromkeys(
            [code for code in COUNTRY_CODE_PATTERN.findall(question.upper()) if code in available_country_codes]
            + [code for code, _ in sorted(named_countries, key=lambda x: -len(x[1]))]
        ))
        if requested_country is None and len(named_countries) == 1:
            requested_country = named_countries[0][0]
        if requested_country is None and len(available_country_codes) == 1:
            requested_country = next(iter(available_country_codes))
        requested_indicator = _choose_indicator(question, available_indicator_codes)
        if requested_indicator is not None and len(selected_countries) > 1:
            requested_years = [int(year) for year in YEAR_PATTERN.findall(question)]
            start_year = min(requested_years) if requested_years else None
            end_year = max(requested_years) if requested_years else None
            comparison_records = pd.concat([
                query_tool.query_indicator(country_code=country_code, indicator_code=requested_indicator,
                                           start_year=start_year, end_year=end_year)
                for country_code in selected_countries
            ], ignore_index=True)
            observed = comparison_records.loc[comparison_records["value"].notna()]
            if observed.empty:
                return AgentResult(agent="data_agent", status="partial",
                    summary="The requested country comparison has no non-missing observations.",
                    data=_records_with_json_nulls(comparison_records),
                    limitations=["No numerical comparison can be computed from missing observations."])
            latest = observed.sort_values("year").groupby("country_code", as_index=False).tail(1)
            summaries = [f"{row.country} ({row.country_code}) was {row.value} in {int(row.year)}"
                         for row in latest.itertuples(index=False)]
            urls = sorted(set(comparison_records.get("source_url", pd.Series(dtype=str)).dropna().astype(str)))
            return AgentResult(agent="data_agent", status="success",
                summary=f"Latest available {requested_indicator} observations: " + "; ".join(summaries) + ".",
                data=_records_with_json_nulls(comparison_records), sources=urls,
                limitations=["The indicator values are compared as reported; no causal interpretation is implied."])

        if requested_country is None or requested_indicator is None:
            return AgentResult(
                agent="data_agent", status="partial",
                summary="I could not select one country and indicator from the loaded data.",
                limitations=["Specify a country code and an indicator available in the loaded WDI dataset."],
            )

        requested_years = [int(year) for year in YEAR_PATTERN.findall(question)]
        requested_start_year = min(requested_years) if requested_years else None
        requested_end_year = max(requested_years) if requested_years else None
        query_records = query_tool.query_indicator(
            country_code=requested_country,
            indicator_code=requested_indicator,
            start_year=requested_start_year,
            end_year=requested_end_year,
        )
        observed_records = query_records.loc[query_records["value"].notna()]
        if observed_records.empty:
            return AgentResult(
                agent="data_agent", status="partial",
                summary="The selected WDI series has no non-missing values for the requested period.",
                data=_records_with_json_nulls(query_records),
                sources=sorted(set(query_records.get("source_url", pd.Series(dtype=str)).dropna().astype(str))),
                limitations=["No numerical comparison can be computed from missing observations."],
            )

        earliest = observed_records.iloc[0]
        latest = observed_records.iloc[-1]
        indicator_name = str(latest.get("indicator_name") or requested_indicator)
        country_name = str(latest.get("country") or requested_country)
        answer = (
            f"{country_name}: {indicator_name} was {latest['value']} in {int(latest['year'])}. "
            f"The earliest non-missing observation in the selected records was "
            f"{earliest['value']} in {int(earliest['year'])}."
        )
        if len(observed_records) > 1:
            answer += " These observations describe a change over time; they do not establish causation."
        evidence_urls = sorted(set(query_records.get("source_url", pd.Series(dtype=str)).dropna().astype(str)))
        return AgentResult(
            agent="data_agent", status="success", summary=answer,
            data=_records_with_json_nulls(query_records), sources=evidence_urls,
            limitations=["Units are reported only when present in the indicator name or source metadata."],
        )
    except Exception as exc:
        return AgentResult(
            agent="data_agent",
            status="error",
            summary="The data query failed.",
            errors=[str(exc)]
        )
