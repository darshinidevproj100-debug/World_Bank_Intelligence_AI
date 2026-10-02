"""World Bank Indicators API ingestion with bounded pagination and retries."""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

DEFAULT_BASE = "https://api.worldbank.org/v2"
LOGGER = logging.getLogger(__name__)
MAX_RETRIES = 10
MAX_PAGES = 10_000
_COUNTRY_RE = re.compile(r"^[A-Z0-9]{3}$")
_INDICATOR_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def fetch_indicator(
    country: str,
    indicator: str,
    *,
    start_year: int | None = None,
    end_year: int | None = None,
    base_url: str = DEFAULT_BASE,
    per_page: int = 1000,
    max_retries: int = 3,
    timeout: float = 30,
    session: requests.Session | None = None,
    raw_output_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Fetch all available pages for one country-indicator pair."""
    country = country.strip().upper() if isinstance(country, str) else ""
    if not _COUNTRY_RE.fullmatch(country):
        raise ValueError("country must be a three-character World Bank country code")
    if not isinstance(indicator, str) or not _INDICATOR_RE.fullmatch(indicator):
        raise ValueError("indicator must be a valid indicator code")
    if not isinstance(per_page, int) or isinstance(per_page, bool) or per_page < 1:
        raise ValueError("per_page must be positive")
    if not isinstance(max_retries, int) or isinstance(max_retries, bool) or not 1 <= max_retries <= MAX_RETRIES:
        raise ValueError(f"max_retries must be between 1 and {MAX_RETRIES}")
    if (not isinstance(timeout, (int, float)) or isinstance(timeout, bool)
            or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("timeout must be positive")
    for name, year in (("start_year", start_year), ("end_year", end_year)):
        if year is not None and (not isinstance(year, int) or isinstance(year, bool) or year < 1 or year > 9999):
            raise ValueError(f"{name} must be an integer year between 1 and 9999")
    if start_year is not None and end_year is not None and start_year > end_year:
        raise ValueError("start_year must be <= end_year")

    client = session or requests.Session()
    url = f"{base_url.rstrip('/')}/country/{country}/indicator/{indicator}"
    LOGGER.info("Fetching World Bank indicator %s for %s", indicator, country)
    params: dict[str, Any] = {"format": "json", "per_page": per_page, "page": 1}
    if start_year is not None or end_year is not None:
        params["date"] = f"{start_year if start_year is not None else 1960}:{end_year if end_year is not None else 2100}"

    observation_records_with_source_metadata: list[tuple[dict[str, Any], str, str]] = []
    page_count = 1

    while params["page"] <= page_count:
        error: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = client.get(url, params=params, timeout=timeout)
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, list) or len(payload) != 2:
                    raise ValueError("Unexpected World Bank API response structure")
                response_metadata, response_records = payload[0], payload[1]
                if not isinstance(response_metadata, dict):
                    raise ValueError("Unexpected World Bank API metadata")
                try:
                    reported_pages = int(response_metadata.get("pages", 1))
                except (TypeError, ValueError) as exc:
                    raise ValueError("World Bank API returned invalid page metadata") from exc
                if reported_pages < 0 or reported_pages > MAX_PAGES:
                    raise ValueError("World Bank API returned invalid page count")
                page_count = reported_pages
                # The API uses a null second element when a valid query has no observations.
                if response_records is None:
                    response_records = []
                if not isinstance(response_records, list) or any(
                    not isinstance(observation_record, dict) for observation_record in response_records
                ):
                    raise ValueError("Unexpected World Bank API observation structure")
                if raw_output_dir is not None:
                    raw_folder = Path(raw_output_dir)
                    raw_folder.mkdir(parents=True, exist_ok=True)
                    year_range = f"{start_year or 'start'}-{end_year or 'end'}"
                    raw_filename = f"{country}_{indicator}_{year_range}_page_{params['page']:04d}.json"
                    (raw_folder / raw_filename).write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                page_ingested_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
                observation_records_with_source_metadata.extend(
                    (observation_record, response.url, page_ingested_at_utc)
                    for observation_record in response_records
                )
                error = None
                LOGGER.debug("Fetched WDI page %s of %s for %s/%s", params["page"], page_count, country, indicator)
                break
            except (requests.RequestException, ValueError, TypeError) as exc:
                error = exc
                if attempt < max_retries - 1:
                    LOGGER.warning(
                        "WDI request failed for %s/%s page %s (attempt %s/%s); retrying",
                        country, indicator, params["page"], attempt + 1, max_retries,
                    )
                    time.sleep(min(2 ** attempt, 8))
        if error is not None:
            raise RuntimeError(f"Failed to fetch {country}/{indicator}: {error}") from error
        params["page"] += 1

    wdi_observation_rows = []
    for observation_record, source_url, ingested_at_utc in observation_records_with_source_metadata:
        required_observation_fields = {"date", "value", "country", "indicator", "countryiso3code"}
        missing_observation_fields = required_observation_fields.difference(observation_record)
        if missing_observation_fields:
            raise RuntimeError(
                f"WDI observation is missing fields: {sorted(missing_observation_fields)}"
            )
        country_metadata = observation_record.get("country") or {}
        indicator_metadata = observation_record.get("indicator") or {}
        if not isinstance(country_metadata, dict) or not isinstance(indicator_metadata, dict):
            raise RuntimeError(f"Unexpected observation metadata for {country}/{indicator}")
        if "value" not in country_metadata or "id" not in indicator_metadata or "value" not in indicator_metadata:
            raise RuntimeError(f"Incomplete country or indicator metadata for {country}/{indicator}")
        observation_date = observation_record.get("date")
        try:
            observation_year = int(observation_date) if observation_date not in (None, "") else None
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Invalid observation year for {country}/{indicator}: {observation_date!r}"
            ) from exc
        wdi_observation_rows.append(
            {
                "country_code": observation_record.get("countryiso3code") or country,
                "country": country_metadata.get("value"),
                "indicator_code": indicator_metadata.get("id") or indicator,
                "indicator_name": indicator_metadata.get("value"),
                "year": observation_year,
                # Preserve the API's null value for missing observations.
                "value": observation_record.get("value"),
                "source_url": source_url,
                "ingested_at_utc": ingested_at_utc,
            }
        )
    LOGGER.info("Fetched %s WDI observations for %s/%s", len(wdi_observation_rows), country, indicator)
    return wdi_observation_rows


def fetch_indicators(
    country: str | list[str],
    indicators: list[str],
    *,
    start_year: int | None = None,
    end_year: int | None = None,
    base_url: str = DEFAULT_BASE,
    per_page: int = 1000,
    max_retries: int = 3,
    timeout: float = 30,
    session: requests.Session | None = None,
    raw_output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Fetch indicators for one or more countries using sequential API calls."""
    if len(indicators) > 60:
        raise ValueError("at most 60 indicators may be requested at once")
    country_codes = [country] if isinstance(country, str) else country
    if not isinstance(country_codes, list) or len(country_codes) > 300:
        raise ValueError("country must be a code or a list containing at most 300 codes")
    if any(
        not isinstance(country_code, str)
        or not _COUNTRY_RE.fullmatch(country_code.strip().upper())
        for country_code in country_codes
    ):
        raise ValueError("every country must be a three-character World Bank country code")
    wdi_observation_rows: list[dict[str, Any]] = []
    for country_code in country_codes:
        for indicator in indicators:
            wdi_observation_rows.extend(fetch_indicator(
                country_code, indicator, start_year=start_year, end_year=end_year,
                base_url=base_url, per_page=per_page, max_retries=max_retries,
                timeout=timeout, session=session, raw_output_dir=raw_output_dir,
            ))
    return pd.DataFrame(wdi_observation_rows, columns=[
        "country_code", "country", "indicator_code", "indicator_name",
        "year", "value", "source_url", "ingested_at_utc"
    ])


def main() -> None:
    """Parse local CLI settings, fetch observations, and write the CSV output."""
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--country", default=os.getenv("DEFAULT_COUNTRY", "GBR"),
        help="One country code or comma-separated country codes",
    )
    parser.add_argument(
        "--indicators", default=os.getenv("WDI_INDICATORS", "NY.GDP.MKTP.CD,SP.POP.TOTL")
    )
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--output", default="data/processed/wdi_sample.csv")
    parser.add_argument("--raw-output-dir", default="data/raw/wdi")
    parser.add_argument("--base-url", default=os.getenv("WB_API_BASE", DEFAULT_BASE))
    parser.add_argument("--per-page", type=int, default=1000)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30)
    cli_arguments = parser.parse_args()

    country_codes = [code.strip().upper() for code in cli_arguments.country.split(",") if code.strip()]
    if not country_codes:
        parser.error("--country must contain at least one country code")
    indicator_codes = [code.strip() for code in cli_arguments.indicators.split(",") if code.strip()]
    if not indicator_codes:
        parser.error("--indicators must contain at least one indicator code")
    world_bank_indicator_frame = fetch_indicators(
        country_codes,
        indicator_codes,
        base_url=cli_arguments.base_url,
        per_page=cli_arguments.per_page,
        max_retries=cli_arguments.max_retries,
        timeout=cli_arguments.timeout,
        raw_output_dir=cli_arguments.raw_output_dir,
        start_year=cli_arguments.start_year,
        end_year=cli_arguments.end_year,
    )
    output_path = Path(cli_arguments.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    world_bank_indicator_frame.to_csv(output_path, index=False)
    print(f"Saved {len(world_bank_indicator_frame)} rows to {output_path.resolve()}")


if __name__ == "__main__":
    main()
