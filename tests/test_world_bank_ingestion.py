"""Mocked API contract and failure-path tests for WDI ingestion."""
import json
import pytest
import requests

from src.data.world_bank_ingestion import fetch_indicator, fetch_indicators


class FakeResponse:
    def __init__(self, payload, url="https://api.worldbank.org/v2/data?page=1"):
        self.payload = payload
        self.url = url

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, *, params, timeout):
        self.calls.append((url, dict(params), timeout))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def observation(year, value):
    return {
        "countryiso3code": "GBR",
        "country": {"value": "United Kingdom"},
        "indicator": {"id": "SP.POP.TOTL", "value": "Population"},
        "date": str(year),
        "value": value,
    }


def test_fetch_indicator_paginates_and_preserves_missing_values():
    session = FakeSession([
        FakeResponse([{"pages": 2}, [observation(2021, None)]], "page-one"),
        FakeResponse([{"pages": 2}, [observation(2020, 67000000)]], "page-two"),
    ])

    rows = fetch_indicator(
        "gbr", "SP.POP.TOTL", start_year=2020, end_year=2021,
        per_page=1, timeout=4, session=session,
    )

    assert [row["year"] for row in rows] == [2021, 2020]
    assert rows[0]["value"] is None
    assert rows[0]["source_url"] == "page-one"
    assert rows[1]["source_url"] == "page-two"
    assert rows[0]["ingested_at_utc"].endswith("+00:00")
    assert [call[1]["page"] for call in session.calls] == [1, 2]
    assert all(call[1]["date"] == "2020:2021" for call in session.calls)
    assert all(call[2] == 4 for call in session.calls)


def test_fetch_indicator_retries_request_errors_with_bounded_backoff(monkeypatch):
    delays = []
    monkeypatch.setattr("src.data.world_bank_ingestion.time.sleep", delays.append)
    session = FakeSession([
        requests.Timeout("temporary timeout"),
        FakeResponse([{"pages": 1}, [observation(2020, 1)]], "page-one"),
    ])

    rows = fetch_indicator("GBR", "SP.POP.TOTL", session=session)

    assert len(rows) == 1
    assert len(session.calls) == 2
    assert delays == [1]


def test_fetch_indicator_accepts_valid_empty_response():
    session = FakeSession([FakeResponse([{"pages": 0}, None])])
    assert fetch_indicator("GBR", "SP.POP.TOTL", session=session) == []


def test_fetch_indicator_surfaces_malformed_observation():
    malformed_record = observation(2020, 1)
    del malformed_record["country"]
    session = FakeSession([FakeResponse([{"pages": 1}, [malformed_record]])])

    with pytest.raises(RuntimeError, match="missing fields"):
        fetch_indicator("GBR", "SP.POP.TOTL", session=session)


def test_fetch_indicator_archives_raw_response_separately(tmp_path):
    api_payload = [{"pages": 1}, [observation(2020, 10)]]
    session = FakeSession([FakeResponse(api_payload, "page-one")])

    fetch_indicator(
        "GBR", "SP.POP.TOTL", start_year=2020, end_year=2020,
        session=session, raw_output_dir=tmp_path / "raw",
    )

    raw_response_path = tmp_path / "raw" / "GBR_SP.POP.TOTL_2020-2020_page_0001.json"
    assert json.loads(raw_response_path.read_text(encoding="utf-8")) == api_payload


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"country": "GB"}, "country"),
        ({"indicator": "bad/code"}, "indicator"),
        ({"per_page": 0}, "per_page"),
        ({"max_retries": 0}, "max_retries"),
        ({"timeout": 0}, "timeout"),
        ({"start_year": 2022, "end_year": 2020}, "start_year"),
    ],
)
def test_fetch_indicator_validates_inputs(kwargs, message):
    options = {"country": "GBR", "indicator": "SP.POP.TOTL"}
    options.update(kwargs)
    country = options.pop("country")
    indicator = options.pop("indicator")

    with pytest.raises(ValueError, match=message):
        fetch_indicator(country, indicator, **options)


def test_fetch_indicators_keeps_stable_empty_schema():
    frame = fetch_indicators("GBR", [], session=FakeSession([]))
    assert list(frame.columns) == [
        "country_code", "country", "indicator_code", "indicator_name",
        "year", "value", "source_url", "ingested_at_utc",
    ]
    assert frame.empty


def test_fetch_indicators_supports_multiple_countries_sequentially():
    usa_observation = observation(2020, 10)
    usa_observation["countryiso3code"] = "USA"
    usa_observation["country"]["value"] = "United States"
    session = FakeSession([
        FakeResponse([{"pages": 1}, [observation(2020, 1)]], "gbr-page"),
        FakeResponse([{"pages": 1}, [usa_observation]], "usa-page"),
    ])

    frame = fetch_indicators(["GBR", "USA"], ["SP.POP.TOTL"], session=session)

    assert frame["country_code"].tolist() == ["GBR", "USA"]
    assert [call[0].split("/country/")[1].split("/")[0] for call in session.calls] == ["GBR", "USA"]
