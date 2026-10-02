"""Isolated end-to-end demonstration using clearly synthetic, test-only fixtures."""
import pandas as pd
from src.workflows.agent_graph import run_agent_workflow

def synthetic_wdi():
    return pd.DataFrame([
        {"country_code": country, "country": name, "indicator_code": "NY.GDP.MKTP.CD",
         "indicator_name": "GDP (current US$)", "year": year, "value": value,
         "source_url": f"https://fixture.invalid/{country}/{year}"}
        for country, name, values in (("GBR", "United Kingdom", [100, 120]), ("IND", "India", [80, 110]))
        for year, value in zip((2020, 2021), values)
    ])

def test_eight_requested_demo_paths_use_fixtures_only(tmp_path, monkeypatch):
    """Synthetic rows/documents are for tests only, never represented as WB observations."""
    wdi_file = tmp_path / "synthetic_wdi_test_only.csv"
    synthetic_wdi().to_csv(wdi_file, index=False)
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "synthetic_report_test_only.md").write_text(
        "Synthetic fixture only. Development project evidence describes rural water resilience in India.", encoding="utf-8")

    gdp = run_agent_workflow("Show the GDP trend for the United Kingdom", wdi_csv_path=wdi_file, trace_path=tmp_path / "1.jsonl")
    assert "United Kingdom" in gdp.answer and gdp.sources

    compare = run_agent_workflow("Compare GDP trends for the United Kingdom and India", wdi_csv_path=wdi_file, trace_path=tmp_path / "2.jsonl")
    assert "GBR" in compare.answer and "IND" in compare.answer

    projects = run_agent_workflow("Find World Bank project information for India", trace_path=tmp_path / "3.jsonl")
    assert projects.status == "partial" and "No Projects" in projects.answer

    search = run_agent_workflow("Search reports about development projects in India", documents_folder=docs, trace_path=tmp_path / "4.jsonl")
    assert search.retrieved_chunks and search.citations

    summary = run_agent_workflow("Summarize the retrieved evidence about rural water resilience", documents_folder=docs, trace_path=tmp_path / "5.jsonl")
    assert "Synthetic fixture only" in summary.answer

    missing = run_agent_workflow("Search reports about an unindexed topic", trace_path=tmp_path / "6.jsonl")
    assert missing.status == "partial" and missing.evidence_gaps

    ambiguous = run_agent_workflow("Show me the figures", trace_path=tmp_path / "7.jsonl")
    assert ambiguous.decision["action"] == "request_clarification"

    monkeypatch.setenv("RETRIEVAL_METHOD", "semantic")
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    failed = run_agent_workflow("Search reports about water", documents_folder=docs, trace_path=tmp_path / "8.jsonl")
    assert failed.status == "partial"
    assert any("Semantic retrieval requires" in error for error in failed.errors)
