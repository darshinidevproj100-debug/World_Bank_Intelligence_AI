"""Local integration tests for safe queries, retrieval, and agent workflow."""
import pandas as pd
import pytest

from src.agents.data_agent import run_data_agent
from src.agents.schemas import AgentResult
from src.agents.validation_agent import validate_results
from src.data.query_tools import WDIQueryTool
from src.retrieval.document_loader import load_document_chunks, retrieve_local_documents
from src.workflows.agent_graph import run_agent_workflow


def wdi_fixture() -> pd.DataFrame:
    return pd.DataFrame([
        {"country_code": "GBR", "country": "United Kingdom", "indicator_code": "NY.GDP.MKTP.CD",
         "indicator_name": "GDP (current US$)", "year": 2020, "value": 2.0, "source_url": "https://example.test/wdi"},
        {"country_code": "GBR", "country": "United Kingdom", "indicator_code": "NY.GDP.MKTP.CD",
         "indicator_name": "GDP (current US$)", "year": 2021, "value": None, "source_url": "https://example.test/wdi"},
        {"country_code": "GBR", "country": "United Kingdom", "indicator_code": "NY.GDP.MKTP.CD",
         "indicator_name": "GDP (current US$)", "year": 2022, "value": 3.0, "source_url": "https://example.test/wdi"},
    ])


def test_query_tool_applies_structured_allowlisted_filters():
    wdi_query_tool = WDIQueryTool(wdi_fixture())
    filtered_records = wdi_query_tool.query_indicator(
        country_code="GBR", indicator_code="NY.GDP.MKTP.CD", start_year=2021, end_year=2022
    )
    assert filtered_records["year"].tolist() == [2021, 2022]
    with pytest.raises(ValueError, match="indicator_code"):
        wdi_query_tool.query_indicator(country_code="GBR", indicator_code="'; DROP TABLE indicators")


def test_data_agent_summarizes_observed_data_and_preserves_null_record():
    data_result = run_data_agent(
        "How did GDP change in GBR from 2020 to 2022?",
        query_tool=WDIQueryTool(wdi_fixture()),
    )
    assert data_result.status == "success"
    assert "GDP (current US$) was 3.0 in 2022" in data_result.answer
    assert len(data_result.data) == 3
    assert data_result.data[1]["value"] is None
    assert data_result.sources == ["https://example.test/wdi"]


def test_data_agent_compares_multiple_countries_from_loaded_evidence():
    records = wdi_fixture()
    india = records.iloc[[0, 2]].copy()
    india["country_code"] = "IND"
    india["country"] = "India"
    india["value"] = [2.5, 3.5]
    tool = WDIQueryTool(pd.concat([records, india], ignore_index=True))
    result = run_data_agent("Compare GDP for the United Kingdom and India", query_tool=tool)
    assert result.status == "success"
    assert "GBR" in result.answer and "IND" in result.answer


def test_local_document_retrieval_preserves_chunk_source(tmp_path):
    source_document_path = tmp_path / "climate_note.md"
    source_document_path.write_text("Climate adaptation evidence for rural water resilience.", encoding="utf-8")
    document_chunks = load_document_chunks(tmp_path, chunk_size=32, overlap=4)
    retrieved_chunks = retrieve_local_documents("climate water resilience", document_chunks)
    assert retrieved_chunks
    assert retrieved_chunks[0]["title"] == "climate_note"
    assert retrieved_chunks[0]["source_url"].endswith("climate_note.md#chunk=0")


def test_local_workflow_returns_structured_wdi_evidence(tmp_path):
    wdi_csv_path = tmp_path / "wdi.csv"
    wdi_fixture().to_csv(wdi_csv_path, index=False)
    response = run_agent_workflow(
        "GDP in GBR between 2020 and 2022", wdi_csv_path=wdi_csv_path
    )
    assert response.selected_agents == ["data_agent"]
    assert response.status == "partial"  # claim entailment is not automated
    assert "GDP (current US$)" in response.answer
    assert response.sources == ["https://example.test/wdi"]
    assert response.validation.status == "partial"
    assert response.validation.limitations


def test_workflow_refuses_to_answer_without_local_evidence():
    response = run_agent_workflow("Compare GDP and population for GBR")
    assert response.status == "partial"
    assert "not provide enough verified evidence" in response.answer
    assert response.sources == []


def test_workflow_uses_world_bank_document_metadata(monkeypatch):
    monkeypatch.setattr(
        "src.workflows.agent_graph.fetch_document_metadata",
        lambda query: [{
            "document_key": "D100", "title": "Education Policy",
            "abstract": "The report summarizes education recovery evidence.",
            "source_url": "https://documents.worldbank.org/record?docid=100",
            "pdf_url": "https://documents.worldbank.org/100.pdf",
            "publication_date": "2024-01-15",
        }],
    )

    response = run_agent_workflow(
        "Explain education policy using reports",
        world_bank_document_query="education policy",
    )

    assert "education recovery evidence" in response.answer
    assert "World Bank D&R metadata/abstract" in response.answer
    assert response.sources == ["https://documents.worldbank.org/record?docid=100"]


def test_validation_reports_successful_result_without_citation():
    validation_result = validate_results([
        AgentResult(agent="example", status="success", summary="Unsupported claim", sources=[])
    ])
    assert validation_result.status == "partial"
    assert validation_result.data["citation_gaps"]
