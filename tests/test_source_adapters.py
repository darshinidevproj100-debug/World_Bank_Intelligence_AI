"""Tests for selected-schema CSV validation and safe table names."""
import pandas as pd
import pytest

from src.data.source_adapters import load_csv, load_finances_one, load_projects
from src.utils.config import get_table_name


def test_csv_adapter_validates_caller_selected_schema(tmp_path):
    csv_path = tmp_path / "verified_export.csv"
    pd.DataFrame({"id": ["P000001"], "label": ["Sample"]}).to_csv(csv_path, index=False)
    loaded_project_records = load_projects(csv_path, required_columns=["id", "label"])
    assert list(loaded_project_records.columns) == ["id", "label"]
    assert loaded_project_records.attrs["source_file"].endswith("verified_export.csv")
    assert loaded_project_records.attrs["ingested_at_utc"].endswith("+00:00")
    with pytest.raises(ValueError, match="missing required columns"):
        load_finances_one(csv_path, required_columns=["id", "amount"])


def test_csv_adapter_rejects_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_csv(tmp_path / "missing.csv")


def test_table_name_allows_only_valid_catalog_schema_and_table(monkeypatch):
    monkeypatch.setenv("DATABRICKS_CATALOG", "sample_catalog")
    monkeypatch.setenv("DATABRICKS_SCHEMA", "public_data")
    assert get_table_name("indicators_gold") == "sample_catalog.public_data.indicators_gold"
    monkeypatch.setenv("DATABRICKS_SCHEMA", "public_data; DROP TABLE users")
    with pytest.raises(ValueError, match="schema"):
        get_table_name("indicators_gold")
    with pytest.raises(ValueError, match="allowlisted"):
        get_table_name("users")
