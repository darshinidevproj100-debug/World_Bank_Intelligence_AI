"""Runtime configuration defaults and validation."""
import pytest
from src.utils.config import get_runtime_config, get_table_name

def test_runtime_configuration_uses_requested_databricks_defaults(monkeypatch):
    monkeypatch.delenv("DATABRICKS_CATALOG", raising=False)
    monkeypatch.delenv("DATABRICKS_SCHEMA", raising=False)
    config = get_runtime_config()
    assert config["catalog"] == "workspace" and config["schema"] == "default"
    assert get_table_name("indicators_gold") == "workspace.default.indicators_gold"
    assert config["default_country"] == "GBR"

def test_runtime_configuration_rejects_bad_numeric_values(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "wrong")
    with pytest.raises(ValueError, match="CHUNK_SIZE"):
        get_runtime_config()

def test_runtime_configuration_rejects_invalid_method_and_chunk_bounds(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "100")
    monkeypatch.setenv("CHUNK_OVERLAP", "100")
    with pytest.raises(ValueError, match="CHUNK_SIZE"):
        get_runtime_config()
    monkeypatch.setenv("CHUNK_OVERLAP", "10")
    monkeypatch.setenv("RETRIEVAL_METHOD", "mystery")
    with pytest.raises(ValueError, match="RETRIEVAL_METHOD"):
        get_runtime_config()
