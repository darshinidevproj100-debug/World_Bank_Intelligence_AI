"""Safe configuration helpers for allowlisted Databricks tables."""
import os
import re

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass


IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
ALLOWED_TABLES = {
    "indicators_bronze", "indicators_silver", "indicators_gold",
    "projects_bronze", "finances_bronze", "documents_metadata",
    "document_chunks", "workflow_logs", "evaluation_results",
}


def get_table_name(table: str) -> str:
    """Return a fully qualified name after allowlisting every SQL identifier."""
    catalog = os.getenv("DATABRICKS_CATALOG", "workspace")
    schema = os.getenv("DATABRICKS_SCHEMA", "default")
    if table not in ALLOWED_TABLES:
        raise ValueError(f"Table is not allowlisted: {table}")
    for identifier_name, identifier_value in (("catalog", catalog), ("schema", schema)):
        if not IDENTIFIER_PATTERN.fullmatch(identifier_value):
            raise ValueError(f"{identifier_name} must be a simple SQL identifier")
    return f"{catalog}.{schema}.{table}"


def get_runtime_config() -> dict[str, object]:
    """Read supported runtime settings from environment with workspace-safe defaults."""
    def integer(key: str, default: int) -> int:
        try:
            value = int(os.getenv(key, str(default)))
        except ValueError as exc:
            raise ValueError(f"{key} must be an integer") from exc
        if value < 0:
            raise ValueError(f"{key} must be non-negative")
        return value
    def weight(key: str, default: float) -> float:
        try:
            value = float(os.getenv(key, str(default)))
        except ValueError as exc:
            raise ValueError(f"{key} must be numeric") from exc
        if not 0 <= value <= 1:
            raise ValueError(f"{key} must be between 0 and 1")
        return value
    config = {
        "catalog": os.getenv("DATABRICKS_CATALOG", "workspace"),
        "schema": os.getenv("DATABRICKS_SCHEMA", "default"),
        "default_country": os.getenv("DEFAULT_COUNTRY", "GBR"),
        "wdi_indicators": [x.strip() for x in os.getenv("WDI_INDICATORS", "NY.GDP.MKTP.CD,SP.POP.TOTL").split(",") if x.strip()],
        "document_storage_path": os.getenv("DOCUMENT_STORAGE_PATH", "data/documents"),
        "chunk_size": integer("CHUNK_SIZE", 1200),
        "chunk_overlap": integer("CHUNK_OVERLAP", 150),
        "retrieval_top_k": integer("RETRIEVAL_TOP_K", 5),
        "retrieval_method": os.getenv("RETRIEVAL_METHOD", "bm25"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", ""),
        "hybrid_lexical_weight": weight("HYBRID_LEXICAL_WEIGHT", .5),
        "hybrid_semantic_weight": weight("HYBRID_SEMANTIC_WEIGHT", .5),
        "max_agent_steps": integer("MAX_AGENT_STEPS", 8),
        "max_retrieval_attempts": integer("MAX_RETRIEVAL_ATTEMPTS", 2),
        "jev_policy_mode": os.getenv("JEV_POLICY_MODE", "provisional"),
        "llm_provider": os.getenv("LLM_PROVIDER", ""),
        "llm_endpoint": os.getenv("LLM_ENDPOINT", ""),
        "llm_model": os.getenv("LLM_MODEL", ""),
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "workflow_log_path": os.getenv("WORKFLOW_LOG_PATH", "data/processed/workflow_traces.jsonl"),
    }
    if config["chunk_size"] <= 0 or config["chunk_overlap"] >= config["chunk_size"]:
        raise ValueError("Require CHUNK_SIZE > CHUNK_OVERLAP >= 0")
    if config["retrieval_top_k"] < 1:
        raise ValueError("RETRIEVAL_TOP_K must be positive")
    if config["retrieval_method"] not in {"keyword", "bm25", "semantic", "hybrid"}:
        raise ValueError("RETRIEVAL_METHOD must be keyword, bm25, semantic, or hybrid")
    if config["retrieval_method"] == "hybrid" and not (config["hybrid_lexical_weight"] + config["hybrid_semantic_weight"]):
        raise ValueError("Hybrid retrieval weights cannot both be zero")
    return config
