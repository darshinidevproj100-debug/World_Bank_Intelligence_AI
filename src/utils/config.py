"""Safe configuration helpers for allowlisted Databricks tables."""
import os
import re


IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
ALLOWED_TABLES = {
    "indicators_bronze", "indicators_silver", "indicators_gold",
    "projects_bronze", "finances_bronze", "documents_metadata",
}


def get_table_name(table: str) -> str:
    """Return a fully qualified name after allowlisting every SQL identifier."""
    catalog = os.getenv("DATABRICKS_CATALOG", "main")
    schema = os.getenv("DATABRICKS_SCHEMA", "world_bank_ai")
    if table not in ALLOWED_TABLES:
        raise ValueError(f"Table is not allowlisted: {table}")
    for identifier_name, identifier_value in (("catalog", catalog), ("schema", schema)):
        if not IDENTIFIER_PATTERN.fullmatch(identifier_value):
            raise ValueError(f"{identifier_name} must be a simple SQL identifier")
    return f"{catalog}.{schema}.{table}"
