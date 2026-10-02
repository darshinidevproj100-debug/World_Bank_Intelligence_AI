"""CSV adapters for source exports whose exact schemas are caller-selected."""
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)


def load_csv(
    path: str | Path, *, required_columns: Sequence[str] | None = None
) -> pd.DataFrame:
    """Load a CSV export and optionally validate caller-supplied required columns.

    No source-specific column names are assumed. The caller must supply verified
    columns from the selected export when asking for schema validation.
    """
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"CSV source does not exist: {source_path}")
    try:
        source_records = pd.read_csv(source_path)
    except pd.errors.EmptyDataError as empty_error:
        raise ValueError("CSV source must contain a header row") from empty_error
    if len(source_records.columns) == 0:
        raise ValueError("CSV source must contain a header row")
    if required_columns is not None:
        required_column_names = set(required_columns)
        missing_columns = required_column_names.difference(source_records.columns)
        if missing_columns:
            raise ValueError(f"CSV is missing required columns: {sorted(missing_columns)}")
    source_records.attrs["source_file"] = str(source_path.resolve())
    source_records.attrs["ingested_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    LOGGER.info("Loaded %s records from CSV source %s", len(source_records), source_path.name)
    return source_records


def load_projects(path: str | Path, *, required_columns: Sequence[str]) -> pd.DataFrame:
    """Load a Projects & Operations export against its verified selected schema."""
    return load_csv(path, required_columns=required_columns)


def load_finances_one(path: str | Path, *, required_columns: Sequence[str]) -> pd.DataFrame:
    """Load one named Finances One export against its verified selected schema."""
    return load_csv(path, required_columns=required_columns)


def load_uk_dataset(path: str | Path, *, required_columns: Sequence[str]) -> pd.DataFrame:
    """Load a selected ONS, OHID, or NHS export with caller-verified columns."""
    return load_csv(path, required_columns=required_columns)
