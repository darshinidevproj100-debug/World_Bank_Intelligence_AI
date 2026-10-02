"""Local JSONL execution trace writer; failures are returned to caller, not hidden."""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

def append_jsonl(path: str | Path, record: dict[str, Any]) -> str | None:
    """Append one JSON-safe workflow record; return an error string on I/O failure."""
    try:
        target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return None
    except OSError as exc:
        LOGGER.warning("Workflow trace could not be written: %s", exc)
        return f"{type(exc).__name__}: {exc}"
