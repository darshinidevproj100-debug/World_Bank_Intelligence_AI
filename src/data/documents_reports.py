"""World Bank Documents & Reports metadata search adapter."""
from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

import requests

DOCUMENTS_API_URL = "https://search.worldbank.org/api/v3/wds"
MAX_PAGE_SIZE = 1000
MAX_RETRIES = 5
MAX_RESPONSE_PAGES = 20
LOGGER = logging.getLogger(__name__)


def _abstract_text(raw_abstract: Any) -> str | None:
    """Extract the text value returned for the API's abstracts metadata field."""
    if isinstance(raw_abstract, str):
        return raw_abstract or None
    if isinstance(raw_abstract, Mapping):
        abstract_value = raw_abstract.get("cdata!")
        return abstract_value if isinstance(abstract_value, str) and abstract_value else None
    return None


def fetch_document_metadata(
    query: str,
    *,
    limit: int = 10,
    offset: int = 0,
    timeout: float = 30,
    max_retries: int = 3,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """Search public D&R metadata with bounded offset pagination.

    Returned abstracts and titles are source evidence; no PDF is downloaded or
    extracted here. Search terms are passed as URL parameters, never interpolated
    into request code. Each result carries the official record and PDF URLs.
    """
    if not isinstance(query, str) or not query.strip() or len(query) > 500:
        raise ValueError("query must contain 1 to 500 characters")
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= MAX_PAGE_SIZE * 10:
        raise ValueError(f"limit must be between 1 and {MAX_PAGE_SIZE * 10}")
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise ValueError("offset must be a non-negative integer")
    if (not isinstance(timeout, (int, float)) or isinstance(timeout, bool)
            or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("timeout must be positive")
    if (not isinstance(max_retries, int) or isinstance(max_retries, bool)
            or not 1 <= max_retries <= MAX_RETRIES):
        raise ValueError(f"max_retries must be between 1 and {MAX_RETRIES}")

    client = session or requests.Session()
    documents: list[dict[str, Any]] = []
    current_offset = offset
    remaining_result_count = limit
    total_results: int | None = None
    response_page_count = 0

    while remaining_result_count > 0 and (total_results is None or current_offset < total_results):
        if response_page_count >= MAX_RESPONSE_PAGES:
            raise RuntimeError("Documents & Reports search exceeded the bounded page limit")
        response_page_count += 1
        page_size = min(remaining_result_count, MAX_PAGE_SIZE)
        request_params = {
            "format": "json",
            "qterm": query.strip(),
            "fl": "display_title,docdt,abstracts,url,pdfurl",
            "rows": page_size,
            "os": current_offset,
        }
        last_request_error: Exception | None = None
        for retry_index in range(max_retries):
            try:
                response = client.get(DOCUMENTS_API_URL, params=request_params, timeout=timeout)
                response.raise_for_status()
                response_payload = response.json()
                if not isinstance(response_payload, dict):
                    raise ValueError("Unexpected Documents & Reports API response")
                total_results = int(response_payload.get("total", 0))
                page_documents = response_payload.get("documents")
                if not isinstance(page_documents, dict):
                    raise ValueError("Documents & Reports API response has no document mapping")
                page_ingested_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
                source_query_url = response.url
                for document_key, document_payload in page_documents.items():
                    if document_key == "facets" or not isinstance(document_payload, dict):
                        continue
                    document_title = document_payload.get("display_title")
                    source_record_url = document_payload.get("url")
                    if not isinstance(document_title, str) or not isinstance(source_record_url, str):
                        raise ValueError("Documents & Reports record is missing its title or source URL")
                    raw_entity_ids = document_payload.get("entityids")
                    entity_id = raw_entity_ids.get("entityid") if isinstance(raw_entity_ids, dict) else None
                    documents.append({
                        "document_key": str(document_key),
                        "document_id": document_payload.get("id"),
                        "entity_id": entity_id,
                        "title": document_title,
                        "publication_date": document_payload.get("docdt"),
                        "abstract": _abstract_text(document_payload.get("abstracts")),
                        "source_url": source_record_url,
                        "pdf_url": document_payload.get("pdfurl"),
                        "query_url": source_query_url,
                        "ingested_at_utc": page_ingested_at_utc,
                    })
                last_request_error = None
                break
            except (requests.RequestException, ValueError, TypeError) as request_error:
                last_request_error = request_error
                if retry_index < max_retries - 1:
                    LOGGER.warning(
                        "D&R metadata query failed (attempt %s/%s); retrying",
                        retry_index + 1, max_retries,
                    )
                    time.sleep(min(2 ** retry_index, 8))
        if last_request_error is not None:
            raise RuntimeError(
                f"Documents & Reports search failed: {type(last_request_error).__name__}"
            ) from last_request_error

        returned_result_count = len(page_documents) - int("facets" in page_documents)
        if returned_result_count <= 0:
            break
        current_offset += returned_result_count
        remaining_result_count -= returned_result_count

    LOGGER.info("Retrieved %s Documents & Reports metadata records", len(documents))
    return documents[:limit]
