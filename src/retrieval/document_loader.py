"""Resilient local document loading and a dependency-free keyword baseline."""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

from src.retrieval.chunking import chunk_document

SUPPORTED_DOCUMENT_SUFFIXES = {".pdf", ".txt", ".md", ".csv"}
TOKEN_PATTERN = re.compile(r"[\w.-]+", re.UNICODE)
STOP_WORDS = {"about", "from", "that", "this", "with", "what", "when", "where", "which", "would"}
LOGGER = logging.getLogger(__name__)


class DocumentChunk(TypedDict, total=False):
    """A passage with stable IDs and source metadata."""
    chunk_id: str
    document_id: str
    title: str
    chunk_index: int
    page_number: int | None
    text: str
    source_url: str
    metadata: dict[str, Any]


def list_documents(folder: str | Path) -> list[Path]:
    """Return supported files in deterministic order; absent folders are empty."""
    root = Path(folder)
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_DOCUMENT_SUFFIXES)


def _read_document_pages(path: Path) -> tuple[str, list[tuple[int | None, str]]]:
    if path.suffix.lower() != ".pdf":
        if path.suffix.lower() == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            return path.stem, [(None, "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows))]
        return path.stem, [(None, path.read_text(encoding="utf-8", errors="replace"))]
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF extraction requires pypdf: pip install .[documents]") from exc
    reader = PdfReader(str(path))
    metadata = reader.metadata
    title = str(metadata.title).strip() if metadata and metadata.title else path.stem
    return title, [(number, page.extract_text() or "") for number, page in enumerate(reader.pages, 1)]


def make_document(text: str, *, document_id: str = "inline", title: str = "Untitled",
                  source: str = "inline", metadata: dict[str, Any] | None = None,
                  source_url: str | None = None) -> dict[str, Any]:
    """Create the shared document schema without inferring absent metadata."""
    safe_metadata = metadata if isinstance(metadata, dict) else {}
    return {
        "document_id": document_id, "title": title, "source": source,
        "source_url": source_url, "country_code": safe_metadata.get("country_code"),
        "country_name": safe_metadata.get("country_name"),
        "publication_year": safe_metadata.get("publication_year"),
        "topic": safe_metadata.get("topic"), "page_number": safe_metadata.get("page_number"),
        "text": text if isinstance(text, str) else "", "metadata": safe_metadata,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }


def load_documents(folder: str | Path) -> list[dict[str, Any]]:
    """Load supported files independently, logging and skipping malformed files."""
    documents: list[dict[str, Any]] = []
    seen: set[str] = set()
    seen_content: set[str] = set()
    root = Path(folder)
    for path in list_documents(root):
        try:
            title, pages = _read_document_pages(path)
            relative = str(path.relative_to(root)).replace("\\", "/")
            content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if content_hash in seen_content:
                LOGGER.info("Skipping duplicate document content at %s", path)
                continue
            seen_content.add(content_hash)
            digest = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:24]
            if digest in seen:
                continue
            seen.add(digest)
            for page, text in pages:
                metadata = {"relative_path": relative, "file_type": path.suffix.lower()}
                documents.append(make_document(text, document_id=digest, title=title,
                    source=relative, source_url=str(path.resolve()), metadata=metadata,
                    ))
                documents[-1]["page_number"] = page
        except Exception as exc:
            LOGGER.warning("Skipping document %s (%s): %s", path, type(exc).__name__, exc)
    return documents


def load_document_chunks(folder: str | Path, *, chunk_size: int = 1200,
                         overlap: int = 150) -> list[DocumentChunk]:
    """Load documents into stable chunks while retaining old source_url fields."""
    chunks: list[DocumentChunk] = []
    for doc in load_documents(folder):
        for chunk in chunk_document(doc, chunk_size=chunk_size, overlap=overlap):
            meta = chunk["metadata"]
            page = doc.get("page_number")
            fragment = f"#page={page}" if page is not None else ""
            chunks.append({
                "chunk_id": chunk["chunk_id"], "document_id": doc["document_id"],
                "title": doc["title"], "chunk_index": chunk["chunk_index"],
                "page_number": page, "text": chunk["text"],
                "source_url": f"{doc['source_url']}{fragment}#chunk={chunk['chunk_index']}",
                "metadata": meta,
            })
    LOGGER.info("Loaded %d document chunks", len(chunks))
    return chunks


def retrieve_local_documents(question: str, document_chunks: list[DocumentChunk], *,
                             top_k: int = 3) -> list[DocumentChunk]:
    """Rank local passages by query-term coverage, preserving deterministic ties."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    query_terms = {t.casefold() for t in TOKEN_PATTERN.findall(question)
                   if t.casefold() not in STOP_WORDS and len(t) > 1}
    if not query_terms:
        return []
    ranked = []
    for index, chunk in enumerate(document_chunks):
        terms = {t.casefold() for t in TOKEN_PATTERN.findall(
            f"{chunk.get('title', '')} {chunk.get('text', '')}")}
        overlap = len(query_terms & terms)
        if overlap:
            ranked.append((overlap / len(query_terms), index, chunk))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in ranked[:top_k]]
