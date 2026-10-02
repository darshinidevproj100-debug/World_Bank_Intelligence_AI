"""Load local public text documents and preserve metadata for each chunk."""
from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import TypedDict

from src.retrieval.chunking import chunk_text


SUPPORTED_DOCUMENT_SUFFIXES = {".pdf", ".txt", ".md"}
TOKEN_PATTERN = re.compile(r"[\w.-]+", re.UNICODE)
STOP_WORDS = {"about", "from", "that", "this", "with", "what", "when", "where", "which", "would"}
LOGGER = logging.getLogger(__name__)


class DocumentChunk(TypedDict):
    """One text chunk with stable local source and document metadata."""

    document_id: str
    title: str
    chunk_index: int
    page_number: int | None
    text: str
    source_url: str


def list_documents(folder: str | Path) -> list[Path]:
    """List local PDF, text, and Markdown files; missing folders are empty."""
    document_folder = Path(folder)
    if not document_folder.exists():
        return []
    return sorted(
        document_path
        for document_path in document_folder.rglob("*")
        if document_path.is_file() and document_path.suffix.lower() in SUPPORTED_DOCUMENT_SUFFIXES
    )


def _read_document_pages(document_path: Path) -> tuple[str, list[tuple[int | None, str]]]:
    """Extract page text and a source title without changing document content."""
    if document_path.suffix.lower() != ".pdf":
        return document_path.stem, [(None, document_path.read_text(encoding="utf-8", errors="replace"))]
    try:
        from pypdf import PdfReader
    except ImportError as import_error:
        raise RuntimeError("PDF extraction requires the optional dependency: pip install .[documents]") from import_error

    pdf_reader = PdfReader(str(document_path))
    document_metadata = pdf_reader.metadata
    document_title = str(document_metadata.title).strip() if document_metadata and document_metadata.title else document_path.stem
    return document_title, [
        (page_number, pdf_page.extract_text() or "")
        for page_number, pdf_page in enumerate(pdf_reader.pages, start=1)
    ]


def load_document_chunks(
    folder: str | Path, *, chunk_size: int = 1200, overlap: int = 150
) -> list[DocumentChunk]:
    """Read local supported documents and attach title, path, page, and chunk metadata."""
    document_chunks: list[DocumentChunk] = []
    for document_path in list_documents(folder):
        document_title, source_pages = _read_document_pages(document_path)
        resolved_path = str(document_path.resolve())
        for page_number, page_text in source_pages:
            for chunk_index, text_chunk in enumerate(
                chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)
            ):
                source_fragment = f"#page={page_number}" if page_number is not None else ""
                document_chunks.append(
                    {
                        "document_id": resolved_path,
                        "title": document_title,
                        "chunk_index": chunk_index,
                        "page_number": page_number,
                        "text": text_chunk,
                        "source_url": f"{resolved_path}{source_fragment}#chunk={chunk_index}",
                    }
                )
    LOGGER.info("Loaded %s local document chunks", len(document_chunks))
    return document_chunks


def retrieve_local_documents(
    question: str,
    document_chunks: list[DocumentChunk],
    *,
    top_k: int = 3,
) -> list[DocumentChunk]:
    """Return the top keyword-overlap chunks using deterministic local ranking.

    This baseline requires no embedding service. Retrieved document text is
    returned as evidence and is never executed as instructions or code.
    """
    if top_k < 1:
        raise ValueError("top_k must be positive")
    query_terms = {
        token.casefold() for token in TOKEN_PATTERN.findall(question)
        if token.casefold() not in STOP_WORDS and len(token) > 1
    }
    if not query_terms:
        return []

    scored_chunks: list[tuple[float, int, DocumentChunk]] = []
    for original_index, document_chunk in enumerate(document_chunks):
        document_terms = {
            token.casefold() for token in TOKEN_PATTERN.findall(
                f"{document_chunk['title']} {document_chunk['text']}"
            )
        }
        overlap_count = len(query_terms & document_terms)
        if overlap_count:
            scored_chunks.append((overlap_count / len(query_terms), original_index, document_chunk))
    scored_chunks.sort(key=lambda scored_chunk: (-scored_chunk[0], scored_chunk[1]))
    return [scored_chunk[2] for scored_chunk in scored_chunks[:top_k]]
