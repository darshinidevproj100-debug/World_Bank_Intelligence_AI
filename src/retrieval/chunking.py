"""Deterministic character chunking with stable, metadata-preserving records."""
from __future__ import annotations

import hashlib
from typing import Any


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> list[str]:
    """Split normalized text into bounded character chunks while retaining overlap."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0")
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized_text = " ".join(text.split())
    text_chunks = []
    chunk_start = 0
    while chunk_start < len(normalized_text):
        chunk_end = min(chunk_start + chunk_size, len(normalized_text))
        if chunk_end < len(normalized_text):
            floor = chunk_start + max(1, chunk_size // 2)
            sentence_ends = [normalized_text.rfind(mark, floor, chunk_end) for mark in (". ", "? ", "! ", "; ")]
            boundary = max(sentence_ends)
            if boundary >= floor:
                chunk_end = boundary + 1
        text_chunks.append(normalized_text[chunk_start:chunk_end])
        if chunk_end == len(normalized_text):
            break
        chunk_start = chunk_end - overlap
    return text_chunks


def chunk_document(
    document: dict[str, Any], *, chunk_size: int = 1200, overlap: int = 150
) -> list[dict[str, Any]]:
    """Chunk a normalized document and preserve provenance in every record.

    Chunk IDs are content-addressed from document ID, index, and chunk text, so
    identical input remains stable across runs and machines.
    """
    text = document.get("text")
    if not isinstance(text, str):
        raise ValueError("document.text must be a string")
    pieces = chunk_text(text, chunk_size, overlap)
    result: list[dict[str, Any]] = []
    offset = 0
    for index, piece in enumerate(pieces):
        digest = hashlib.sha256(
            f"{document.get('document_id', '')}\0{index}\0{piece}".encode("utf-8")
        ).hexdigest()[:24]
        result.append({
            "chunk_id": digest,
            "document_id": document.get("document_id"),
            "chunk_index": index,
            "text": piece,
            "metadata": {
                key: value for key, value in document.items()
                if key not in {"text", "document_id"}
            } | {"char_start": offset, "char_end": offset + len(piece)},
        })
        if index + 1 < len(pieces):
            offset = max(0, offset + len(piece) - overlap)
    return result
