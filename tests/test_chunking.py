"""Tests for deterministic document chunk boundaries."""
import pytest
from src.retrieval.chunking import chunk_text


def test_chunking_preserves_text_content():
    text = "abcdefghij"
    chunks = chunk_text(text, chunk_size=4, overlap=1)
    assert chunks == ["abcd", "defg", "ghij"]


def test_invalid_chunk_parameters():
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=5, overlap=5)
