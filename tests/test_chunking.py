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


def test_empty_and_short_text_do_not_create_empty_chunks():
    assert chunk_text("  \n  ", chunk_size=5, overlap=1) == []
    assert chunk_text("short text", chunk_size=20, overlap=3) == ["short text"]


def test_long_text_has_expected_overlap_and_keeps_all_words():
    result = chunk_text("one two three four five six seven eight nine ten", chunk_size=15, overlap=4)
    assert len(result) > 1
    assert all(result[i][-4:] == result[i + 1][:4] for i in range(len(result) - 1))
    assert "ten" in result[-1]


def test_sentence_boundary_is_used_when_available():
    chunks = chunk_text("First sentence has enough words to break. Second sentence remains complete.", chunk_size=50, overlap=3)
    assert chunks[0].endswith(".")
