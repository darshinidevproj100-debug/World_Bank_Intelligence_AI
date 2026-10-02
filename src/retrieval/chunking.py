"""Deterministic whitespace-normalizing text chunker with fixed overlap."""
def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> list[str]:
    """Split text into bounded character chunks while retaining overlap."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0")
    normalized_text = " ".join(text.split())
    text_chunks = []
    chunk_start = 0
    while chunk_start < len(normalized_text):
        chunk_end = min(chunk_start + chunk_size, len(normalized_text))
        text_chunks.append(normalized_text[chunk_start:chunk_end])
        if chunk_end == len(normalized_text):
            break
        chunk_start = chunk_end - overlap
    return text_chunks
