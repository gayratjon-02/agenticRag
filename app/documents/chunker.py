def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping fixed-size chunks. Deterministic.

    Strategy (fixed, so the same input always yields the same chunks):
      - Operate on raw characters.
      - Each chunk is `chunk_size` characters long.
      - Consecutive chunks overlap by `chunk_overlap` characters, i.e. the window
        advances by `chunk_size - chunk_overlap` each step.
      - Whitespace-only chunks are dropped; empty input yields no chunks.

    This is intentionally simple and reproducible; a token-aware splitter could
    replace it later without changing callers.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must not be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    step = chunk_size - chunk_overlap
    chunks: list[str] = []
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
    return chunks
