import pytest

from app.documents.chunker import chunk_text


def test_chunking_is_deterministic() -> None:
    text = "abcdefghij" * 30

    assert chunk_text(text, 100, 20) == chunk_text(text, 100, 20)


def test_chunk_size_and_overlap_respected() -> None:
    text = "x" * 250  # step = 80 → windows at 0, 80, 160 (covers all 250 chars)

    chunks = chunk_text(text, 100, 20)

    assert [len(c) for c in chunks] == [100, 100, 90]


def test_empty_text_yields_no_chunks() -> None:
    assert chunk_text("", 100, 20) == []


def test_whitespace_only_text_yields_no_chunks() -> None:
    assert chunk_text("   \n  ", 100, 20) == []


def test_invalid_params_raise() -> None:
    with pytest.raises(ValueError):
        chunk_text("abc", 0, 0)
    with pytest.raises(ValueError):
        chunk_text("abc", 100, 100)
    with pytest.raises(ValueError):
        chunk_text("abc", 100, -1)
