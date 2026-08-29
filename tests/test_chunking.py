import pytest

from ragsource.chunking import chunk_text, normalize_text


def test_normalize_text_removes_noise():
    assert normalize_text("hello   world\n\n\nnext") == "hello world\n\nnext"


def test_chunk_text_preserves_overlap():
    chunks = chunk_text("One sentence. " * 30, size=100, overlap=20)
    assert len(chunks) > 1
    assert all(chunks)


def test_chunk_text_rejects_invalid_settings():
    with pytest.raises(ValueError):
        chunk_text("text", size=10, overlap=10)
