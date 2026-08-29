import re


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    """Split text into overlapping chunks while preferring sentence boundaries."""
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Require size > 0 and 0 <= overlap < size")
    clean = normalize_text(text)
    if not clean:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(start + size, len(clean))
        if end < len(clean):
            boundary = max(clean.rfind(". ", start, end), clean.rfind("\n", start, end))
            if boundary > start + size // 2:
                end = boundary + 1
        part = clean[start:end].strip()
        if part:
            chunks.append(part)
        if end == len(clean):
            break
        start = max(end - overlap, start + 1)
    return chunks
