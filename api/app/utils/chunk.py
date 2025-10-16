def chunk_text(text: str, size: int, overlap: int):
    if not text:
        return []
    chunks = []
    i = 0
    n = len(text)
    step = max(1, size - overlap)
    while i < n:
        end = min(i + size, n)
        chunks.append(text[i:end])
        i += step
    return chunks
