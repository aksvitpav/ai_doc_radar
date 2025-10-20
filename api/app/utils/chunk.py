from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List


def chunk_text(text: str, chunk_size: int, chunk_overlap: int):
    if not text:
        return []
    chunks = []
    i = 0
    n = len(text)
    step = max(1, chunk_size - chunk_overlap)
    while i < n:
        end = min(i + chunk_size, n)
        chunks.append(text[i:end])
        i += step
    return chunks


def sentence_chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    if not text:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )

    return splitter.split_text(text)
