from pathlib import Path
from fastapi import HTTPException


def extract_text_from_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    if ext == ".pdf":
        from pypdf import PdfReader
        text = []
        with path.open("rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text.append(page.extract_text() or "")
        return "".join(text)
    if ext == ".docx":
        from docx import Document
        doc = Document(str(path))
        return "".join(p.text for p in doc.paragraphs)
    if ext == ".doc":
        import subprocess
        try:
            out = subprocess.check_output(["antiword", str(path)])
            return out.decode("utf-8", errors="ignore")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"antiword failed: {e}")
    raise HTTPException(status_code=415, detail=f"Unsupported file type: {ext}")
