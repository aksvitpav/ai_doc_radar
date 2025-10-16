from pathlib import Path
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile

from api.app.config import settings
from api.app.deps import ingest

router = APIRouter()


@router.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    results = []
    for uf in files:
        ext = Path(uf.filename).suffix.lower()
        if ext not in {".txt", ".pdf", ".docx", ".doc"}:
            raise HTTPException(status_code=415, detail=f"Unsupported: {uf.filename}")

        dest = settings.STORAGE_DIR / uf.filename
        with dest.open("wb") as out:
            while True:
                chunk = await uf.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)

        r = ingest.upsert_file(dest)
        results.append({"file": uf.filename, **r})

    return {"ok": True, "results": results}
