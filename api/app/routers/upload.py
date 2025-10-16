from pathlib import Path
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException
from api.app.services.ingest_service import IngestService
from api.app.config import settings
from api.app.deps import collection

router = APIRouter()

ingest = IngestService(
    storage_dir=settings.STORAGE_DIR,
    collection=collection,
    chunk_size=settings.CHUNK_SIZE,
    chunk_overlap=settings.CHUNK_OVERLAP,
)


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
