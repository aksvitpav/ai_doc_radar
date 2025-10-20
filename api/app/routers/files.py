import os
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from starlette.responses import FileResponse

from api.app.config import settings
from api.app.deps import collection, ingest, ollama
from api.app.services.catalog_service import CatalogService

router = APIRouter()

catalog = CatalogService(
    collection=collection,
    storage_dir=settings.STORAGE_DIR,
    ollama=ollama,
    default_lang=settings.DEFAULT_LANG,
)


@router.get("/files")
def list_files(
        limit: int = Query(200, ge=1, le=5000),
        summarize: bool = Query(False),
        lang: Optional[str] = Query(None),
):
    return {"files": catalog.list_files(limit=limit, summarize=summarize, lang=lang)}


@router.get("/files/{filename}")
def get_file(filename: str, summarize: bool = False, lang: Optional[str] = None):
    item = catalog.get_file(filename, summarize=summarize, lang=lang)
    if not item:
        raise HTTPException(status_code=404, detail="File not found in index")
    return item


@router.put("/files/{filename}")
async def update_file(filename: str, file: UploadFile = File(...)):
    return ingest.update_file_from_upload(filename, file)


@router.delete("/files/{filename}")
def delete_file(filename: str):
    return ingest.delete_file_and_index(filename)


@router.get("/files/download/{file_name}")
def download_file(file_name: str):
    file_path = os.path.join(settings.STORAGE_DIR, file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, filename=file_name, media_type='application/octet-stream')
