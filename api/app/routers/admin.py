from fastapi import APIRouter
from api.app.deps import client, collection
from api.app.services.ingest_service import IngestService
from api.app.config import settings

router = APIRouter()

ingest = IngestService(
    storage_dir=settings.STORAGE_DIR,
    collection=collection,
    chunk_size=settings.CHUNK_SIZE,
    chunk_overlap=settings.CHUNK_OVERLAP,
)


@router.get("/healthz")
def healthz():
    return {"status": "ok", "heartbeat": client.heartbeat()}


@router.post("/sync-index")
def sync_index():
    return ingest.sync_index()


@router.post("/reindex-all")
def reindex_all():
    return ingest.reindex_all()
