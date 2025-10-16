from fastapi import APIRouter

from api.app.deps import client, ingest

router = APIRouter()


@router.get("/healthz")
def healthz():
    return {"status": "ok", "heartbeat": client.heartbeat()}


@router.post("/sync-index")
def sync_index():
    return ingest.sync_index()


@router.post("/reindex-all")
def reindex_all():
    return ingest.reindex_all()
