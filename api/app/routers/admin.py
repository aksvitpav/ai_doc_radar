from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from api.app import deps

router = APIRouter()

class ReindexRequest(BaseModel):
    force_index: bool = False

@router.get("/healthz")
def healthz():
    return {"status": "ok", "heartbeat": deps.client.heartbeat()}

@router.post("/sync-index")
def sync_index():
    return deps.ingest.sync_index()

@router.post("/reindex-all")
def reindex_all(req: ReindexRequest):
    return deps.ingest.reindex_all(force=req.force_index)