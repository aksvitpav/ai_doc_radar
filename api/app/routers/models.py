from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from api.app.deps import ollama, registry, rebuild_collection_with_embedding
from api.app.config import settings
from api.app.services.ingest_service import IngestService

router = APIRouter()


class PullRequest(BaseModel):
    name: str  # e.g., "llama3.1:8b-q4_K_M" or "mxbai-embed-large"


class SelectChatRequest(BaseModel):
    model: str


class SelectEmbeddingRequest(BaseModel):
    model: str
    reindex: bool = True


@router.get("/models/installed")
def list_installed_models():
    res = ollama.list()
    return {
        "installed": res.get("models", []),
        "current": {
            "chat": registry.get_chat_model(),
            "embedding": registry.get_embedding_model(),
        },
    }


@router.post("/models/pull")
def pull_model(req: PullRequest):
    try:
        for _ in ollama.pull(model=req.name, stream=True):
            pass
        return {"pulled": req.name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pull failed: {e}")


@router.get("/models/current")
def current_models():
    return {
        "chat": registry.get_chat_model(),
        "embedding": registry.get_embedding_model(),
    }


@router.post("/models/select/chat")
def select_chat_model(req: SelectChatRequest):
    registry.set_chat_model(req.model)
    return {"chat": registry.get_chat_model()}


@router.post("/models/select/embedding")
def select_embedding_model(req: SelectEmbeddingRequest):
    registry.set_embedding_model(req.model)
    new_collection = rebuild_collection_with_embedding(req.model)

    out = {"embedding": req.model, "reindexed": False, "indexed": []}
    if req.reindex:
        ingest = IngestService(
            storage_dir=settings.STORAGE_DIR,
            collection=new_collection,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        out = ingest.reindex_all()
        out["embedding"] = req.model
        out["reindexed"] = True

    return out
