import re
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from api.app import deps
from api.app.config import settings
from api.app.services.ingest_service import IngestService
from api.app.utils.logger import setup_logger

logger = setup_logger()
router = APIRouter()


class PullRequest(BaseModel):
    name: str


class SelectChatRequest(BaseModel):
    model: str
    max_tokens: int | None = None


class SelectEmbeddingRequest(BaseModel):
    model: str
    max_tokens: int | None = None
    reindex: bool = True
    force_reindex: bool = False


def get_installed_model_names() -> list[str]:
    try:
        models = deps.ollama.list().get("models", [])
        return [m.model for m in models if hasattr(m, "model")]
    except Exception as e:
        logger.error(f"Failed to list installed models: {e}")
        return []


def extract_context_length(info: any, default: int) -> int:
    try:
        if hasattr(info, "parameters") and isinstance(info.parameters, dict):
            return info.parameters.get("context_length") or info.parameters.get("num_ctx") or default
        elif isinstance(info, dict):
            return info.get("context_length") or info.get("num_ctx") or default
        elif isinstance(info, str):
            match = re.search(r"(?:context_length|num_ctx):\s*(\d+)", info)
            if match:
                return int(match.group(1))
        return default
    except Exception as e:
        logger.warning(f"Failed to extract context_length: {e}")
        return default


@router.get("/models/installed")
def list_installed_models():
    res = deps.ollama.list()
    return {
        "installed": res.get("models", []),
        "current": {
            "chat_model": deps.registry.get_chat_model(),
            "chat_model_max_tokens": deps.registry.get_chat_model_max_tokens(),
            "embedding_model": deps.registry.get_embedding_model(),
            "embedding_model_max_tokens": deps.registry.get_embedding_model_max_tokens(),
        },
    }


@router.post("/models/pull")
def pull_model(req: PullRequest):
    try:
        for _ in deps.ollama.pull(model=req.name, stream=True):
            pass
        return {"pulled": req.name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pull failed: {e}")


@router.get("/models/current")
def current_models():
    return {
        "chat_model": deps.registry.get_chat_model(),
        "chat_model_max_tokens": deps.registry.get_chat_model_max_tokens(),
        "embedding_model": deps.registry.get_embedding_model(),
        "embedding_model_max_tokens": deps.registry.get_embedding_model_max_tokens(),
    }


@router.post("/models/select/chat")
def select_chat_model(req: SelectChatRequest):
    if req.model not in get_installed_model_names():
        raise HTTPException(status_code=400, detail=f"Model '{req.model}' is not installed. Please pull it first.")

    max_tokens = req.max_tokens
    if max_tokens is None:
        try:
            info = deps.ollama.show(req.model)
            max_tokens = extract_context_length(info, default=4096)
            logger.info(f"Model {req.model}: context_length = {max_tokens}")
        except Exception as e:
            max_tokens = 4096
            logger.error(f"Failed to get model info: {e}")

    if max_tokens < 512 or max_tokens > 131072:
        raise HTTPException(status_code=400, detail="Invalid max_tokens value")

    deps.registry.set_chat_model(req.model, max_tokens=max_tokens)
    deps.rag.max_context_tokens = max_tokens

    return {
        "chat_model": deps.registry.get_chat_model(),
        "chat_model_max_tokens": deps.registry.get_chat_model_max_tokens(),
    }


@router.post("/models/select/embedding")
def select_embedding_model(req: SelectEmbeddingRequest):
    if req.model not in get_installed_model_names():
        raise HTTPException(status_code=400,
                            detail=f"Embedding model '{req.model}' is not installed. Please pull it first.")

    max_tokens = req.max_tokens
    if max_tokens is None:
        try:
            info = deps.ollama.show(req.model)
            max_tokens = extract_context_length(info, default=1024)
            logger.info(f"Embedding model {req.model}: context_length = {max_tokens}")
        except Exception as e:
            max_tokens = 1024
            logger.error(f"Failed to get embedding model info: {e}")

    if max_tokens < 512 or max_tokens > 131072:
        raise HTTPException(status_code=400, detail="Invalid max_tokens value")

    deps.registry.set_embedding_model(req.model, max_tokens=max_tokens)

    new_collection = deps.rebuild_collection_with_embedding(req.model)

    deps.collection = new_collection
    deps.ingest = IngestService(
        storage_dir=settings.STORAGE_DIR,
        collection=new_collection,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    deps.rag.collection = new_collection
    out = {
        "embedding_model": deps.registry.get_embedding_model(),
        "embedding_model_max_tokens": deps.registry.get_embedding_model_max_tokens(),
        "reindexed": False,
        "indexed": []
    }

    if req.reindex:
        out = deps.ingest.reindex_all(force=req.force_reindex)
        out["embedding_model"] = deps.registry.get_embedding_model()
        out["embedding_model_max_tokens"] = deps.registry.get_embedding_model_max_tokens()
        out["reindexed"] = True

    return out
