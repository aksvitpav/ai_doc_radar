from fastapi import APIRouter
from api.app.schemas.chat import ChatRequest, ChatResponse
from api.app.services.rag_service import RagService
from api.app.deps import collection, ollama, get_sqlite_conn, registry
from api.app.config import settings

router = APIRouter()

rag = RagService(
    collection=collection,
    ollama=ollama,
    sqlite_conn=get_sqlite_conn(),
    top_k=settings.TOP_K,
    max_tokens=settings.CHAT_MODEL_MAX_CONTEXT_TOKENS,
    history_turns=settings.HISTORY_TURNS,
    default_lang=settings.DEFAULT_LANG,
    model_registry=registry,
)


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    return rag.answer(
        user_id=req.user_id,
        query=req.message,
        top_k=req.top_k or settings.TOP_K,
        lang=req.lang or settings.DEFAULT_LANG,
    )
