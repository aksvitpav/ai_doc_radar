import json

from fastapi import APIRouter
from starlette.responses import StreamingResponse

from api.app.config import settings
from api.app.deps import rag
from api.app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    return rag.answer(
        user_id=req.user_id,
        query=req.message,
        top_k=req.top_k or settings.TOP_K,
        lang=req.lang or settings.DEFAULT_LANG,
    )


@router.post("/chat/stream")
def chat_stream(req: ChatRequest, format_type: str = "json"):
    def generate_text():
        for chunk in rag.stream_answer(
                user_id=req.user_id,
                query=req.message,
                top_k=req.top_k or settings.TOP_K,
                lang=req.lang or settings.DEFAULT_LANG
        ):
            text = chunk.get("content", "")
            if format_type == "json":
                yield json.dumps(chunk) + "\n"
            else:
                yield text

    return StreamingResponse(
        generate_text(),
        media_type="text/plain" if format_type == "text" else "application/json",
        headers={"X-Accel-Buffering": "no"}
    )
