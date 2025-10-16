from typing import Optional, List, Dict
from pydantic import BaseModel

class ChatRequest(BaseModel):
    user_id: str
    message: str
    top_k: Optional[int] = None
    lang: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    citations: List[Dict]
