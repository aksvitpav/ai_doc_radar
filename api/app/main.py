from fastapi import FastAPI
from api.app.routers import upload, admin, chat, files, models

app = FastAPI(title="Local RAG (UA)", version="1.1.0")

app.include_router(upload.router, prefix="", tags=["upload"])
app.include_router(admin.router, prefix="", tags=["admin"])
app.include_router(chat.router, prefix="", tags=["chat"])
app.include_router(files.router, prefix="", tags=["files"])
app.include_router(models.router, prefix="", tags=["models"])
