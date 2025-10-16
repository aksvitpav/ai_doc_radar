import sqlite3
from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from api.app.config import settings
from api.app.services.model_registry import ModelRegistry
from ollama import Client as OllamaClient

CONFIG_PATH = Path(settings.CONFIG_DIR) / "runtime_config.json"
registry = ModelRegistry(
    config_path=CONFIG_PATH,
    default_chat=settings.CHAT_MODEL,
    default_embed=settings.EMBEDDING_MODEL,
)

client = chromadb.PersistentClient(path=str(settings.CHROMA_DIR))


def _make_collection(embed_model: str):
    ef = OllamaEmbeddingFunction(
        url=settings.OLLAMA_URL,
        model_name=embed_model,
    )
    try:
        col = client.get_or_create_collection(name="documents", embedding_function=ef)
    except Exception:
        col = client.create_collection(name="documents", embedding_function=ef)
    return col


collection = _make_collection(registry.get_embedding_model())

ollama = OllamaClient(host=settings.OLLAMA_URL)


def rebuild_collection_with_embedding(embed_model: str):
    try:
        client.delete_collection("documents")
    except Exception:
        pass
    return _make_collection(embed_model)


def get_sqlite_conn() -> sqlite3.Connection:
    conn = sqlite3.connect("/app/chat_history.db", check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS history (
            user_id TEXT,
            ts INTEGER,
            role TEXT,
            content TEXT
        );
        """
    )
    return conn
