from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    STORAGE_DIR: Path = Path("/app/storage")
    CHROMA_DIR: Path = Path("/app/chroma")
    CONFIG_DIR: Path = Path("/app/config")

    OLLAMA_URL: str = "http://ollama:11434"
    CHAT_MODEL: str = "llama3.1:8b-instruct-q4_K_M"
    CHAT_MODEL_MAX_TOKENS: int = 4096
    EMBEDDING_MODEL: str = "mxbai-embed-large"
    EMBEDDING_MODEL_MAX_TOKENS: int = 1024

    TOP_K: int = 5
    HISTORY_TURNS: int = 4
    CHUNK_SIZE: int = 1200
    CHUNK_OVERLAP: int = 200

    DEFAULT_LANG: str = "uk"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
