import json
from pathlib import Path
from threading import RLock


class ModelRegistry:
    """
    Persistent registry for current chat/embedding model names.
    Backed by a JSON file under CONFIG_DIR, survives restarts.
    """

    def __init__(self, config_path: Path, default_chat_model: str, default_chat_model_max_tokens: int,
                 default_embedding_model: str,
                 default_embedding_model_max_tokens: int):
        self.path = config_path
        self._lock = RLock()

        self._state = {
            "chat_model": default_chat_model,
            "chat_model_max_tokens": default_chat_model_max_tokens,
            "embedding_model": default_embedding_model,
            "embedding_model_max_tokens": default_embedding_model_max_tokens,
        }
        self._load_or_init()

    def _load_or_init(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                for k in ("chat_model", "chat_model_max_tokens", "embedding_model", "embedding_model_max_tokens"):
                    if k in data:
                        self._state[k] = data[k]
            except Exception:
                pass
        self._persist()

    def _persist(self):
        with self._lock:
            self.path.write_text(json.dumps(self._state, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_chat_model(self) -> str:
        with self._lock:
            return self._state["chat_model"]

    def get_embedding_model(self) -> str:
        with self._lock:
            return self._state["embedding_model"]

    def get_chat_model_max_tokens(self) -> int:
        with self._lock:
            return self._state.get("chat_model_max_tokens")

    def get_embedding_model_max_tokens(self) -> int:
        with self._lock:
            return self._state.get("embedding_model_max_tokens")

    def set_chat_model(self, name: str, max_tokens: int = None):
        with self._lock:
            self._state["chat_model"] = name
            if max_tokens is not None:
                self._state["chat_model_max_tokens"] = max_tokens
            self._persist()

    def set_embedding_model(self, name: str, max_tokens: int = None):
        with self._lock:
            self._state["embedding_model"] = name
            if max_tokens is not None:
                self._state["embedding_model_max_tokens"] = max_tokens
            self._persist()
