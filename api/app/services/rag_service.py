import time
from typing import Dict, List, Iterator
from api.app.utils.logger import setup_logger
from api.app.repositories.history_repo import HistoryRepo
from api.app.services.model_registry import ModelRegistry


def system_prompt(lang: str) -> str:
    if lang.lower().startswith("uk"):
        return (
            "Ти — корисний помічник для пошуку по документах українською. "
            "Відповідай ЛИШЕ на основі наданого контексту. "
            "Якщо відповіді немає у контексті — скажи, що не знаєш. "
            "Наприкінці за можливості наведи назви файлів (цитації)."
        )
    return (
        "You are a helpful assistant for document QA. "
        "Answer ONLY using the provided context; if not present, say you don't know. "
        "Cite filenames when applicable."
    )


class RagService:
    def __init__(self, collection, ollama, sqlite_conn, top_k: int,
                 history_turns: int, default_lang: str, model_registry: ModelRegistry):
        self.collection = collection
        self.ollama = ollama
        self.history = HistoryRepo(sqlite_conn)
        self.top_k = top_k
        self.history_turns = history_turns
        self.default_lang = default_lang
        self.registry = model_registry
        self.logger = setup_logger()

    def _count_tokens(self, text: str) -> int:
        return len(text) // 4

    def _build_messages(self, user_id: str, query: str, ctx_blocks: List[str], lang: str):
        msgs = [{"role": "system", "content": system_prompt(lang)}]
        msgs.extend(self.history.recall(user_id, self.history_turns))

        if lang.lower().startswith("uk"):
            user_prompt = (
                f"Питання: {query}\n"
                f"Контекст: {' --- '.join(ctx_blocks)}\n\n"
                "Відповідай лише на основі контексту. "
                "Надай повну, структуровану відповідь. "
                "Якщо відповідь містить перелік — наведи всі пункти списком. "
                "Не вигадуй нічого поза контекстом. "
                "Не зупиняйся раніше, ніж наведеш усі релевантні факти з контексту. "
                "Наприкінці за можливості наведи назви файлів (цитації)."
            )
        else:
            user_prompt = (
                f"Question: {query}\n"
                f"Context: {' --- '.join(ctx_blocks)}\n\n"
                "Answer only using the context. "
                "Provide a complete and structured response. "
                "If the answer involves a list — include all items. "
                "Do not make anything up beyond the context. "
                "Do not stop until all relevant facts from the context are covered. "
                "Cite filenames when applicable."
            )

        msgs.append({"role": "user", "content": user_prompt})
        return msgs

    def _truncate_ctx_blocks(self, ctx_blocks: List[str], max_tokens: int = None) -> List[str]:
        max_tokens = max_tokens or self.registry.get_chat_model_max_tokens()
        truncated = []
        total_tokens = 0
        for block in ctx_blocks:
            block_tokens = self._count_tokens(block)
            if total_tokens + block_tokens > max_tokens:
                break
            truncated.append(block)
            total_tokens += block_tokens
        return truncated

    def _prepare_messages(self, user_id: str, query: str, top_k: int, lang: str):
        res = self.collection.query(query_texts=[query], n_results=top_k)
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]

        ctx_blocks, citations = [], []
        for d, m in zip(docs, metas):
            if d:
                ctx_blocks.append(d)
                citations.append({
                    "file": m.get("file_name"),
                    "path": m.get("file_path"),
                    "chunk": m.get("chunk_index")
                })

        history = self.history.recall(user_id, self.history_turns)
        unique_history = []
        seen = set()
        for msg in history:
            key = (msg["role"], msg["content"])
            if key not in seen:
                unique_history.append(msg)
                seen.add(key)

        history_token_count = sum(self._count_tokens(m["content"]) for m in unique_history)
        available_tokens = self.registry.get_chat_model_max_tokens() - history_token_count - 500
        truncated_ctx = self._truncate_ctx_blocks(ctx_blocks, max_tokens=available_tokens)

        messages = [{"role": "system", "content": system_prompt(lang)}]
        messages.append(self._build_messages(user_id, query, truncated_ctx, lang or self.default_lang)[-1])

        return messages, citations

    def _save_history(self, user_id: str, query: str, answer: str):
        now = int(time.time())
        self.history.append(user_id, "user", query, now)
        self.history.append(user_id, "assistant", answer, now)

    def answer(self, user_id: str, query: str, top_k: int, lang: str) -> Dict:
        messages, citations = self._prepare_messages(user_id, query, top_k, lang)
        chat_model = self.registry.get_chat_model()

        start = time.time()
        out = self.ollama.chat(
            model=chat_model,
            messages=messages,
            options={"temperature": 0.2},
            keep_alive="15m"
        )
        duration = time.time() - start
        answer = out["message"]["content"]

        self._save_history(user_id, query, answer)
        self.logger.info("LLM response time: %.2f seconds", duration)
        return {"answer": answer, "citations": citations}

    def stream_answer(self, user_id: str, query: str, top_k: int, lang: str) -> Iterator[Dict]:
        messages, citations = self._prepare_messages(user_id, query, top_k, lang)
        chat_model = self.registry.get_chat_model()
        buffer = ""
        chunk_size = 10

        for chunk in self.ollama.chat(
                model=chat_model,
                messages=messages,
                stream=True,
                options={"temperature": 0.2},
                keep_alive="15m"
        ):
            text = chunk.get("message", {}).get("content", "")
            if text:
                buffer += text
                while len(buffer) >= chunk_size:
                    yield {"type": "partial", "content": buffer[:chunk_size]}
                    buffer = buffer[chunk_size:]

        if buffer:
            yield {"type": "partial", "content": buffer}

        final_text = buffer
        self._save_history(user_id, query, final_text)
        yield {"type": "final", "content": final_text, "citations": citations}
