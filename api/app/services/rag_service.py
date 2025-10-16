import time
from typing import Dict, List
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
    def __init__(self, collection, ollama, sqlite_conn, top_k: int, max_tokens: int, history_turns: int, default_lang: str,
                 model_registry: ModelRegistry):
        self.collection = collection
        self.ollama = ollama
        self.history = HistoryRepo(sqlite_conn)
        self.top_k = top_k
        self.max_context_tokens = max_tokens
        self.history_turns = history_turns
        self.default_lang = default_lang
        self.registry = model_registry
        self.logger = setup_logger()

    def _count_tokens(self, text: str) -> int:
        # Легка оцінка: 1 токен ≈ 4 символи
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
        max_tokens = max_tokens or self.max_context_tokens
        truncated = []
        total_tokens = 0

        for block in ctx_blocks:
            block_tokens = self._count_tokens(block)
            if total_tokens + block_tokens > max_tokens:
                break
            truncated.append(block)
            total_tokens += block_tokens

        return truncated

    def answer(self, user_id: str, query: str, top_k: int, lang: str) -> Dict:
        # 1. Отримати релевантні документи
        res = self.collection.query(query_texts=[query], n_results=top_k)
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]

        # 2. Побудувати контекст і цитати
        ctx_blocks, citations = [], []
        for d, m in zip(docs, metas):
            if not d:
                continue
            ctx_blocks.append(d)
            citations.append({
                "file": m.get("file_name"),
                "path": m.get("file_path"),
                "chunk": m.get("chunk_index")
            })

        # 3. Історія без дублікатів
        history = self.history.recall(user_id, self.history_turns)
        unique_history = []
        seen = set()
        for msg in history:
            key = (msg["role"], msg["content"])
            if key not in seen:
                unique_history.append(msg)
                seen.add(key)

        # 4. Оцінка токенів історії
        history_token_count = sum(self._count_tokens(m["content"]) for m in unique_history)
        available_tokens = self.max_context_tokens - history_token_count - 500  # запас для відповіді

        # 5. Обрізання контексту
        truncated_ctx = self._truncate_ctx_blocks(ctx_blocks, max_tokens=available_tokens)
        self.logger.info("Truncated context blocks: %s", truncated_ctx)

        # 6. Побудова prompt'ів
        messages = [{"role": "system", "content": system_prompt(lang)}]
        messages.append(self._build_messages(user_id, query, truncated_ctx, lang or self.default_lang)[-1])

        self.logger.info("Messages: %s", messages)
        self.logger.info("Estimated token count: %d", sum(self._count_tokens(m["content"]) for m in messages))

        # 7. Виклик моделі
        chat_model = self.registry.get_chat_model()
        start = time.time()
        out = self.ollama.chat(
            model=chat_model,
            messages=messages,
            options={"temperature": 0.2},
        )
        duration = time.time() - start
        answer = out["message"]["content"]

        self.logger.info("LLM response: %s", answer)
        self.logger.info("LLM response time: %.2f seconds", duration)

        # 8. Зберегти історію
        now = int(time.time())
        self.history.append(user_id, "user", query, now)
        self.history.append(user_id, "assistant", answer, now)

        return {"answer": answer, "citations": citations}