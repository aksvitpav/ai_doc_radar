import time
import threading
from typing import Dict, List, Iterator
from sklearn.metrics.pairwise import cosine_similarity

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

    def _build_user_prompt(self, query: str, ctx_blocks: List[str], lang: str) -> str:
        if lang.lower().startswith("uk"):
            return (
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
            return (
                f"Question: {query}\n"
                f"Context: {' --- '.join(ctx_blocks)}\n\n"
                "Answer only using the context. "
                "Provide a complete and structured response. "
                "If the answer involves a list — include all items. "
                "Do not make anything up beyond the context. "
                "Do not stop until all relevant facts from the context are covered. "
                "Cite filenames when applicable."
            )

    @staticmethod
    def filter_relevant_history(query_embedding: List[float], history: List[Dict], threshold: float = 0.8,
                                current_model: str = "") -> List[Dict]:
        relevant = []
        for msg in history:
            emb = msg.get("embedding")
            model = msg.get("embedding_model")
            if emb and model == current_model and msg["role"] == "user":
                sim = cosine_similarity([query_embedding], [emb])[0][0]
                if sim >= threshold:
                    relevant.append(msg)
            elif msg["role"] == "assistant":
                relevant.append(msg)
        return relevant

    def _build_messages(self, user_id: str, query: str, ctx_blocks: List[str], lang: str,
                        query_embedding: List[float], embedding_model: str) -> List[Dict]:
        messages = [{"role": "system", "content": system_prompt(lang)}]
        raw_history = self.history.recall(user_id, self.history_turns)
        filtered_history = self.filter_relevant_history(query_embedding=query_embedding, history=raw_history,
                                                        current_model=embedding_model)
        messages.extend(filtered_history)
        user_prompt = self._build_user_prompt(query, ctx_blocks, lang)
        messages.append({"role": "user", "content": user_prompt})
        self.logger.info("Prompt is generated: %s", user_prompt)
        return messages

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
        distances = res.get("distances", [[]])[0]

        self.logger.info("Relevance scores (distance): %s", distances)

        ctx_blocks, citations = [], []
        for i in range(min(top_k, len(docs))):
            doc = docs[i]
            meta = metas[i]
            score = distances[i]

            ctx_blocks.append(doc)
            citations.append({
                "file": meta.get("file_name"),
                "path": meta.get("file_path"),
                "chunk": meta.get("chunk_index")
            })
            self.logger.info("Chunk selected: %.4f | %s", score, doc[:100].replace("\n", " "))

        embedding_model = self.registry.get_embedding_model()
        query_embedding = self.ollama.embeddings(model=embedding_model, prompt=query)["embedding"]

        history = self.history.recall(user_id, self.history_turns)
        history_token_count = sum(self._count_tokens(m["content"]) for m in history)
        available_tokens = self.registry.get_chat_model_max_tokens() - history_token_count - 500
        truncated_ctx = self._truncate_ctx_blocks(ctx_blocks, max_tokens=available_tokens)

        messages = self._build_messages(user_id, query, truncated_ctx, lang or self.default_lang,
                                        query_embedding, embedding_model)
        return messages, citations

    def _save_history_async(self, user_id: str, query: str, answer: str):
        def task():
            try:
                now = int(time.time())
                embedding_model = self.registry.get_embedding_model()
                embedding = self.ollama.embeddings(model=embedding_model, prompt=query)["embedding"]
                self.history.append(user_id, "user", query, now, embedding_model=embedding_model, embedding=embedding)
                self.history.append(user_id, "assistant", answer, now)
            except Exception as e:
                self.logger.warning("Error saving history: %s", str(e))

        threading.Thread(target=task, daemon=True).start()

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

        self._save_history_async(user_id, query, answer)
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
        self._save_history_async(user_id, query, final_text)
        yield {"type": "final", "content": final_text, "citations": citations}
