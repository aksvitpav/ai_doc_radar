import json
import sqlite3
from typing import List, Dict


class HistoryRepo:

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS history(
                user_id TEXT,
                ts INTEGER,
                role TEXT,
                content TEXT
            );
            """
        )
        self._ensure_column("embedding_model", "TEXT")
        self._ensure_column("embedding", "TEXT")

    def _ensure_column(self, column_name: str, column_type: str):
        cur = self.conn.execute("PRAGMA table_info(history)")
        columns = [row[1] for row in cur.fetchall()]
        if column_name not in columns:
            self.conn.execute(f"ALTER TABLE history ADD COLUMN {column_name} {column_type};")
            self.conn.commit()

    def append(self, user_id: str, role: str, content: str, ts: int, embedding_model: str = None,
               embedding: List[float] = None):
        self.conn.execute(
            "INSERT INTO history (user_id, ts, role, content, embedding_model, embedding) VALUES (?, ?, ?, ?, ?, ?)",
            (
                user_id,
                ts,
                role,
                content,
                embedding_model,
                json.dumps(embedding) if embedding else None
            ),
        )
        self.conn.commit()

    def recall(self, user_id: str, turns: int) -> List[Dict[str, str]]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT role, content, embedding_model, embedding FROM history WHERE user_id=? ORDER BY ts DESC LIMIT ?",
            (user_id, turns * 2),
        )
        rows = cur.fetchall()[::-1]
        result = []
        for role, content, model, emb in rows:
            item = {"role": role, "content": content}
            if model:
                item["embedding_model"] = model
            if emb:
                try:
                    item["embedding"] = json.loads(emb)
                except Exception:
                    pass
            result.append(item)
        return result
