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

    def append(self, user_id: str, role: str, content: str, ts: int):
        self.conn.execute(
            "INSERT INTO history (user_id, ts, role, content) VALUES (?, ?, ?, ?)",
            (user_id, ts, role, content),
        )
        self.conn.commit()

    def recall(self, user_id: str, turns: int) -> List[Dict[str, str]]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT role, content FROM history WHERE user_id=? ORDER BY ts DESC LIMIT ?",
            (user_id, turns * 2),
        )
        rows = cur.fetchall()[::-1]
        return [{"role": r, "content": c} for (r, c) in rows]
