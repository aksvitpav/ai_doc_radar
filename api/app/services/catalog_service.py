from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from api.app.deps import registry


class CatalogService:
    def __init__(self, collection, storage_dir: Path, ollama=None, default_lang: str = "uk"):
        self.collection = collection
        self.storage_dir = storage_dir
        self.ollama = ollama
        self.default_lang = default_lang

    def _excerpt_from_docs(self, docs: List[str], max_chars: int = 400) -> str:
        buf, total = [], 0
        for d in docs:
            if not d:
                continue
            left = max_chars - total
            if left <= 0:
                break
            snip = d.replace("", " ").strip()[:left]
            buf.append(snip)
            total += len(snip)
        return " ".join(buf).strip()

    def _summarize(self, text: str, lang: Optional[str]) -> str:
        if not self.ollama:
            return text
        prompt = (
            "Стисло (1–2 речення) опиши зміст фрагмента українською:"
            if (lang or self.default_lang).lower().startswith("uk")
            else "Summarize in 1–2 sentences (brief, informative):"
        )
        res = self.ollama.chat(
            model=registry.get_chat_model(),
            messages=[{"role": "user", "content": prompt + text[:1200]}],
            options={"temperature": 0.2},
        )
        return res["message"]["content"].strip()

    def list_files(self, limit: int = 200, summarize: bool = False, lang: Optional[str] = None) -> List[Dict]:
        data = self.collection.get(include=["metadatas", "documents"], limit=1_000_000)
        metas = data.get("metadatas", [])
        docs = data.get("documents", [])

        group = defaultdict(lambda: {"docs": {}, "count": 0, "mtime": 0, "name": None, "path": None})
        for m, d in zip(metas, docs):
            fp = m.get("file_path")
            if not fp:
                continue
            gi = group[fp]
            gi["count"] += 1
            ci = m.get("chunk_index", 0)
            if ci not in gi["docs"]:
                gi["docs"][ci] = d
            mmt = m.get("file_mtime") or 0
            if mmt > gi["mtime"]:
                gi["mtime"] = mmt
            gi["name"] = m.get("file_name") or Path(fp).name
            gi["path"] = fp

        items = []
        for fp, g in group.items():
            first_docs = [g["docs"][i] for i in sorted(g["docs"].keys())[:3]]
            excerpt = self._excerpt_from_docs(first_docs, max_chars=400)
            size = None
            p = Path(fp)
            if p.exists():
                try:
                    size = p.stat().st_size
                except Exception:
                    size = None

            desc = self._summarize(excerpt, lang) if summarize and excerpt else excerpt

            items.append({
                "file_name": g["name"],
                "file_path": fp,
                "size_bytes": size,
                "chunk_count": g["count"],
                "mtime": g["mtime"],
                "description": desc,
            })

        items.sort(key=lambda x: x["mtime"], reverse=True)
        return items[:max(1, limit)]

    def get_file(self, filename: str, summarize: bool = False, lang: Optional[str] = None) -> Optional[Dict]:
        from os.path import basename
        name = basename(filename)
        where = {"file_name": name}

        data = self.collection.get(include=["metadatas", "documents"], where=where, limit=1_000_000)
        metas = data.get("metadatas", [])
        docs = data.get("documents", [])
        if not metas:
            return None

        by_idx = {}
        mtime, path_ = 0, None
        for m, d in zip(metas, docs):
            ci = m.get("chunk_index", 0)
            if ci not in by_idx:
                by_idx[ci] = d
            mmt = m.get("file_mtime") or 0
            if mmt > mtime:
                mtime = mmt
            path_ = m.get("file_path", path_)

        first_docs = [by_idx[i] for i in sorted(by_idx.keys())[:4]]
        excerpt = self._excerpt_from_docs(first_docs, max_chars=800)
        p = Path(path_) if path_ else None
        size = p.stat().st_size if p and p.exists() else None
        desc = self._summarize(excerpt, lang) if summarize and excerpt else excerpt

        return {
            "file_name": name,
            "file_path": path_,
            "size_bytes": size,
            "chunk_count": len(by_idx),
            "mtime": mtime,
            "description": desc,
        }
