from pathlib import Path
from typing import Dict, Any, Set
from fastapi import HTTPException, UploadFile

from api.app.utils.hashing import sha256_file
from api.app.utils.extract import extract_text_from_file
from api.app.utils.chunk import chunk_text, sentence_chunk_text


class IngestService:
    def __init__(self, storage_dir: Path, collection, chunk_size: int, chunk_overlap: int):
        self.storage_dir = storage_dir
        self.collection = collection
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def upsert_file(self, path: Path) -> Dict[str, Any]:
        file_hash = sha256_file(path)
        mtime = int(path.stat().st_mtime)

        existing = self.collection.get(
            where={"file_path": str(path)}, include=["metadatas"], limit=1_000_000
        )
        existing_hashes = {m.get("file_hash") for m in existing.get("metadatas", [])}
        if existing_hashes and (file_hash in existing_hashes):
            return {"indexed": False, "reason": "no_change"}

        if existing.get("metadatas"):
            self.collection.delete(where={"file_path": str(path)})

        text = extract_text_from_file(path)
        chunks = sentence_chunk_text(text, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

        ids = [f"{path.name}:{file_hash}:{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "file_path": str(path),
                "file_name": path.name,
                "file_hash": file_hash,
                "file_mtime": mtime,
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]
        self.collection.add(ids=ids, documents=chunks, metadatas=metadatas)
        return {"indexed": True, "chunks": len(chunks)}

    def _list_indexed_files(self) -> Set[str]:
        data = self.collection.get(include=["metadatas"], limit=1_000_000)
        files = set()
        for m in data.get("metadatas", []):
            fp = m.get("file_path")
            if fp:
                files.add(fp)
        return files

    def sync_index(self):
        indexed_files = self._list_indexed_files()
        deleted = []
        for f in indexed_files:
            p = Path(f)
            if not p.exists():
                self.collection.delete(where={"file_path": f})
                deleted.append(f)

        changed = []
        for path in self.storage_dir.iterdir():
            if path.is_file() and path.suffix.lower() in {".txt", ".pdf", ".docx", ".doc"}:
                existing = self.collection.get(
                    where={"file_path": str(path)}, include=["metadatas"], limit=1_000_000
                )
                existing_hashes = {m.get("file_hash") for m in existing.get("metadatas", [])}
                cur_hash = sha256_file(path)
                if existing_hashes and (cur_hash in existing_hashes):
                    continue
                self.upsert_file(path)
                changed.append(str(path))

        return {"deleted_from_index": deleted, "reindexed": changed}

    def reindex_all(self):
        indexed = []
        for path in sorted(self.storage_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in {".txt", ".pdf", ".docx", ".doc"}:
                r = self.upsert_file(path)
                indexed.append({"file": path.name, **r})
        return {"ok": True, "indexed": indexed}

    def delete_file_and_index(self, file_name: str):
        safe = Path(file_name).name
        target = self.storage_dir / safe
        self.collection.delete(where={"file_name": safe})
        if target.exists():
            try:
                target.unlink()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to remove file: {e}")
        return {"deleted": safe}

    def update_file_from_upload(self, file_name: str, upload: UploadFile):
        safe = Path(file_name).name
        if Path(upload.filename).name != safe:
            raise HTTPException(status_code=400, detail="filename mismatch with route")

        dest = self.storage_dir / safe
        with dest.open("wb") as out:
            while True:
                chunk = upload.file.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)

        r = self.upsert_file(dest)
        return {"updated": safe, **r}
