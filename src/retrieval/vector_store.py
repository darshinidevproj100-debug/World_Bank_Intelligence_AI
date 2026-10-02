"""Small JSON-backed local vector store for development; no managed service needed."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


class LocalVectorStore:
    """Persistent cosine index storing source records and vectors in one JSON file."""
    def __init__(self, path: str | Path, *, model_id: str):
        self.path, self.model_id = Path(path), model_id
        self.records: dict[str, dict[str, Any]] = {}
        self._load()
    def _load(self):
        if self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("model_id") != self.model_id: raise ValueError("stored embedding model does not match")
            self.records = payload.get("records", {})
    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"model_id": self.model_id, "records": self.records}, ensure_ascii=False), encoding="utf-8")
    def add_documents(self, documents: list[dict[str, Any]], embeddings: list[list[float]]) -> int:
        if len(documents) != len(embeddings): raise ValueError("documents and embeddings must align")
        dimensions = {len(vector) for vector in embeddings}
        if len(dimensions) > 1: raise ValueError("embedding dimensions must match")
        existing_dimensions = {len(row["embedding"]) for row in self.records.values()}
        if dimensions and existing_dimensions and dimensions != existing_dimensions:
            raise ValueError("embedding dimensions must match the existing index")
        for doc, vector in zip(documents, embeddings):
            key = doc.get("chunk_id") or doc.get("document_id")
            if not key: raise ValueError("each record needs chunk_id or document_id")
            norm = math.sqrt(sum(float(x)**2 for x in vector))
            if not norm: raise ValueError("embedding vector must be non-zero")
            self.records[str(key)] = {"document": doc, "embedding": [float(x)/norm for x in vector]}
        self._save(); return len(documents)
    def search(self, query_embedding: list[float], *, top_k: int = 5, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if top_k < 1: raise ValueError("top_k must be positive")
        qnorm = math.sqrt(sum(float(x)**2 for x in query_embedding))
        if not qnorm: raise ValueError("query embedding vector must be non-zero")
        q = [float(x)/qnorm for x in query_embedding]; found = []
        for key, row in self.records.items():
            doc = row["document"]; meta = doc.get("metadata", {})
            if any(meta.get(k, doc.get(k)) != v for k,v in (filters or {}).items()): continue
            if len(row["embedding"]) != len(q): raise ValueError("query/index embedding dimensions do not match")
            score = sum(a*b for a,b in zip(q, row["embedding"]))
            item = dict(doc); item.update(score=score, retrieval_method="semantic", component_scores={"semantic": score}, embedding_model=self.model_id)
            found.append((score,key,item))
        found.sort(key=lambda x:(-x[0],x[1])); return [x[2] for x in found[:top_k]]
    def delete(self, ids: list[str]) -> int:
        count = sum(self.records.pop(str(key), None) is not None for key in ids)
        self._save(); return count
    def get_by_id(self, record_id: str) -> dict[str, Any] | None:
        row = self.records.get(record_id); return row["document"] if row else None
    def health_check(self) -> dict[str, Any]:
        return {"healthy": True, "count": len(self.records), "model_id": self.model_id, "path": str(self.path)}
