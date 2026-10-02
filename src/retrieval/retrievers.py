"""Swappable deterministic retrievers and an optional injected embedding backend."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Callable, Protocol

TOKEN = re.compile(r"[\w.-]+", re.UNICODE)


class Retriever(Protocol):
    def search(self, query: str, *, top_k: int = 5, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]: ...


def _meta(record: dict[str, Any]) -> dict[str, Any]:
    return record.get("metadata") if isinstance(record.get("metadata"), dict) else record


def _matches(record: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    metadata = _meta(record)
    return all(metadata.get(k) == v or record.get(k) == v for k, v in (filters or {}).items())


def infer_metadata_filters(query: str, documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract only exact metadata values named in the query; omit ambiguous fields."""
    normalized_query = query.casefold()
    filters: dict[str, Any] = {}
    for key in ("country_code", "country_name", "publication_year", "topic", "source", "document_id"):
        values: dict[str, Any] = {}
        for doc in documents:
            value = _meta(doc).get(key) if _meta(doc).get(key) is not None else doc.get(key)
            if isinstance(value, (str, int, float)):
                values[str(value)] = value
        matched = [value for value in values.values()
                   if len(str(value)) >= 3 and re.search(
                       rf"(?<!\w){re.escape(str(value).casefold())}(?!\w)", normalized_query)]
        if len(matched) == 1:
            filters[key] = matched[0]
    return filters


def _terms(text: str) -> list[str]:
    return [w.casefold() for w in TOKEN.findall(text)]


class KeywordRetriever:
    """Simple query term coverage baseline."""
    method = "keyword"
    def __init__(self, documents: list[dict[str, Any]]): self.documents = documents
    def search(self, query: str, *, top_k: int = 5, filters=None) -> list[dict[str, Any]]:
        q = set(_terms(query))
        if top_k < 1: raise ValueError("top_k must be positive")
        out = []
        for i, doc in enumerate(self.documents):
            if not _matches(doc, filters): continue
            d = set(_terms(doc.get("text", "") + " " + str(doc.get("title", ""))))
            score = len(q & d) / len(q) if q else 0.0
            if score: out.append((score, i, doc))
        out.sort(key=lambda x: (-x[0], x[1]))
        return [_result(doc, score, self.method) for score, _, doc in out[:top_k]]


class BM25Retriever:
    """Local BM25 lexical ranking (Okapi, k1=1.5, b=0.75)."""
    method = "bm25"
    def __init__(self, documents: list[dict[str, Any]], *, k1: float = 1.5, b: float = .75):
        self.documents, self.k1, self.b = documents, k1, b
        self.tokens = [_terms(d.get("text", "") + " " + str(d.get("title", ""))) for d in documents]
        self.avgdl = sum(map(len, self.tokens)) / max(1, len(self.tokens))
        self.df = Counter(term for terms in self.tokens for term in set(terms))
    def search(self, query: str, *, top_k: int = 5, filters=None) -> list[dict[str, Any]]:
        if top_k < 1: raise ValueError("top_k must be positive")
        q = set(_terms(query)); n = len(self.documents); ranked = []
        for i, (doc, terms) in enumerate(zip(self.documents, self.tokens)):
            if not _matches(doc, filters): continue
            counts = Counter(terms); score = 0.0
            for term in q:
                tf = counts[term]
                if tf:
                    idf = math.log(1 + (n - self.df[term] + .5) / (self.df[term] + .5))
                    score += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * len(terms) / max(self.avgdl, 1)))
            if score: ranked.append((score, i, doc))
        max_score = max((x[0] for x in ranked), default=1.0)
        ranked.sort(key=lambda x: (-x[0], x[1]))
        return [_result(doc, score / max_score, self.method, raw=score) for score, _, doc in ranked[:top_k]]


def _result(doc: dict[str, Any], score: float, method: str, **components: float) -> dict[str, Any]:
    result = dict(doc)
    result.update({"retrieval_method": method, "score": float(score), "component_scores": {method: float(score), **components}})
    return result


class SemanticRetriever:
    """Cosine dense retrieval through an injected batch embedder; never fakes semantic mode."""
    method = "semantic"
    def __init__(self, documents: list[dict[str, Any]], embed: Callable[[list[str]], list[list[float]]], *, model_id: str):
        if not model_id: raise ValueError("model_id is required for semantic retrieval")
        self.documents, self.embed, self.model_id = documents, embed, model_id
        vectors = embed([d.get("text", "") for d in documents]) if documents else []
        if len(vectors) != len(documents): raise ValueError("embedder returned the wrong vector count")
        if len({len(v) for v in vectors}) > 1: raise ValueError("embedding dimensions must match")
        self.vectors = [_normalize(v) for v in vectors]
    def search(self, query: str, *, top_k: int = 5, filters=None) -> list[dict[str, Any]]:
        if top_k < 1: raise ValueError("top_k must be positive")
        if not self.documents: return []
        query_vectors = self.embed([query])
        if len(query_vectors) != 1: raise ValueError("embedder must return one query vector")
        qv = _normalize(query_vectors[0]); ranked = []
        if len(qv) != len(self.vectors[0]): raise ValueError("query/document embedding dimensions do not match")
        for i, (doc, vec) in enumerate(zip(self.documents, self.vectors)):
            if _matches(doc, filters): ranked.append((sum(a*b for a,b in zip(qv, vec)), i, doc))
        ranked.sort(key=lambda x: (-x[0], x[1]))
        return [_result(doc, score, self.method, embedding_model=self.model_id) for score, _, doc in ranked[:top_k]]


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(float(x) ** 2 for x in vector))
    if not norm: raise ValueError("embedding vector must have non-zero magnitude")
    return [float(x) / norm for x in vector]


class HybridRetriever:
    """Fuse lexical and semantic ranked scores after per-query min-max calibration."""
    method = "hybrid"
    def __init__(self, lexical: Retriever, semantic: Retriever, *, lexical_weight: float = .5, semantic_weight: float = .5):
        if min(lexical_weight, semantic_weight) < 0 or lexical_weight + semantic_weight <= 0:
            raise ValueError("hybrid weights must be non-negative and not both zero")
        total = lexical_weight + semantic_weight
        self.lexical, self.semantic = lexical, semantic
        self.weights = (lexical_weight / total, semantic_weight / total)
    def search(self, query: str, *, top_k: int = 5, filters=None) -> list[dict[str, Any]]:
        if top_k < 1: raise ValueError("top_k must be positive")
        size = max(top_k * 4, top_k)
        left = self.lexical.search(query, top_k=size, filters=filters)
        right = self.semantic.search(query, top_k=size, filters=filters)
        maps = {}
        for results, name in ((left, "lexical"), (right, "semantic")):
            scores = [float(r.get("score", 0)) for r in results]
            low, high = min(scores, default=0), max(scores, default=0)
            for r, score in zip(results, scores):
                key = r.get("chunk_id") or r.get("document_id") or r.get("source_url") or r.get("text")
                norm = (score - low) / (high - low) if high > low else (1.0 if score > 0 else 0.0)
                maps.setdefault(key, {"doc": r, "lexical": 0.0, "semantic": 0.0})[name] = norm
        fused = []
        for key, item in maps.items():
            score = self.weights[0] * item["lexical"] + self.weights[1] * item["semantic"]
            record = dict(item["doc"]); record.update(retrieval_method=self.method, score=score,
                component_scores={"lexical": item["lexical"], "semantic": item["semantic"]})
            fused.append((score, str(key), record))
        fused.sort(key=lambda x: (-x[0], x[1]))
        return [x[2] for x in fused[:top_k]]


def make_retriever(documents: list[dict[str, Any]], *, method: str = "bm25",
                   embed: Callable[[list[str]], list[list[float]]] | None = None,
                   model_id: str | None = None, lexical_weight: float = .5,
                   semantic_weight: float = .5) -> Retriever:
    """Construct a requested backend and fail clearly when semantic support is absent."""
    if method == "keyword": return KeywordRetriever(documents)
    if method == "bm25": return BM25Retriever(documents)
    if method == "semantic":
        if embed is None or not model_id: raise RuntimeError("Semantic retrieval requires an injected embedder and EMBEDDING_MODEL.")
        return SemanticRetriever(documents, embed, model_id=model_id)
    if method == "hybrid":
        if embed is None or not model_id: raise RuntimeError("Hybrid retrieval requires an injected embedder and EMBEDDING_MODEL.")
        return HybridRetriever(BM25Retriever(documents), SemanticRetriever(documents, embed, model_id=model_id),
                               lexical_weight=lexical_weight, semantic_weight=semantic_weight)
    raise ValueError(f"Unsupported retrieval method: {method}")
