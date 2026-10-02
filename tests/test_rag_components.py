"""Offline tests for RAG ingestion, indexes, context, and stable citations."""
import json
from src.retrieval.chunking import chunk_document
from src.retrieval.context import build_context, generate_answer
from src.retrieval.document_loader import load_documents, load_document_chunks, make_document
from src.retrieval.retrievers import BM25Retriever, HybridRetriever, KeywordRetriever, SemanticRetriever, infer_metadata_filters
from src.retrieval.vector_store import LocalVectorStore

def test_loader_supports_csv_text_markdown_and_skips_bad_pdf(tmp_path):
    (tmp_path / "a.txt").write_text("World Bank water investment evidence.", encoding="utf-8")
    (tmp_path / "b.md").write_text("Climate report.", encoding="utf-8")
    (tmp_path / "c.csv").write_text("country,topic\nGBR,water\n", encoding="utf-8")
    (tmp_path / "broken.pdf").write_bytes(b"not a valid PDF")
    docs = load_documents(tmp_path)
    assert len(docs) == 3
    assert all(set(("document_id", "title", "source", "source_url", "text", "metadata", "ingested_at")) <= set(d) for d in docs)
    chunks = load_document_chunks(tmp_path, chunk_size=12, overlap=2)
    assert chunks and all(chunk["chunk_id"] for chunk in chunks)

def test_duplicate_file_content_is_skipped(tmp_path):
    (tmp_path / "a.txt").write_text("same document", encoding="utf-8")
    (tmp_path / "b.txt").write_text("same document", encoding="utf-8")
    assert len(load_documents(tmp_path)) == 1

def test_chunk_document_ids_are_deterministic_and_metadata_survives():
    doc = make_document("GDP increased in 2020. GDP stabilized in 2021.", document_id="d1",
                        title="GDP", metadata={"country_code": "GBR", "topic": "economy"})
    left = chunk_document(doc, chunk_size=30, overlap=5)
    right = chunk_document(doc, chunk_size=30, overlap=5)
    assert [c["chunk_id"] for c in left] == [c["chunk_id"] for c in right]
    assert left[0]["metadata"]["country_code"] == "GBR"

def test_keyword_bm25_semantic_and_hybrid_rank_with_metadata_filters():
    docs = [
        {"chunk_id": "a", "document_id": "a", "country_code": "GBR", "title": "GDP", "text": "GDP value and economic growth"},
        {"chunk_id": "b", "document_id": "b", "country_code": "IND", "title": "water", "text": "Water project rural development"},
    ]
    assert KeywordRetriever(docs).search("GDP growth", filters={"country_code": "GBR"})[0]["chunk_id"] == "a"
    assert BM25Retriever(docs).search("GDP growth")[0]["chunk_id"] == "a"
    def embed(texts):
        return [[1, 0] if "gdp" in text.casefold() else [0, 1] for text in texts]
    semantic = SemanticRetriever(docs, embed, model_id="mock-v1")
    assert semantic.search("GDP")[0]["chunk_id"] == "a"
    hybrid = HybridRetriever(BM25Retriever(docs), semantic, lexical_weight=.6, semantic_weight=.4)
    assert hybrid.search("GDP", top_k=1)[0]["retrieval_method"] == "hybrid"
    assert infer_metadata_filters("GDP evidence in GBR", docs) == {"country_code": "GBR"}

def test_context_is_bounded_deduplicated_and_cites_real_passages():
    passage = {"chunk_id": "c1", "document_id": "d1", "title": "GDP report", "source_url": "https://example.test/r", "text": "GDP increased with evidence."}
    context = build_context([passage, dict(passage)], max_chars=200)
    assert len(context["retrieved_chunks"]) == 1
    assert context["citations"][0]["chunk_id"] == "c1"
    output = generate_answer("GDP?", context)
    assert output["generation_model"] == "offline-extractive"
    assert output["citations"] == context["citations"]

def test_configured_generation_receives_evidence_safety_instruction():
    context = build_context([{"chunk_id": "c", "document_id": "d", "source_url": "https://x", "text": "Evidence."}])
    observed = {}
    def generator(question, evidence, instruction):
        observed.update(question=question, evidence=evidence, instruction=instruction)
        return "Answer [E1]"
    output = generate_answer("Question?", context, generator=generator, model_name="mock-model")
    assert output["generation_model"] == "mock-model"
    assert "untrusted data" in observed["instruction"]

def test_local_vector_store_persists_and_supports_crud(tmp_path):
    path = tmp_path / "index.json"
    store = LocalVectorStore(path, model_id="mock-v1")
    doc = {"chunk_id": "c1", "document_id": "d1", "text": "GDP evidence"}
    assert store.add_documents([doc], [[1, 0]]) == 1
    assert LocalVectorStore(path, model_id="mock-v1").get_by_id("c1") == doc
    assert store.search([1, 0])[0]["chunk_id"] == "c1"
    assert store.delete(["c1"]) == 1
