# Local RAG Guide

## Load, chunk, retrieve

`load_documents(folder)` supports PDF (optional `pypdf`), TXT, Markdown, and CSV, and `make_document(text, ...)` accepts inline text. Records share document ID, title, source and optional URL/country/year/topic/page fields, text, metadata, and ingestion time. A malformed file is logged and skipped. PDFs retain page numbers where extraction provides pages. No OCR is included.

`chunk_document` emits deterministic `chunk_id`, `document_id`, index, passage text, and preserved metadata including character bounds. Size and overlap are configurable. Existing `chunk_text` remains available.

`KeywordRetriever` uses query term coverage. `BM25Retriever` is an offline Okapi BM25 ranker with fixed documented `k1=1.5`, `b=0.75`. `SemanticRetriever` requires an injected batch embedder and explicit model ID; it normalizes vectors and uses cosine similarity. Missing models do not fall back silently. `HybridRetriever` min-max normalizes each component per query, applies configured weights, deduplicates by chunk ID, and returns component scores. Metadata filters are accepted by each retriever.

`LocalVectorStore` stores records and normalized vectors in JSON for development/testing. It is not a distributed or high-volume index. Databricks Vector Search is not required or enabled.

## Context and answers

`build_context` removes exact duplicate passages, requires a source reference, enforces a character limit, and retains citation metadata. It places retrieved text in an evidence-only context. An injected external generator receives the question, bounded context, and explicit instruction to treat retrieved text as untrusted evidence and to answer only from it. With no generator, `generate_answer` quotes retrieved passages. With no evidence, it explicitly declines to invent an answer. No provider is called by default and no credentials are stored in source.

The Research Agent accepts a retriever or a local document corpus. It returns structured answer, citations, retrieved chunks, retrieval method, model name, and limitations within its result data. Documents & Reports API metadata is separately labeled and is never represented as a downloaded report body.

## Run locally

```powershell
pip install -r requirements.txt
pip install ".[documents]"  # only when PDF input is needed
python -m app.app --question "Summarize evidence about rural water resilience" --documents-folder data/documents --json
```

`CHUNK_SIZE`, `CHUNK_OVERLAP`, `RETRIEVAL_TOP_K`, `RETRIEVAL_METHOD`, and hybrid weights are listed in `.env.example`. An embedding model and provider must be explicitly supplied by an integrator before semantic retrieval is usable.
