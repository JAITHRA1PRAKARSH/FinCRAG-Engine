import pytest
from src.ingestion.chunker import load_all_filings, identify_company
from src.retrieval.vector_store import build_vector_store

def test_company_identification():
    company, ticker = identify_company("data/filings/AAPL_10K_2025-10-31.md")
    assert ticker == "AAPL"
    assert "Apple" in company

def test_chunking_and_metadata():
    chunks = load_all_filings()
    assert len(chunks) > 0
    first_chunk = chunks[0]
    assert "Company:" in first_chunk.page_content
    assert "ticker" in first_chunk.metadata

def test_vector_search_returns_results():
    chunks = load_all_filings()
    store = build_vector_store(chunks)
    results = store.similarity_search("Apple net sales revenue", k=2)
    assert len(results) == 2