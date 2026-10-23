"""
Unit & Pipeline Tests for Learn RAG Application.
Validates chunking, metadata preservation, 1-indexed page numbers,
citation formatting, configuration checks, and grounding prompts.
"""

import pytest
from langchain_core.documents import Document

from rag.config import AppConfig, validate_config
from rag.ingestion import (
    compute_document_id,
    split_document_pages,
    load_pdf_from_bytes,
)
from rag.retrieval import format_citations
from rag.generation import build_context_string, generate_grounded_answer, GROUNDED_SYSTEM_PROMPT


def test_config_validation_missing_keys():
    """Test that missing keys or invalid configurations are properly flagged."""
    cfg_empty = AppConfig(
        qdrant_url="",
        qdrant_api_key="",
        openrouter_api_key="",
    )
    is_valid, issues = validate_config(cfg_empty)
    assert not is_valid
    assert any("QDRANT_URL" in issue for issue in issues)
    assert any("QDRANT_API_KEY" in issue for issue in issues)
    assert any("OPENROUTER_API_KEY" in issue for issue in issues)

    # Localhost forbidden test
    cfg_localhost = AppConfig(
        qdrant_url="http://localhost:6333",
        qdrant_api_key="valid_key",
        openrouter_api_key="valid_key",
    )
    is_valid, issues = validate_config(cfg_localhost)
    assert not is_valid
    assert any("localhost" in issue for issue in issues)


def test_document_hash_generation():
    """Test deterministic document hash generation."""
    sample_bytes_1 = b"%PDF-1.4 Mock PDF Content A"
    sample_bytes_2 = b"%PDF-1.4 Mock PDF Content B"
    
    hash_1a = compute_document_id(sample_bytes_1, "file1.pdf")
    hash_1b = compute_document_id(sample_bytes_1, "file1.pdf")
    hash_2 = compute_document_id(sample_bytes_2, "file2.pdf")
    
    assert hash_1a == hash_1b
    assert hash_1a != hash_2
    assert len(hash_1a) == 16


def test_split_document_pages_chunking():
    """Test that text splitting adheres to chunk size 1000 and overlap 400."""
    page_1_text = "Word " * 300  # ~1500 chars
    page_2_text = "Section " * 250  # ~2000 chars
    
    docs = [
        Document(page_content=page_1_text, metadata={"page": 1, "filename": "test.pdf"}),
        Document(page_content=page_2_text, metadata={"page": 2, "filename": "test.pdf"}),
    ]
    
    doc_id = "test_doc_123"
    chunks = split_document_pages(
        documents=docs,
        doc_id=doc_id,
        filename="test.pdf",
        chunk_size=1000,
        chunk_overlap=400,
    )
    
    assert len(chunks) > 2
    for chunk in chunks:
        assert len(chunk.page_content) <= 1000
        assert chunk.metadata["doc_id"] == doc_id
        assert chunk.metadata["filename"] == "test.pdf"
        assert chunk.metadata["page"] in [1, 2]
        assert "chunk_id" in chunk.metadata
        assert chunk.metadata["chunk_id"].startswith(f"{doc_id}_chunk_")


def test_empty_pdf_handling():
    """Test that 0-byte or corrupted inputs raise descriptive ValueErrors."""
    with pytest.raises(ValueError, match="empty"):
        load_pdf_from_bytes(b"", "empty.pdf")


def test_citation_formatting():
    """Test that citations are deduplicated and formatted with human-friendly page numbers."""
    chunks = [
        Document(
            page_content="Content from page 3",
            metadata={"filename": "lecture_notes.pdf", "page": 3, "similarity_score": 0.88},
        ),
        Document(
            page_content="Another piece from page 1",
            metadata={"filename": "lecture_notes.pdf", "page": 1, "similarity_score": 0.92},
        ),
        Document(
            page_content="Second piece from page 3",
            metadata={"filename": "lecture_notes.pdf", "page": 3, "similarity_score": 0.85},
        ),
    ]
    
    citations = format_citations(chunks)
    
    assert "lecture_notes.pdf" in citations["citations_by_file"]
    assert citations["citations_by_file"]["lecture_notes.pdf"] == [1, 3]
    assert "lecture_notes.pdf (Pages 1, 3)" in citations["citation_summary"]
    assert len(citations["source_details"]) == 3


def test_context_builder_labels():
    """Test context block construction contains source filename and page numbers."""
    chunks = [
        Document(page_content="Retrieval Augmented Generation bridges LLMs with external data.",
                 metadata={"filename": "rag_intro.pdf", "page": 2}),
        Document(page_content="Embeddings represent semantic meaning in dense vector space.",
                 metadata={"filename": "rag_intro.pdf", "page": 4}),
    ]
    
    context_str = build_context_string(chunks)
    assert "Passage 1 | Source: rag_intro.pdf | Page 2" in context_str
    assert "Passage 2 | Source: rag_intro.pdf | Page 4" in context_str
    assert "Retrieval Augmented Generation bridges" in context_str


def test_empty_chunks_grounded_fallback():
    """Test that asking with no retrieved chunks immediately yields the grounded fallback without calling LLM."""
    result = generate_grounded_answer(
        query="What is the speed of light?",
        retrieved_chunks=[],
        openrouter_api_key="mock_key",
        model_name="recursive/flint-v1.0:free",
    )
    
    assert result["answer"] == "The provided document does not contain information to answer this question."
    assert result["grounded"] is False
    assert result["chunks"] == []


def test_grounded_system_prompt_rules():
    """Test that the system prompt strictly forbids hallucination and enforces document-only answers."""
    assert "EXCLUSIVELY" in GROUNDED_SYSTEM_PROMPT
    assert "The provided document does not contain information to answer this question." in GROUNDED_SYSTEM_PROMPT
    assert "Do not invent or hallucinate" in GROUNDED_SYSTEM_PROMPT
