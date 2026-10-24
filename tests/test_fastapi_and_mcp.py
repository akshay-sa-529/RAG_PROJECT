"""
Unit and Integration Tests for FastAPI Backend, MongoDB Persistence, and MCP Tools.
Validates:
1. MongoDB document & chunk storage with automatic fallback.
2. MCP chunking tool logic & metadata integrity.
3. MCP retrieval tool & grounding thinking trace.
4. FastAPI REST endpoints via TestClient.
"""

from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from api import app
from rag.db import get_rag_database, InMemoryStore
from rag.mcp_tools import execute_mcp_chunking, execute_mcp_retrieval
from rag.generation import generate_grounded_answer


@pytest.fixture
def test_client():
    """Provides a TestClient fixture for FastAPI endpoints."""
    return TestClient(app)



def test_mongodb_document_and_chunk_storage():
    """Tests saving and retrieving documents and chunks in the database layer."""
    db = get_rag_database()
    doc_id = "test_mongo_doc_001"
    filename = "sample_paper.pdf"

    # Save document metadata
    saved_doc = db.save_document(
        doc_id=doc_id,
        filename=filename,
        file_size_bytes=4096,
        total_pages=3,
        total_chunks=5,
        metadata={"author": "AI Researcher"}
    )
    assert saved_doc["doc_id"] == doc_id
    assert saved_doc["total_pages"] == 3

    # Retrieve document
    retrieved_doc = db.get_document(doc_id)
    assert retrieved_doc is not None
    assert retrieved_doc["filename"] == filename

    # Save chunks
    sample_chunks = [
        Document(
            page_content=f"This is chunk number {i} with semantic information on Page {i+1}.",
            metadata={"doc_id": doc_id, "page": i + 1, "page_number": i + 1, "chunk_id": f"{doc_id}_chunk_{i}"}
        )
        for i in range(5)
    ]
    saved_chunks_count = db.save_chunks(doc_id=doc_id, chunks=sample_chunks)
    assert saved_chunks_count == 5

    # Retrieve chunks
    chunks_from_db = db.get_chunks(doc_id)
    assert len(chunks_from_db) == 5
    assert chunks_from_db[0]["page"] == 1
    assert chunks_from_db[0]["chunk_id"] == f"{doc_id}_chunk_0"

    # Clean up
    db.delete_document(doc_id)
    assert db.get_document(doc_id) is None
    assert len(db.get_chunks(doc_id)) == 0


def test_mcp_chunking_execution():
    """Tests that execute_mcp_chunking handles pages, produces chunks, and stores to MongoDB."""
    pages = [
        Document(page_content="Paragraph on deep learning and attention mechanisms." * 15, metadata={"page": 1}),
        Document(page_content="Section discussing transformer architectures and vector spaces." * 15, metadata={"page": 2}),
    ]
    doc_id = "mcp_test_doc_002"
    filename = "deep_learning.pdf"

    result = execute_mcp_chunking(
        doc_id=doc_id,
        filename=filename,
        pages=pages,
        chunk_size=500,
        chunk_overlap=100,
        store_in_qdrant=False  # isolated test without requiring live cloud cluster
    )

    assert result["status"] == "success"
    assert result["doc_id"] == doc_id
    assert result["total_chunks"] > 2
    assert result["mongodb_saved_count"] == result["total_chunks"]

    # Verify chunks are readable from MongoDB
    db = get_rag_database()
    saved = db.get_chunks(doc_id)
    assert len(saved) == result["total_chunks"]
    assert all("chunk_id" in ch for ch in saved)

    # Clean up
    db.delete_document(doc_id)


def test_mcp_thinking_trace_and_session_logging():
    """Tests that generation includes a structured thinking trace and logs to MongoDB."""
    db = get_rag_database()
    doc_id = "thinking_test_doc_003"

    sample_chunk = Document(
        page_content="Transformers utilize self-attention mechanisms to weigh token importance dynamically.",
        metadata={"doc_id": doc_id, "page": 4, "similarity_score": 0.94}
    )

    with patch("rag.generation.get_llm_client") as mock_get_llm:
        mock_llm_inst = MagicMock()
        mock_llm_inst.invoke.return_value = MagicMock(content="Transformers use self-attention mechanisms to weigh tokens.")
        mock_get_llm.return_value = mock_llm_inst

        result = generate_grounded_answer(
            query="What mechanism do transformers use to weigh token importance?",
            retrieved_chunks=[sample_chunk],
            openrouter_api_key="mock_key",
            model_name="recursive/flint-v1.0:free",
            doc_id=doc_id,
            include_thinking=True
        )

    # Check thinking trace
    assert "thinking_trace" in result

    trace = result["thinking_trace"]
    assert len(trace) >= 2
    assert any(step.get("phase") == "Question Analysis" for step in trace)
    assert any(step.get("phase") == "Context Inspection" for step in trace)

    # Verify thinking session logged into database
    recent = db.get_recent_sessions(limit=5)
    assert len(recent) > 0
    latest = recent[-1]
    assert "query" in latest
    assert "thinking_trace" in latest


def test_fastapi_root_and_health(test_client):
    """Tests GET / and GET /health endpoints."""
    res_root = test_client.get("/")
    assert res_root.status_code == 200
    data_root = res_root.json()
    assert data_root["service"] == "PaperMind RAG API"
    assert "fastapi" in data_root["features"]
    assert "mongodb" in data_root["features"]

    res_health = test_client.get("/health")
    assert res_health.status_code == 200
    data_health = res_health.json()
    assert "mongodb" in data_health
    assert "qdrant" in data_health


def test_fastapi_chunk_and_document_endpoints(test_client):
    """Tests POST /documents/chunk, GET /documents, GET /documents/{doc_id}/chunks, and DELETE /documents/{doc_id}."""
    doc_id = "fastapi_test_doc_004"
    payload = {
        "doc_id": doc_id,
        "filename": "fastapi_guide.pdf",
        "pages": [
            "FastAPI is a modern, fast (high-performance) web framework for building APIs with Python 3.8+ based on standard Python type hints.",
            "Key features of FastAPI include fast performance, rapid development, automatic interactive documentation, and standards compliance."
        ],
        "chunk_size": 200,
        "chunk_overlap": 50
    }

    # 1. Chunk & save document
    res_chunk = test_client.post("/documents/chunk", json=payload)
    assert res_chunk.status_code == 200
    chunk_data = res_chunk.json()
    assert chunk_data["status"] == "success"
    assert chunk_data["doc_id"] == doc_id
    assert chunk_data["total_chunks"] >= 2

    # 2. List documents
    res_list = test_client.get("/documents")
    assert res_list.status_code == 200
    docs = res_list.json()["documents"]
    assert any(d["doc_id"] == doc_id for d in docs)

    # 3. Get document metadata
    res_doc = test_client.get(f"/documents/{doc_id}")
    assert res_doc.status_code == 200
    assert res_doc.json()["filename"] == "fastapi_guide.pdf"

    # 4. Get document chunks
    res_chunks = test_client.get(f"/documents/{doc_id}/chunks")
    assert res_chunks.status_code == 200
    assert res_chunks.json()["total_chunks"] == chunk_data["total_chunks"]

    # 5. Query endpoint with empty or missing retrieval
    query_payload = {
        "query": "What is Python?",
        "doc_id": doc_id,
        "top_k": 2
    }
    with patch("rag.generation.get_llm_client") as mock_get_llm:
        mock_llm_inst = MagicMock()
        mock_llm_inst.invoke.return_value = MagicMock(content="FastAPI is a modern web framework.")
        mock_get_llm.return_value = mock_llm_inst

        res_query = test_client.post("/query", json=query_payload)
        assert res_query.status_code == 200
        query_data = res_query.json()
        assert "answer" in query_data
        assert "thinking_trace" in query_data
        assert "citations" in query_data


    # 6. Delete document
    res_del = test_client.delete(f"/documents/{doc_id}")
    assert res_del.status_code == 200
    assert res_del.json()["deleted"] is True
