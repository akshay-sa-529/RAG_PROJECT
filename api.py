"""
FastAPI Backend Service for Learn RAG (PaperMind AI).
Provides RESTful APIs for document uploading, MCP-driven chunking,
MongoDB persistence, thinking-phase retrieval, and mounts the Model Context Protocol (MCP) SSE server.
"""

import os
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag.config import get_config, validate_config
from rag.db import get_rag_database
from rag.ingestion import (
    compute_document_id,
    load_pdf_from_bytes,
    get_qdrant_client,
    delete_document_vectors,
)
from rag.mcp_tools import (
    mcp_server,
    execute_mcp_chunking,
    execute_mcp_retrieval,
)
from rag.generation import generate_grounded_answer

app = FastAPI(
    title="PaperMind RAG API",
    description="Production-grade RAG backend with FastAPI, MongoDB persistence, and Model Context Protocol (MCP) thinking & chunking.",
    version="1.0.0"
)

# CORS middleware for open integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount MCP Server SSE endpoint
try:
    app.mount("/mcp", mcp_server.sse_app())
except Exception as e:
    pass


# -----------------------------------------------------------------------------
# Request & Response Schemas
# -----------------------------------------------------------------------------

class ChunkTextRequest(BaseModel):
    doc_id: Optional[str] = Field(None, description="Optional custom document ID")
    filename: str = Field(..., description="Document filename")
    pages: List[str] = Field(..., description="List of page text strings")
    chunk_size: Optional[int] = Field(None, description="Character chunk size (default 1000)")
    chunk_overlap: Optional[int] = Field(None, description="Overlap characters (default 400)")


class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language question to ask")
    doc_id: Optional[str] = Field(None, description="Target document ID to scope retrieval")
    model_name: Optional[str] = Field(None, description="OpenRouter model identifier")
    top_k: Optional[int] = Field(4, description="Number of passages to retrieve")


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.get("/", tags=["General"])
def root_info():
    """Returns general service overview and active feature flags."""
    config = get_config()
    db = get_rag_database()
    return {
        "service": "PaperMind RAG API",
        "version": "1.0.0",
        "features": {
            "fastapi": "Active",
            "mongodb": db.connection_status["storage_mode"],
            "mcp": "Active (Tools & SSE endpoint mounted at /mcp)",
            "vector_store": "Qdrant Cloud",
            "embeddings": config.embedding_model_name,
            "default_model": config.openrouter_model
        },
        "docs_url": "/docs"
    }


@app.get("/health", tags=["General"])
def health_check():
    """Performs deep health check across Qdrant Cloud and MongoDB."""
    config = get_config()
    db = get_rag_database()
    is_valid, config_issues = validate_config(config)

    qdrant_healthy = False
    qdrant_error = None
    if config.qdrant_url and config.qdrant_api_key:
        try:
            client = get_qdrant_client(config.qdrant_url, config.qdrant_api_key)
            client.get_collections()
            qdrant_healthy = True
        except Exception as exc:
            qdrant_error = str(exc)

    return {
        "status": "healthy" if (qdrant_healthy or db.is_connected) else "degraded",
        "mongodb": db.connection_status,
        "qdrant": {
            "configured": bool(config.qdrant_url and config.qdrant_api_key),
            "reachable": qdrant_healthy,
            "error": qdrant_error,
            "collection": config.qdrant_collection_name
        },
        "config_valid": is_valid,
        "config_issues": config_issues
    }


@app.get("/config", tags=["General"])
def get_configuration():
    """Returns safe configuration metadata without exposing secret keys."""
    config = get_config()
    db = get_rag_database()
    return {
        "qdrant_url": config.qdrant_url[:20] + "..." if config.qdrant_url else "",
        "qdrant_collection": config.qdrant_collection_name,
        "openrouter_model": config.openrouter_model,
        "embedding_model": config.embedding_model_name,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "top_k": config.top_k,
        "mongodb": db.connection_status
    }


@app.post("/documents/upload", tags=["Ingestion & Chunking"])
async def upload_and_chunk_pdf(
    file: UploadFile = File(...),
    chunk_size: Optional[int] = Form(None),
    chunk_overlap: Optional[int] = Form(None)
):
    """
    Uploads a PDF file, extracts 1-indexed pages, splits with MCP chunking,
    saves chunks to MongoDB, and indexes vectors into Qdrant Cloud.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF documents are supported for ingestion."
        )

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes)."
        )

    doc_id = compute_document_id(file_bytes, file.filename)

    try:
        pages = load_pdf_from_bytes(file_bytes, file.filename)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to extract pages from PDF: {exc}"
        )

    mcp_result = execute_mcp_chunking(
        doc_id=doc_id,
        filename=file.filename,
        pages=pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        store_in_qdrant=True
    )

    mcp_result["file_size_kb"] = round(len(file_bytes) / 1024, 2)
    mcp_result["total_pages"] = len(pages)
    return mcp_result


@app.post("/documents/chunk", tags=["Ingestion & Chunking"])
def chunk_raw_pages(payload: ChunkTextRequest):
    """
    Executes MCP chunking on raw text pages, writes chunks to MongoDB,
    and indexes vectors into Qdrant Cloud.
    """
    from langchain_core.documents import Document

    doc_id = payload.doc_id or compute_document_id(
        "".join(payload.pages).encode("utf-8"), payload.filename
    )

    docs = [
        Document(page_content=text, metadata={"page": i + 1, "filename": payload.filename, "doc_id": doc_id})
        for i, text in enumerate(payload.pages)
    ]

    return execute_mcp_chunking(
        doc_id=doc_id,
        filename=payload.filename,
        pages=docs,
        chunk_size=payload.chunk_size,
        chunk_overlap=payload.chunk_overlap,
        store_in_qdrant=True
    )


@app.get("/documents", tags=["Documents & Chunks"])
def list_documents():
    """Lists all registered documents saved in MongoDB."""
    db = get_rag_database()
    return {"documents": db.list_documents()}


@app.get("/documents/{doc_id}", tags=["Documents & Chunks"])
def get_document(doc_id: str):
    """Retrieves document record from MongoDB."""
    db = get_rag_database()
    doc = db.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document with ID '{doc_id}' not found.")
    return doc


@app.get("/documents/{doc_id}/chunks", tags=["Documents & Chunks"])
def get_document_chunks(doc_id: str, limit: Optional[int] = Query(None, ge=1, le=500)):
    """Retrieves chunks stored in MongoDB for a specific document."""
    db = get_rag_database()
    chunks = db.get_chunks(doc_id)
    if not chunks:
        # Check if doc exists
        doc = db.get_document(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail=f"No chunks or document found for ID '{doc_id}'.")
    if limit:
        chunks = chunks[:limit]
    return {
        "doc_id": doc_id,
        "total_chunks": len(chunks),
        "chunks": chunks
    }


@app.delete("/documents/{doc_id}", tags=["Documents & Chunks"])
def delete_document(doc_id: str):
    """Deletes document and chunks from MongoDB and Qdrant Cloud vectors."""
    config = get_config()
    db = get_rag_database()

    # 1. Delete from MongoDB
    db.delete_document(doc_id)

    # 2. Delete vectors from Qdrant Cloud
    deleted_vectors = 0
    if config.qdrant_url and config.qdrant_api_key:
        try:
            client = get_qdrant_client(config.qdrant_url, config.qdrant_api_key)
            deleted_vectors = delete_document_vectors(client, config.qdrant_collection_name, doc_id)
        except Exception as exc:
            pass

    return {
        "doc_id": doc_id,
        "deleted": True,
        "deleted_vectors_qdrant": deleted_vectors
    }


@app.post("/query", tags=["Conversational Intelligence"])
def query_rag(payload: QueryRequest):
    """
    Executes grounded RAG query generation:
    1. Orchestrates MCP retrieval tool during thinking phase.
    2. Builds visible thinking trace.
    3. Generates grounded answer with exact 1-indexed page citations.
    4. Logs thinking trace to MongoDB.
    """
    config = get_config()
    model = payload.model_name or config.openrouter_model

    # 1. Execute MCP retrieval tool
    mcp_retrieval = execute_mcp_retrieval(
        query=payload.query,
        doc_id=payload.doc_id,
        top_k=payload.top_k
    )

    retrieved_chunks = mcp_retrieval.get("raw_docs", [])

    # 2. Execute grounded answer generation with thinking trace
    result = generate_grounded_answer(
        query=payload.query,
        retrieved_chunks=retrieved_chunks,
        openrouter_api_key=config.openrouter_api_key,
        model_name=model,
        doc_id=payload.doc_id,
        include_thinking=True
    )

    clean_chunks = [
        {
            "page": ch.metadata.get("page", 1),
            "score": ch.metadata.get("similarity_score"),
            "content": ch.page_content
        }
        for ch in retrieved_chunks
    ]

    return {
        "query": payload.query,
        "answer": result["answer"],
        "citations": result["citations"],
        "thinking_trace": result.get("thinking_trace", []),
        "retrieval_source": mcp_retrieval.get("retrieval_source"),
        "chunks": clean_chunks,
        "model_used": result["model_used"],
        "grounded": result["grounded"]
    }


@app.get("/history", tags=["Conversational Intelligence"])
def get_thinking_history(limit: int = Query(20, ge=1, le=100)):
    """Retrieves recent thinking and query sessions from MongoDB."""
    db = get_rag_database()
    return {"sessions": db.get_recent_sessions(limit)}


if __name__ == "__main__":
    import uvicorn
    config = get_config()
    uvicorn.run("api:app", host=config.fastapi_host, port=config.fastapi_port, reload=True)
