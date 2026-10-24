"""
Model Context Protocol (MCP) Server and Tool Suite for Learn RAG.
Exposes standard MCP tools for document chunking, MongoDB persistence,
and semantic retrieval during the LLM thinking phase.
Supports both direct Python invocation, FastAPI SSE mounting, and stdio execution.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from mcp.server.mcpserver import MCPServer

from rag.config import get_config
from rag.db import get_rag_database
from rag.ingestion import (
    get_qdrant_client,
    ensure_collection_exists,
    index_chunks_to_qdrant,
)
from rag.retrieval import (
    get_embedding_model,
    retrieve_relevant_chunks,
    format_citations,
)

logger = logging.getLogger(__name__)

# Initialize MCP Server instance
mcp_server = MCPServer(
    name="PaperMind-RAG-MCP",
    version="1.0.0",
    instructions="Model Context Protocol server for RAG chunking, MongoDB storage, and thinking context retrieval."
)


# ---------------------------------------------------------------------------
# Direct Python Tool Implementations (Callable by Agents & Ingestion Pipelines)
# ---------------------------------------------------------------------------

def execute_mcp_chunking(
    doc_id: str,
    filename: str,
    pages: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    store_in_qdrant: bool = True
) -> Dict[str, Any]:
    """
    Core MCP Chunking Tool logic:
    1. Splits pages into semantic chunks respecting character boundaries and overlap.
    2. Persists structured chunk records into MongoDB.
    3. Indexes vectors and metadata into Qdrant Cloud.
    """
    config = get_config()
    db = get_rag_database()

    c_size = chunk_size or config.chunk_size
    c_overlap = chunk_overlap or config.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=c_size,
        chunk_overlap=c_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    raw_chunks = splitter.split_documents(pages)
    enriched_chunks: List[Document] = []

    for idx, chunk in enumerate(raw_chunks):
        page_num = chunk.metadata.get("page", chunk.metadata.get("page_number", 1))
        chunk_id = f"{doc_id}_chunk_{idx}"

        chunk.metadata.update({
            "doc_id": doc_id,
            "filename": filename,
            "page": page_num,
            "page_number": page_num,
            "chunk_index": idx,
            "chunk_id": chunk_id,
            "total_chunks": len(raw_chunks)
        })
        enriched_chunks.append(chunk)

    # 1. Persist to MongoDB
    saved_count_mongo = db.save_chunks(doc_id=doc_id, chunks=enriched_chunks)
    db.save_document(
        doc_id=doc_id,
        filename=filename,
        file_size_bytes=sum(len(p.page_content) for p in pages),
        total_pages=len(pages),
        total_chunks=len(enriched_chunks),
        metadata={"chunk_size": c_size, "chunk_overlap": c_overlap, "mcp_processed": True}
    )

    # 2. Vector Indexing to Qdrant Cloud
    qdrant_indexed_count = 0
    if store_in_qdrant and config.qdrant_url and config.qdrant_api_key:
        try:
            q_client = get_qdrant_client(config.qdrant_url, config.qdrant_api_key)
            embeddings = get_embedding_model(config.embedding_model_name)
            qdrant_indexed_count = index_chunks_to_qdrant(
                client=q_client,
                embeddings_model=embeddings,
                collection_name=config.qdrant_collection_name,
                chunks=enriched_chunks,
                doc_id=doc_id,
                overwrite_existing=True
            )
        except Exception as q_err:
            logger.warning(f"Qdrant indexing encountered an issue (chunks still safe in MongoDB): {q_err}")

    preview = [
        {
            "chunk_index": c.metadata["chunk_index"],
            "page": c.metadata["page"],
            "preview": c.page_content[:150] + ("..." if len(c.page_content) > 150 else "")
        }
        for c in enriched_chunks[:3]
    ]

    return {
        "doc_id": doc_id,
        "filename": filename,
        "total_chunks": len(enriched_chunks),
        "mongodb_saved_count": saved_count_mongo,
        "qdrant_indexed_count": qdrant_indexed_count,
        "storage_mode": db.connection_status["storage_mode"],
        "sample_previews": preview,
        "status": "success"
    }


def execute_mcp_retrieval(
    query: str,
    doc_id: Optional[str] = None,
    top_k: Optional[int] = None
) -> Dict[str, Any]:
    """
    Core MCP Retrieval Tool logic:
    Retrieves most relevant document passages scoped to doc_id for LLM thinking.
    Prioritizes Qdrant Cloud vector search and falls back gracefully to MongoDB text matching.
    """
    config = get_config()
    db = get_rag_database()
    k = top_k or config.top_k

    retrieved_docs: List[Document] = []
    retrieval_source = "Qdrant Cloud"

    # Attempt Qdrant Cloud retrieval
    if config.qdrant_url and config.qdrant_api_key:
        try:
            q_client = get_qdrant_client(config.qdrant_url, config.qdrant_api_key)
            embeddings = get_embedding_model(config.embedding_model_name)
            retrieved_docs = retrieve_relevant_chunks(
                client=q_client,
                embeddings_model=embeddings,
                collection_name=config.qdrant_collection_name,
                query=query,
                doc_id=doc_id,
                top_k=k
            )
        except Exception as e:
            logger.warning(f"Qdrant retrieval failed, falling back to MongoDB search: {e}")
            retrieved_docs = []

    # Fallback to MongoDB stored chunks if Qdrant didn't return results
    if not retrieved_docs and doc_id:
        mongo_chunks = db.get_chunks(doc_id)
        if mongo_chunks:
            retrieval_source = "MongoDB Stored Chunks"
            # Basic keyword score ranking for fallback
            q_terms = [t.lower() for t in query.split() if len(t) > 2]
            scored = []
            for ch in mongo_chunks:
                text_lower = ch.get("content", "").lower()
                score = sum(1 for term in q_terms if term in text_lower)
                scored.append((score, ch))
            scored.sort(key=lambda x: x[0], reverse=True)
            for sc, ch in scored[:k]:
                doc = Document(
                    page_content=ch.get("content", ""),
                    metadata={
                        "doc_id": doc_id,
                        "page": ch.get("page", 1),
                        "page_number": ch.get("page_number", 1),
                        "similarity_score": min(0.95, 0.4 + (sc * 0.1)) if sc > 0 else 0.3
                    }
                )
                retrieved_docs.append(doc)

    citations = format_citations(retrieved_docs)
    passages = [
        {
            "passage_index": idx + 1,
            "page": doc.metadata.get("page", 1),
            "score": doc.metadata.get("similarity_score"),
            "content": doc.page_content
        }
        for idx, doc in enumerate(retrieved_docs)
    ]

    return {
        "query": query,
        "doc_id": doc_id,
        "retrieval_source": retrieval_source,
        "retrieved_count": len(retrieved_docs),
        "passages": passages,
        "citations": citations,
        "raw_docs": retrieved_docs
    }


# ---------------------------------------------------------------------------
# Registered MCP Server Tools (For standard MCP clients via stdio/SSE)
# ---------------------------------------------------------------------------

@mcp_server.tool(
    name="chunk_document_mcp",
    description="Splits document text into semantic overlapping chunks, writes to MongoDB, and registers vectors into Qdrant Cloud."
)
def chunk_document_mcp(
    doc_id: str,
    filename: str,
    pages_text: List[str],
    chunk_size: int = 1000,
    chunk_overlap: int = 400
) -> str:
    """MCP tool for chunking documents."""
    docs = [
        Document(page_content=text, metadata={"page": i + 1, "filename": filename, "doc_id": doc_id})
        for i, text in enumerate(pages_text)
    ]
    res = execute_mcp_chunking(
        doc_id=doc_id,
        filename=filename,
        pages=docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return json.dumps(res, indent=2)


@mcp_server.tool(
    name="retrieve_context_mcp",
    description="Retrieves the most semantically relevant text passages from Qdrant/MongoDB for a query, strictly scoped to doc_id."
)
def retrieve_context_mcp(
    query: str,
    doc_id: str,
    top_k: int = 4
) -> str:
    """MCP tool for retrieving grounded context."""
    res = execute_mcp_retrieval(query=query, doc_id=doc_id, top_k=top_k)
    # Exclude non-serializable raw_docs for MCP JSON response
    res_clean = {k: v for k, v in res.items() if k != "raw_docs"}
    return json.dumps(res_clean, indent=2)


@mcp_server.tool(
    name="inspect_chunks_mcp",
    description="Inspects chunks stored in MongoDB for a given document ID."
)
def inspect_chunks_mcp(doc_id: str, limit: int = 10) -> str:
    """MCP tool for inspecting chunks stored in MongoDB."""
    db = get_rag_database()
    chunks = db.get_chunks(doc_id)[:limit]
    res = {
        "doc_id": doc_id,
        "total_returned": len(chunks),
        "chunks": [
            {
                "chunk_id": ch.get("chunk_id"),
                "chunk_index": ch.get("chunk_index"),
                "page": ch.get("page"),
                "length": ch.get("length"),
                "excerpt": ch.get("content", "")[:200]
            }
            for ch in chunks
        ]
    }
    return json.dumps(res, indent=2)


@mcp_server.tool(
    name="get_document_metadata_mcp",
    description="Retrieves document details from MongoDB by doc_id."
)
def get_document_metadata_mcp(doc_id: str) -> str:
    """MCP tool for retrieving document metadata from MongoDB."""
    db = get_rag_database()
    doc = db.get_document(doc_id)
    if not doc:
        return json.dumps({"error": f"Document with ID '{doc_id}' not found."}, indent=2)
    return json.dumps(doc, indent=2)


if __name__ == "__main__":
    # Allows running the MCP server standalone via stdio (e.g. from Claude Desktop or Antigravity)
    import asyncio
    asyncio.run(mcp_server.run_stdio_async())
