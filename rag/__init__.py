"""
Learn RAG package initialization.
Exports key ingestion, retrieval, generation, and configuration functions.
"""

from rag.config import AppConfig, get_config, validate_config
from rag.ingestion import (
    compute_document_id,
    load_pdf_from_bytes,
    split_document_pages,
    get_qdrant_client,
    ensure_collection_exists,
    is_document_indexed,
    delete_document_vectors,
    index_chunks_to_qdrant,
)
from rag.retrieval import (
    get_embedding_model,
    retrieve_relevant_chunks,
    format_citations,
)
from rag.generation import (
    generate_grounded_answer,
    build_context_string,
)
from rag.db import (
    get_rag_database,
    MongoRAGDatabase,
)
from rag.mcp_tools import (
    mcp_server,
    execute_mcp_chunking,
    execute_mcp_retrieval,
)

__all__ = [
    "AppConfig",
    "get_config",
    "validate_config",
    "compute_document_id",
    "load_pdf_from_bytes",
    "split_document_pages",
    "get_qdrant_client",
    "ensure_collection_exists",
    "is_document_indexed",
    "delete_document_vectors",
    "index_chunks_to_qdrant",
    "get_embedding_model",
    "retrieve_relevant_chunks",
    "format_citations",
    "generate_grounded_answer",
    "build_context_string",
    "get_rag_database",
    "MongoRAGDatabase",
    "mcp_server",
    "execute_mcp_chunking",
    "execute_mcp_retrieval",
]
