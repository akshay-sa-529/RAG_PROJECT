"""
Document Ingestion Module for Learn RAG.
Handles PDF uploading, text extraction with PyPDFLoader,
page metadata preservation (1-indexed), chunking with RecursiveCharacterTextSplitter,
document hashing, and Qdrant Cloud vector indexing with duplicate prevention.
"""

import hashlib
import os
import tempfile
import uuid
from typing import List, Optional, Tuple, Dict, Any

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models


def compute_document_id(file_bytes: bytes, filename: str = "") -> str:
    """
    Computes a deterministic SHA-256 hash from PDF binary contents.
    Ensures identical files yield the exact same document ID.
    """
    hasher = hashlib.sha256()
    hasher.update(file_bytes)
    return hasher.hexdigest()[:16]


def load_pdf_from_bytes(file_bytes: bytes, filename: str) -> List[Document]:
    """
    Saves PDF bytes to a temporary file, extracts pages using LangChain's PyPDFLoader,
    preserves and converts 0-indexed page numbers to human-friendly 1-indexed numbers,
    and returns a list of Document objects.
    """
    if not file_bytes:
        raise ValueError(f"Uploaded file '{filename}' is empty (0 bytes).")

    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"rag_{uuid.uuid4().hex[:8]}_{filename}")
    
    try:
        with open(temp_path, "wb") as f:
            f.write(file_bytes)
            
        loader = PyPDFLoader(temp_path)
        raw_docs = loader.load()
        
        if not raw_docs:
            raise ValueError(
                f"Could not extract pages from '{filename}'. The file may be corrupted or password-protected."
            )
            
        processed_docs: List[Document] = []
        total_text_length = 0
        
        for doc in raw_docs:
            # PyPDFLoader produces 0-indexed 'page' in metadata (page 0 is page 1)
            raw_page = doc.metadata.get("page", 0)
            human_page = int(raw_page) + 1  # 1-indexed human-friendly page number
            
            clean_text = doc.page_content.strip()
            total_text_length += len(clean_text)
            
            normalized_metadata: Dict[str, Any] = {
                "source": filename,
                "filename": filename,
                "page": human_page,
                "page_number": human_page,
                "raw_page": raw_page,
                "total_pages": len(raw_docs)
            }
            
            processed_docs.append(
                Document(page_content=doc.page_content, metadata=normalized_metadata)
            )
            
        if total_text_length == 0:
            raise ValueError(
                f"PDF '{filename}' contains {len(raw_docs)} page(s) but no extractable text was found. "
                "This document may consist purely of scanned images without an OCR text layer."
            )
            
        return processed_docs
        
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def split_document_pages(
    documents: List[Document],
    doc_id: str,
    filename: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 400
) -> List[Document]:
    """
    Splits document pages into chunks using LangChain's RecursiveCharacterTextSplitter.
    Associates each chunk with doc_id, filename, 1-indexed page, and chunk_index.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    raw_chunks = splitter.split_documents(documents)
    enriched_chunks: List[Document] = []
    
    for idx, chunk in enumerate(raw_chunks):
        page_num = chunk.metadata.get("page", 1)
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
        
    return enriched_chunks


def get_qdrant_client(qdrant_url: str, qdrant_api_key: str) -> QdrantClient:
    """Instantiates and returns a QdrantClient connected to Qdrant Cloud."""
    return QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=30)


def ensure_collection_exists(
    client: QdrantClient,
    collection_name: str,
    vector_dim: int = 384
) -> None:
    """
    Ensures that the target collection exists in Qdrant Cloud with cosine distance.
    Creates collection and payload index on metadata.doc_id if missing.
    """
    collections = client.get_collections().collections
    existing_names = [c.name for c in collections]
    
    if collection_name not in existing_names:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_dim,
                distance=models.Distance.COSINE
            )
        )
        
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="metadata.doc_id",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
    except Exception:
        pass


def is_document_indexed(
    client: QdrantClient,
    collection_name: str,
    doc_id: str
) -> Tuple[bool, int]:
    """Checks if a document with doc_id is already indexed in the collection."""
    try:
        ensure_collection_exists(client, collection_name)
        count_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.doc_id",
                    match=models.MatchValue(value=doc_id)
                )
            ]
        )
        res = client.count(collection_name=collection_name, count_filter=count_filter)
        return res.count > 0, res.count
    except Exception:
        return False, 0


def delete_document_vectors(
    client: QdrantClient,
    collection_name: str,
    doc_id: str
) -> int:
    """Removes all vector points associated with a specific document ID from Qdrant Cloud."""
    delete_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.doc_id",
                match=models.MatchValue(value=doc_id)
            )
        ]
    )
    
    try:
        count_res = client.count(collection_name=collection_name, count_filter=delete_filter)
        deleted_count = count_res.count
    except Exception:
        deleted_count = 0
        
    client.delete(
        collection_name=collection_name,
        points_selector=models.FilterSelector(filter=delete_filter)
    )
    return deleted_count


def index_chunks_to_qdrant(
    client: QdrantClient,
    embeddings_model: Any,
    collection_name: str,
    chunks: List[Document],
    doc_id: str,
    overwrite_existing: bool = True
) -> int:
    """Embeds document chunks and stores them into the Qdrant Cloud collection."""
    if not chunks:
        return 0
        
    ensure_collection_exists(client, collection_name, vector_dim=384)
    
    if overwrite_existing:
        delete_document_vectors(client, collection_name, doc_id)
        
    try:
        from langchain_qdrant import QdrantVectorStore
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings_model
        )
    except ImportError:
        from langchain_community.vectorstores import Qdrant
        vector_store = Qdrant(
            client=client,
            collection_name=collection_name,
            embeddings=embeddings_model
        )
        
    deterministic_ids = [
        str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.metadata["chunk_id"]))
        for chunk in chunks
    ]
    
    vector_store.add_documents(chunks, ids=deterministic_ids)
    return len(chunks)
