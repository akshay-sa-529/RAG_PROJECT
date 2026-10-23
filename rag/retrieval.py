"""
Retrieval Module for Learn RAG.
Handles embedding generation with Hugging Face sentence-transformers/all-MiniLM-L6-v2,
similarity search against Qdrant Cloud with document-scoped filtering,
and source citation formatting.
"""

from typing import List, Optional, Dict, Any, Tuple
from langchain_core.documents import Document
from qdrant_client import QdrantClient, models


_EMBEDDING_MODEL_CACHE: Dict[str, Any] = {}


def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Any:
    """Returns a cached instance of the Hugging Face embedding model."""
    global _EMBEDDING_MODEL_CACHE
    if model_name not in _EMBEDDING_MODEL_CACHE:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            _EMBEDDING_MODEL_CACHE[model_name] = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            _EMBEDDING_MODEL_CACHE[model_name] = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
    return _EMBEDDING_MODEL_CACHE[model_name]


def get_vector_store(
    client: QdrantClient,
    collection_name: str,
    embeddings_model: Any
) -> Any:
    """Returns a LangChain Qdrant vector store connected to Qdrant Cloud."""
    try:
        from langchain_qdrant import QdrantVectorStore
        return QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings_model
        )
    except ImportError:
        from langchain_community.vectorstores import Qdrant
        return Qdrant(
            client=client,
            collection_name=collection_name,
            embeddings=embeddings_model
        )


def retrieve_relevant_chunks(
    client: QdrantClient,
    embeddings_model: Any,
    collection_name: str,
    query: str,
    doc_id: Optional[str] = None,
    top_k: int = 4
) -> List[Document]:
    """
    Performs similarity search in Qdrant Cloud using cosine distance.
    Filters by doc_id to guarantee chunks originate exclusively from the target document.
    """
    vector_store = get_vector_store(client, collection_name, embeddings_model)
    
    qdrant_filter: Optional[models.Filter] = None
    if doc_id:
        qdrant_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.doc_id",
                    match=models.MatchValue(value=doc_id)
                )
            ]
        )
        
    try:
        results_with_scores: List[Tuple[Document, float]] = vector_store.similarity_search_with_score(
            query=query,
            k=top_k,
            filter=qdrant_filter
        )
    except TypeError:
        results_with_scores = vector_store.similarity_search_with_score(
            query=query,
            k=top_k
        )
        if doc_id:
            results_with_scores = [
                (doc, score) for doc, score in results_with_scores
                if doc.metadata.get("doc_id") == doc_id
            ]

    retrieved_docs: List[Document] = []
    for doc, score in results_with_scores:
        doc.metadata["similarity_score"] = float(score)
        retrieved_docs.append(doc)
        
    return retrieved_docs


def format_citations(chunks: List[Document]) -> Dict[str, Any]:
    """Extracts, deduplicates, and formats page-number citations from retrieved chunks."""
    if not chunks:
        return {
            "citations_by_file": {},
            "citation_summary": "No citations available (no document chunks retrieved)",
            "source_details": []
        }
        
    file_pages_map: Dict[str, set] = {}
    source_details: List[Dict[str, Any]] = []
    
    for chunk in chunks:
        filename = chunk.metadata.get("filename", chunk.metadata.get("source", "Document.pdf"))
        page = chunk.metadata.get("page", chunk.metadata.get("page_number", 1))
        
        if filename not in file_pages_map:
            file_pages_map[filename] = set()
        file_pages_map[filename].add(page)
        
        source_details.append({
            "filename": filename,
            "page": page,
            "chunk_index": chunk.metadata.get("chunk_index", 0),
            "score": chunk.metadata.get("similarity_score"),
            "content_preview": chunk.page_content[:150] + ("..." if len(chunk.page_content) > 150 else "")
        })
        
    formatted_sources: List[str] = []
    sorted_map: Dict[str, List[int]] = {}
    
    for filename, pages in file_pages_map.items():
        sorted_pages = sorted(list(pages))
        sorted_map[filename] = sorted_pages
        
        if len(sorted_pages) == 1:
            formatted_sources.append(f"{filename} (Page {sorted_pages[0]})")
        else:
            pages_str = ", ".join(str(p) for p in sorted_pages)
            formatted_sources.append(f"{filename} (Pages {pages_str})")
            
    summary_string = " | ".join(formatted_sources)
    
    return {
        "citations_by_file": sorted_map,
        "citation_summary": summary_string,
        "source_details": source_details
    }
