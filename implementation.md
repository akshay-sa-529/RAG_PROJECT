# Complete Step-by-Step Implementation Guide: Building a Cloud-Native RAG Application

This guide provides a comprehensive, step-by-step walkthrough for building a production-grade, beginner-friendly **Retrieval-Augmented Generation (RAG)** application from scratch. 

The application allows users to upload any text-based PDF, index its contents into a managed **Qdrant Cloud** vector database, and ask natural language questions. Answers are generated using **OpenRouter** (`recursive/flint-v1.0:free`), grounded **exclusively** in the uploaded document, and accompanied by **exact page-number citations** and an interactive chunk inspection panel.

---

## 1. Project Overview & How RAG Works

Large Language Models (LLMs) are trained on massive public datasets, but they do not know about private, recent, or domain-specific documents (such as your school syllabus, research papers, or company handbooks). Fine-tuning an LLM on every document is expensive, slow, and prone to hallucinations.

**Retrieval-Augmented Generation (RAG)** solves this by retrieving relevant text excerpts from a document at query time and injecting them into the LLM's prompt as trusted context.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                             THE RAG WORKFLOW                                │
└─────────────────────────────────────────────────────────────────────────────┘

[PHASE A: DOCUMENT INGESTION & CLOUD INDEXING]
 1. Upload PDF ──► 2. PyPDFLoader ──► 3. RecursiveTextSplitter ──► 4. Embeddings ──► 5. Qdrant Cloud
   (Streamlit)      (Extract Pages     (1,000 chars size,           (all-MiniLM-L6      (learning_rag
                     + 1-idx Page)      400 chars overlap)           384 dimensions)     Collection)

[PHASE B: RETRIEVAL & GROUNDED ANSWER GENERATION]
 6. User Query ──► 7. Vectorize Query ──► 8. Cosine Similarity ──► 9. OpenRouter LLM ──► 10. Answer +
   ("What is...")   (all-MiniLM-L6)          (Scoped to doc_id)     (recursive/flint)        Page Citations
```

### The 4 Core Stages Explained:
1. **Ingestion & Page Extraction:** Read the PDF page-by-page. LangChain's `PyPDFLoader` numbers pages starting at `0` (page 0 is page 1). We convert this to human-friendly 1-indexed numbers (`page + 1`) and compute a unique SHA-256 hash (`doc_id`) of the file bytes.
2. **Chunking:** Documents are too large to embed as single blocks. We split pages into overlapping chunks of **1,000 characters** with **400 characters of overlap** using `RecursiveCharacterTextSplitter`. Overlap ensures that sentences split across chunk boundaries do not lose their semantic context.
3. **Embedding Generation:** Each text chunk is converted into a 384-dimensional dense numerical vector using Hugging Face's `sentence-transformers/all-MiniLM-L6-v2`. Semantic meaning is captured in geometric space: chunks with similar meanings point in similar directions.
4. **Cloud Storage & Filtered Retrieval:** Vectors and metadata (`doc_id`, `filename`, `page`, `chunk_index`) are stored in **Qdrant Cloud**. When a question is asked, it is converted into a vector, and Qdrant computes **Cosine Similarity** to find the top matching chunks. The search is filtered by `doc_id` so that multiple uploaded documents never contaminate each other.
5. **Grounded Answer Generation:** The retrieved passages are inserted into a prompt sent to OpenRouter (`recursive/flint-v1.0:free`). A strict grounding prompt instructs the model to answer using *only* the provided excerpts. If the information is missing, the model responds with: `"The provided document does not contain information to answer this question."`

---

## 2. Technology Stack & Component Roles

| Technology | Role | Why It Was Chosen |
|---|---|---|
| **Python 3.10+** | Core programming language | Industry standard for AI, LangChain, and data science |
| **Streamlit (`streamlit`)** | Web application framework | Enables rapid creation of reactive, interactive web UIs in pure Python |
| **LangChain Core (`langchain-core`)** | Document & prompt abstractions | Provides standardized schemas for `Document`, prompts, and messages |
| **PyPDFLoader (`langchain-community`, `pypdf`)** | PDF text extraction | Extracts clean digital text and preserves per-page metadata |
| **RecursiveCharacterTextSplitter** | Text chunking | Splits text respecting paragraph, sentence, and word boundaries |
| **Hugging Face (`sentence-transformers`)** | Local embedding model | `all-MiniLM-L6-v2` runs fast on CPU, produces 384-dim normalized vectors |
| **Qdrant Cloud (`qdrant-client`, `langchain-qdrant`)** | Cloud vector database | Managed, high-speed vector index with metadata filtering. **No Docker needed.** |
| **OpenRouter (`langchain-openai`)** | LLM API gateway | Provides OpenAI-compatible endpoint for models like `recursive/flint-v1.0:free` |
| **python-dotenv** | Secret management | Loads API keys from `.env` without hardcoding credentials |
| **pytest** | Automated test suite | Verifies chunking, metadata preservation, and prompt grounding |

---

## 3. Prerequisites & Required Accounts

Before writing code, set up the following free accounts:

### 1. Python 3.10+
- Download and install Python from [python.org](https://www.python.org/downloads/).
- During installation on Windows, ensure you check **"Add Python to PATH"**.

### 2. Qdrant Cloud Account (Free Tier)
1. Sign up at [https://cloud.qdrant.io/](https://cloud.qdrant.io/).
2. Click **Create Cluster**, choose the **Free Tier (1 GB RAM / 0.5 vCPU)**, and select a region (e.g., AWS `us-east-1` or `eu-west-2`).
3. Once created, copy the **Cluster URL** (e.g., `https://your-cluster-id.region.aws.cloud.qdrant.io:6333` or `https://your-cluster-id.region.aws.cloud.qdrant.io`).
4. Click **Data Access Control** > **API Keys** > **Create API Key**. Copy your API key.

### 3. OpenRouter Account
1. Sign up at [https://openrouter.ai/](https://openrouter.ai/).
2. Go to **Keys** ([https://openrouter.ai/keys](https://openrouter.ai/keys)) and click **Create Key**.
3. Name it `learn-rag-workshop` and copy the generated secret key (starts with `sk-or-v1-...`).
4. The project uses the free-tier model `recursive/flint-v1.0:free`, so credit purchase is not required to get started.

---

## 4. Project Folder Structure & File Explanations

Organize your workspace as follows:

```text
learn-rag/
├── app.py                      # Streamlit UI & interactive chat workflow
├── rag/                        # Core modular RAG package
│   ├── __init__.py             # Public package exports
│   ├── config.py               # Environment configuration and validation
│   ├── ingestion.py            # PDF loading, SHA-256 hashing, chunking, Qdrant upload
│   ├── retrieval.py            # Embedding cache, scoped similarity search, citations
│   └── generation.py           # Grounded prompt engineering and OpenRouter LLM call
├── tests/                      # Automated test suite
│   ├── __init__.py             # Test package initializer
│   ├── test_rag_pipeline.py    # Unit tests for chunking, citations, and grounding
│   └── validate_cloud_setup.py # End-to-end live cloud infrastructure test
├── requirements.txt            # Python package dependencies
├── .env.example                # Template for environment configuration
├── .gitignore                  # Git ignore rules for secrets and virtualenvs
├── README.md                   # Project summary and workshop guide
└── implementation.md           # This comprehensive recreation guide
```

### Explanation of Each File:

- **`app.py`**: The Streamlit user interface. Manages PDF file uploads, displays step-by-step ingestion status indicators, maintains chat session history, renders grounded answers with source citation badges, and provides an expandable chunk inspection panel.
- **`rag/config.py`**: Reads configuration settings from `.env`, provides the `AppConfig` dataclass, and validates that cloud credentials are set while explicitly rejecting `localhost` endpoints.
- **`rag/ingestion.py`**: Handles PDF binary processing, computes a unique SHA-256 `doc_id`, loads pages via `PyPDFLoader`, normalizes 0-indexed page numbers to 1-indexed numbers, chunks text using `RecursiveCharacterTextSplitter` (1,000 characters, 400 overlap), and uploads vectors into Qdrant Cloud.
- **`rag/retrieval.py`**: Loads and caches the Hugging Face `all-MiniLM-L6-v2` embedding model, executes similarity searches on Qdrant Cloud filtered by `metadata.doc_id` to prevent multi-document mixing, and extracts deduplicated page citations.
- **`rag/generation.py`**: Prepares context blocks, enforces strict grounding instructions via `GROUNDED_SYSTEM_PROMPT`, calls OpenRouter (`recursive/flint-v1.0:free`) at `temperature=0.2`, and returns answers bundled with citations.
- **`rag/__init__.py`**: Exposes the primary functions of the `rag` package through a clean, unified public interface.
- **`requirements.txt`**: Specifies all required Python libraries with compatible version constraints.
- **`.env.example`**: A safe configuration template showing required environment variables without real secrets.
- **`.env`**: The local configuration file containing your actual Qdrant Cloud URL, API Key, and OpenRouter API Key (never committed to Git).
- **`.gitignore`**: Protects your secret keys, virtual environments, and bytecode caches from version control.
- **`tests/test_rag_pipeline.py`**: Automated test suite verifying document hashing, chunking boundaries, 1-indexed page conversion, citation formatting, and anti-hallucination refusal.
- **`tests/validate_cloud_setup.py`**: Live end-to-end cloud validation script testing Qdrant Cloud and OpenRouter with a programmatic test document.
- **`README.md`**: High-level overview, architecture diagrams, and quickstart workshop demonstration steps.
- **`implementation.md`**: Complete, self-contained implementation manual for reproducing the application from an empty directory.

---

## 5. Python Virtual Environment & Dependency Setup

Run the following commands in **Windows PowerShell**:

### Step 5.1: Create & Activate Virtual Environment
```powershell
# Open terminal in project directory
python -m venv venv

# Activate virtual environment on Windows
.\venv\Scripts\Activate.ps1
```

*(If you receive an execution policy restriction error in PowerShell, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and re-run the activate command).*

### Step 5.2: Create `requirements.txt`
Create `requirements.txt` with the exact package dependencies:

```text
# Core Streamlit UI
streamlit>=1.35.0

# LangChain Ecosystem
langchain>=0.2.0,<0.4.0
langchain-community>=0.2.0
langchain-core>=0.2.0
langchain-openai>=0.1.7
langchain-qdrant>=0.1.0
langchain-huggingface>=0.0.3

# Embeddings & Vector Search
sentence-transformers>=2.6.0
qdrant-client>=1.9.0

# PDF Processing
pypdf>=4.2.0

# Configuration & Utilities
python-dotenv>=1.0.1

# Testing
pytest>=7.0.0
```

Install all dependencies:
```powershell
pip install -r requirements.txt
```

---

## 6. Environment Configuration & Security

### Step 6.1: Create `.gitignore`
Prevent sensitive keys and virtual environments from accidentally being pushed to GitHub:

```gitignore
# Environment secrets
.env
*.env
!.env.example
.env.local

# Python virtual environment
venv/
.venv/
env/

# Python cache & test artifacts
__pycache__/
*.py[cod]
.pytest_cache/

# Operating system files
.DS_Store
Thumbs.db
```

### Step 6.2: Create `.env.example`
Provide a safe, sanitized template for students:

```env
# =====================================================================
# Learn RAG: Environment Configuration Template
# =====================================================================
# Copy this file to '.env' and fill in your actual credentials.
# Never commit your '.env' file containing real secrets to version control!

# Qdrant Cloud Vector Database Configuration
QDRANT_URL=https://your-cluster-id.us-east-1-0.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key_here
QDRANT_COLLECTION_NAME=learning_rag

# OpenRouter LLM Configuration
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=recursive/flint-v1.0:free
```

### Step 6.3: Create `.env`
Create your actual `.env` file and populate it with your real keys:

```env
QDRANT_URL=https://your-actual-cluster-url.aws.cloud.qdrant.io
QDRANT_API_KEY=your_actual_qdrant_api_key
QDRANT_COLLECTION_NAME=learning_rag
OPENROUTER_API_KEY=sk-or-v1-your_actual_openrouter_key
OPENROUTER_MODEL=recursive/flint-v1.0:free
```

---

## 7. Step-by-Step Code Implementation

### Step 7.1: Configuration Module (`rag/config.py`)
This module reads `.env`, validates that credentials are set, and explicitly blocks `localhost` to guarantee a cloud-native vector store.

```python
"""
Configuration module for Learn RAG application.
Loads and validates settings from environment variables and .env file.
"""

import os
from dataclasses import dataclass
from typing import List, Tuple
from dotenv import load_dotenv

# Load variables from .env file if present
load_dotenv(override=False)


@dataclass(frozen=True)
class AppConfig:
    """Holds application configuration parameters."""
    
    # Qdrant Cloud configuration
    qdrant_url: str = os.getenv("QDRANT_URL", "").strip()
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "").strip()
    qdrant_collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "learning_rag").strip()
    
    # OpenRouter API configuration
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "").strip()
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct").strip()
    
    # RAG pipeline constants per workshop specifications
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 400
    top_k: int = 4


def get_config() -> AppConfig:
    """Returns the current application configuration."""
    return AppConfig()


def validate_config(config: AppConfig) -> Tuple[bool, List[str]]:
    """
    Validates whether the necessary environment configuration is present and valid.
    
    Returns:
        Tuple[bool, List[str]]: (is_valid, list of human-friendly error messages)
    """
    issues: List[str] = []
    
    # Validate Qdrant URL
    if not config.qdrant_url:
        issues.append("QDRANT_URL is not set. Please provide your Qdrant Cloud cluster URL.")
    elif "localhost" in config.qdrant_url or "127.0.0.1" in config.qdrant_url:
        issues.append("QDRANT_URL points to localhost. Per workshop requirements, a cloud Qdrant cluster is required.")
    elif not (config.qdrant_url.startswith("https://") or config.qdrant_url.startswith("http://")):
        issues.append("QDRANT_URL must start with 'https://' or 'http://'.")
    elif "your-cluster-id" in config.qdrant_url:
        issues.append("QDRANT_URL contains placeholder value from .env.example. Replace it with your actual cluster URL.")
        
    # Validate Qdrant API Key
    if not config.qdrant_api_key:
        issues.append("QDRANT_API_KEY is not set. Please provide your Qdrant Cloud API key.")
    elif "your_qdrant_api_key_here" in config.qdrant_api_key:
        issues.append("QDRANT_API_KEY contains placeholder text. Replace it with your actual key.")
        
    # Validate OpenRouter API Key
    if not config.openrouter_api_key:
        issues.append("OPENROUTER_API_KEY is not set. Please provide your OpenRouter API key.")
    elif "your_openrouter_api_key_here" in config.openrouter_api_key:
        issues.append("OPENROUTER_API_KEY contains placeholder text. Replace it with your actual key.")
        
    # Validate Model name
    if not config.openrouter_model:
        issues.append("OPENROUTER_MODEL is empty. Defaulting to 'mistralai/mistral-7b-instruct'.")

    return len(issues) == 0, issues
```

---

### Step 7.2: Ingestion Module (`rag/ingestion.py`)
Handles PDF upload bytes, SHA-256 document hashing, page extraction via `PyPDFLoader`, conversion to 1-indexed page numbers, 1000/400 chunking, and Qdrant Cloud collection initialization.

```python
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
```

---

### Step 7.3: Retrieval Module (`rag/retrieval.py`)
Loads and caches the Hugging Face embedding model, performs similarity searches in Qdrant Cloud scoped to the active `doc_id`, and formats page citations.

```python
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
```

---

### Step 7.4: Generation Module (`rag/generation.py`)
Constructs the grounded prompt, communicates with OpenRouter, handles fallback gracefully, and extracts clean answers with citations.

```python
"""
Generation Module for Learn RAG.
Connects to OpenRouter API (recursive/flint-v1.0:free by default),
enforces strict context grounding, handles fallback when information is missing,
and bundles answers with source citations.
"""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from rag.retrieval import format_citations


GROUNDED_SYSTEM_PROMPT = """You are an accurate, strictly grounded AI teaching assistant for a technical RAG workshop.
Your mission is to answer questions using EXCLUSIVELY the provided context passages from the uploaded document.

CRITICAL RULES:
1. Answer the user's question using the provided document context passages. Extract, summarize, and synthesize any relevant details, explanations, features, or uses explicitly present in the text.
2. Do NOT use outside knowledge, prior assumptions, or extrapolate beyond what is explicitly stated in the context.
3. If the provided context contains NO relevant information or mentions nothing about the subject of the question, reply with:
   "The provided document does not contain information to answer this question."
4. Keep the answer direct, factual, and well-explained for students. Do not invent or hallucinate facts or citations."""


def build_context_string(chunks: List[Document]) -> str:
    """Formats retrieved chunks into a clearly labeled context block with page indicators."""
    if not chunks:
        return ""
        
    formatted_parts: List[str] = []
    for idx, chunk in enumerate(chunks, 1):
        filename = chunk.metadata.get("filename", chunk.metadata.get("source", "Document.pdf"))
        page = chunk.metadata.get("page", chunk.metadata.get("page_number", 1))
        content = chunk.page_content.strip()
        
        part = f"--- [Passage {idx} | Source: {filename} | Page {page}] ---\n{content}"
        formatted_parts.append(part)
        
    return "\n\n".join(formatted_parts)


def get_llm_client(
    openrouter_api_key: str,
    model_name: str = "mistralai/mistral-7b-instruct",
    temperature: float = 0.2
) -> ChatOpenAI:
    """Creates and returns a LangChain ChatOpenAI instance configured for OpenRouter."""
    return ChatOpenAI(
        model=model_name,
        openai_api_key=openrouter_api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=temperature,
        max_tokens=1024,
        default_headers={
            "HTTP-Referer": "https://github.com/learn-rag-workshop",
            "X-Title": "Learn RAG Workshop"
        }
    )


def generate_grounded_answer(
    query: str,
    retrieved_chunks: List[Document],
    openrouter_api_key: str,
    model_name: str = "mistralai/mistral-7b-instruct"
) -> Dict[str, Any]:
    """Sends retrieved document context and question to the OpenRouter chat model."""
    citations = format_citations(retrieved_chunks)
    
    if not retrieved_chunks:
        return {
            "answer": "The provided document does not contain information to answer this question.",
            "citations": citations,
            "chunks": [],
            "model_used": model_name,
            "grounded": False
        }
        
    context_text = build_context_string(retrieved_chunks)
    
    user_prompt = f"""DOCUMENT CONTEXT:
{context_text}

USER QUESTION:
{query}

ANSWER (grounded strictly in the document context above):"""

    llm = get_llm_client(openrouter_api_key=openrouter_api_key, model_name=model_name)
    
    messages = [
        SystemMessage(content=GROUNDED_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ]
    
    try:
        response = llm.invoke(messages)
    except Exception as e:
        fallback_model = "meta-llama/llama-3.1-8b-instruct"
        if ("404" in str(e) or "not available" in str(e).lower()) and model_name != fallback_model:
            fallback_llm = get_llm_client(openrouter_api_key=openrouter_api_key, model_name=fallback_model, temperature=0.2)
            response = fallback_llm.invoke(messages)
            model_name = fallback_model
        else:
            raise e
            
    answer_text = response.content.strip()
    
    is_not_found = (
        "does not contain information" in answer_text.lower()
        or "not found in the provided" in answer_text.lower()
        or "context does not mention" in answer_text.lower()
    )
    
    return {
        "answer": answer_text,
        "citations": citations,
        "chunks": retrieved_chunks,
        "model_used": model_name,
        "grounded": not is_not_found
    }
```

---

### Step 7.5: Package Initialization (`rag/__init__.py`)

```python
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
]
```

---

### Step 7.6: Streamlit Application (`app.py`)
The complete Streamlit interface featuring live progress status, infrastructure health checks, active document management, source citation cards, and an educational diagnostics expander.

```python
"""
Learn RAG — Chat with Your PDF
================================
Interactive Streamlit demonstration application for Retrieval-Augmented Generation (RAG).
Uses Qdrant Cloud as the vector database, Hugging Face sentence-transformers for embeddings,
and OpenRouter (recursive/flint-v1.0:free) for strictly grounded answer generation.
"""

import os
import streamlit as st

from rag.config import get_config, validate_config
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
from rag.generation import generate_grounded_answer


st.set_page_config(
    page_title="Learn RAG — Chat with Your PDF",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .citation-card {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "active_document" not in st.session_state:
    st.session_state.active_document = None

config = get_config()
is_config_valid, config_issues = validate_config(config)

with st.sidebar:
    st.title("📚 Learn RAG")
    st.markdown("**Workshop Demonstration App**")
    st.markdown("---")
    
    with st.expander("💡 How RAG Works (Workshop Guide)", expanded=False):
        st.markdown(
            """
            **The RAG Pipeline in 4 Core Stages:**
            1. **📄 Ingestion & Chunking**: `PyPDFLoader` preserves 1-indexed pages; `RecursiveCharacterTextSplitter` chunks text (1,000 size, 400 overlap).
            2. **🧠 Dense Embeddings**: `all-MiniLM-L6-v2` converts chunks into 384-dimensional vectors.
            3. **☁️ Qdrant Cloud**: Chunks, metadata (`doc_id`, `page`), and vectors are stored in cloud collection `learning_rag`.
            4. **🔍 Retrieval & Generation**: Cosine similarity retrieves top chunks, and OpenRouter answers grounded strictly in the context.
            """
        )
    
    st.markdown("### ☁️ Infrastructure Status")
    if config.qdrant_url and "localhost" not in config.qdrant_url and config.qdrant_api_key:
        cluster_host = config.qdrant_url.split("//")[-1].split(":")[0]
        st.success(f"**Qdrant Cloud:** Connected  \n`{cluster_host}`")
    else:
        st.error("**Qdrant Cloud:** Missing or Invalid Configuration")
        
    if config.openrouter_api_key:
        st.success(f"**LLM:** `{config.openrouter_model}`")
    else:
        st.error("**OpenRouter API:** Missing Key")
        
    st.info(f"**Qdrant Collection:** `{config.qdrant_collection_name}`")
    
    if not is_config_valid:
        with st.expander("⚠️ Configuration Checklist", expanded=True):
            for issue in config_issues:
                st.write(f"• {issue}")
            st.caption("Edit `.env` in the project root to resolve these.")

    st.markdown("---")
    
    st.markdown("### 📄 Active Document")
    if st.session_state.active_document:
        doc_info = st.session_state.active_document
        st.markdown(f"**File:** `{doc_info['filename']}`")
        st.markdown(f"**Doc ID:** `{doc_info['doc_id']}`")
        st.markdown(f"**Pages:** {doc_info.get('total_pages', 'N/A')} | **Chunks:** {doc_info.get('total_chunks', 'N/A')}")
        
        col_del, col_switch = st.columns(2)
        with col_del:
            if st.button("🗑️ Delete from Cloud", use_container_width=True):
                with st.spinner("Deleting vectors from Qdrant Cloud..."):
                    try:
                        client = get_qdrant_client(config.qdrant_url, config.qdrant_api_key)
                        del_count = delete_document_vectors(
                            client, config.qdrant_collection_name, doc_info["doc_id"]
                        )
                        st.session_state.active_document = None
                        st.session_state.messages = []
                        st.success(f"Removed {del_count} vectors from Qdrant Cloud!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to delete vectors: {e}")
        with col_switch:
            if st.button("🔄 Change PDF", use_container_width=True):
                st.session_state.active_document = None
                st.rerun()
    else:
        st.write("No PDF currently active. Upload a document below.")

    show_diagnostics = st.checkbox(
        "🩺 Show Retrieval Diagnostics",
        value=True,
        help="Displays retrieved chunk text, source page numbers, and cosine similarity scores."
    )
    
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.markdown('<div class="main-header">Learn RAG — Chat with Your PDF</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">A complete, grounded question-answering system using Qdrant Cloud and OpenRouter.</div>',
    unsafe_allow_html=True,
)

if not is_config_valid:
    st.warning("⚠️ **Application Setup Required:** Please configure your credentials in `.env` before proceeding.")
    with st.expander("📋 View Missing Configuration Details"):
        for issue in config_issues:
            st.error(issue)
    st.stop()

st.markdown("### 1. Document Upload & Ingestion")

uploaded_file = st.file_uploader(
    "Choose a PDF document to index into Qdrant Cloud:",
    type=["pdf"],
    help="Upload any text-based PDF (e.g. documentation, lecture notes, textbook chapters).",
)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name
    doc_id = compute_document_id(file_bytes, filename)
    
    try:
        qdrant_client = get_qdrant_client(config.qdrant_url, config.qdrant_api_key)
        already_indexed, existing_chunks = is_document_indexed(
            qdrant_client, config.qdrant_collection_name, doc_id
        )
    except Exception as e:
        st.error(f"❌ Could not connect to Qdrant Cloud: {e}. Check your `QDRANT_URL` and `QDRANT_API_KEY`.")
        already_indexed, existing_chunks = False, 0

    col_info, col_action = st.columns([3, 1])
    
    with col_info:
        if already_indexed:
            st.success(
                f"✅ **Document Already Indexed!**  \n"
                f"`{filename}` is already present in Qdrant Cloud collection `{config.qdrant_collection_name}` "
                f"({existing_chunks} chunks stored). Document ID: `{doc_id}`."
            )
        else:
            st.info(
                f"📄 **Ready to Index:** `{filename}` ({len(file_bytes) / 1024:.1f} KB). "
                f"Computed Document ID: `{doc_id}`."
            )

    with col_action:
        button_label = "🔄 Re-Index Document" if already_indexed else "🚀 Index to Cloud"
        start_indexing = st.button(button_label, use_container_width=True, type="primary")

    if already_indexed and (not st.session_state.active_document or st.session_state.active_document.get("doc_id") != doc_id):
        if st.button("👉 Use this indexed document for chat", use_container_width=False):
            st.session_state.active_document = {
                "doc_id": doc_id,
                "filename": filename,
                "total_chunks": existing_chunks,
            }
            st.rerun()

    if start_indexing:
        with st.status("Processing PDF through RAG Pipeline...", expanded=True) as status:
            try:
                status.write("📄 **Step 1/4:** Loading PDF and preserving 1-indexed page metadata...")
                pages = load_pdf_from_bytes(file_bytes, filename)
                status.write(f"✓ Extracted {len(pages)} page(s) with human-friendly page numbers.")

                status.write(
                    f"✂️ **Step 2/4:** Splitting text (Chunk size: {config.chunk_size}, "
                    f"Overlap: {config.chunk_overlap} chars)..."
                )
                chunks = split_document_pages(
                    documents=pages,
                    doc_id=doc_id,
                    filename=filename,
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.chunk_overlap,
                )
                status.write(f"✓ Created {len(chunks)} chunks with doc_id and page metadata attached.")

                status.write(
                    f"🧠 **Step 3/4:** Loading embedding model (`{config.embedding_model_name}`)..."
                )
                embeddings = get_embedding_model(config.embedding_model_name)
                status.write("✓ Embedding model ready (384-dimensional vectors).")

                status.write(
                    f"☁️ **Step 4/4:** Indexing vectors and metadata into Qdrant Cloud "
                    f"collection `{config.qdrant_collection_name}`..."
                )
                indexed_count = index_chunks_to_qdrant(
                    client=qdrant_client,
                    embeddings_model=embeddings,
                    collection_name=config.qdrant_collection_name,
                    chunks=chunks,
                    doc_id=doc_id,
                    overwrite_existing=True,
                )
                
                st.session_state.active_document = {
                    "doc_id": doc_id,
                    "filename": filename,
                    "total_pages": len(pages),
                    "total_chunks": indexed_count,
                }
                
                status.update(
                    label=f"🎉 Successfully indexed '{filename}' ({indexed_count} chunks) in Qdrant Cloud!",
                    state="complete",
                    expanded=False,
                )
                st.success(
                    f"Ready! You can now ask questions about `{filename}` in the chat section below."
                )
                st.rerun()

            except Exception as exc:
                status.update(label=f"❌ Ingestion failed: {exc}", state="error")
                st.error(f"Error during ingestion: {exc}")

st.markdown("---")
st.markdown("### 2. Grounded Question Answering")

if not st.session_state.active_document:
    st.info("👆 Please upload and index a PDF above to begin asking questions.")
else:
    active_doc = st.session_state.active_document
    st.caption(
        f"Active Document: **{active_doc['filename']}** | Scoped Document ID: `{active_doc['doc_id']}` | "
        f"Vectors Stored in Qdrant Cloud: `{config.qdrant_collection_name}`"
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            if msg["role"] == "assistant":
                citations = msg.get("citations", {})
                citation_summary = citations.get("citation_summary", "")
                
                if citation_summary and "No citations" not in citation_summary:
                    st.markdown(
                        f"""
                        <div class="citation-card">
                            <strong>📄 Source Citations:</strong> {citation_summary}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                
                chunks = msg.get("chunks", [])
                if chunks:
                    with st.expander(f"🔍 Inspect Retrieved Chunks ({len(chunks)} passages from Qdrant Cloud)", expanded=show_diagnostics):
                        for i, ch in enumerate(chunks, 1):
                            pg = ch.get("page", 1)
                            score = ch.get("score")
                            score_text = f" | Cosine Similarity Score: {score:.4f}" if score is not None else ""
                            char_len = len(ch.get("content", ""))
                            st.markdown(f"**Chunk {i} (Source Page: {pg}{score_text} | Length: {char_len} chars):**")
                            st.code(ch.get("content", ""), language="text")
                            st.divider()

    user_query = st.chat_input("Ask a question about the uploaded PDF document...")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching Qdrant Cloud vectors & generating grounded answer..."):
                try:
                    qdrant_client = get_qdrant_client(config.qdrant_url, config.qdrant_api_key)
                    embeddings = get_embedding_model(config.embedding_model_name)
                    
                    retrieved_docs = retrieve_relevant_chunks(
                        client=qdrant_client,
                        embeddings_model=embeddings,
                        collection_name=config.qdrant_collection_name,
                        query=user_query,
                        doc_id=active_doc["doc_id"],
                        top_k=config.top_k,
                    )
                    
                    result = generate_grounded_answer(
                        query=user_query,
                        retrieved_chunks=retrieved_docs,
                        openrouter_api_key=config.openrouter_api_key,
                        model_name=config.openrouter_model,
                    )
                    
                    answer_text = result["answer"]
                    citations = result["citations"]
                    citation_summary = citations.get("citation_summary", "")
                    
                    st.markdown(answer_text)
                    
                    if citation_summary and "No citations" not in citation_summary:
                        st.markdown(
                            f"""
                            <div class="citation-card">
                                <strong>📄 Source Citations:</strong> {citation_summary}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        
                    chunk_details = [
                        {
                            "page": doc.metadata.get("page", 1),
                            "score": doc.metadata.get("similarity_score"),
                            "content": doc.page_content,
                        }
                        for doc in retrieved_docs
                    ]
                    
                    if chunk_details:
                        with st.expander(f"🔍 Inspect Retrieved Chunks ({len(chunk_details)} passages from Qdrant Cloud)", expanded=show_diagnostics):
                            for i, ch in enumerate(chunk_details, 1):
                                pg = ch["page"]
                                sc = ch["score"]
                                sc_text = f" | Cosine Similarity Score: {sc:.4f}" if sc is not None else ""
                                char_len = len(ch["content"])
                                st.markdown(f"**Chunk {i} (Source Page: {pg}{sc_text} | Length: {char_len} chars):**")
                                st.code(ch["content"], language="text")
                                st.divider()

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer_text,
                            "citations": citations,
                            "chunks": chunk_details,
                            "model": result.get("model_used"),
                        }
                    )

                except Exception as query_exc:
                    err_msg = f"❌ Error retrieving answer: {query_exc}"
                    st.error(err_msg)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": err_msg,
                            "citations": {},
                            "chunks": [],
                        }
                    )
```

---

### Step 7.7: Automated Test Suite (`tests/test_rag_pipeline.py`)
Create `tests/__init__.py` (empty file) and `tests/test_rag_pipeline.py` to automate testing of configuration validation, SHA-256 hashing, chunking size, 1-indexed page conversion, citation formatting, and grounding refusals without requiring live API calls:

```python
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
        model_name="mistralai/mistral-7b-instruct",
    )
    
    assert result["answer"] == "The provided document does not contain information to answer this question."
    assert result["grounded"] is False
    assert result["chunks"] == []


def test_grounded_system_prompt_rules():
    """Test that the system prompt strictly forbids hallucination and enforces document-only answers."""
    assert "EXCLUSIVELY" in GROUNDED_SYSTEM_PROMPT
    assert "The provided document does not contain information to answer this question." in GROUNDED_SYSTEM_PROMPT
    assert "Do not invent or hallucinate" in GROUNDED_SYSTEM_PROMPT
```

---

## 8. Exact Commands to Run the Application on Windows

Open **PowerShell** in the project directory:

```powershell
# 1. Activate the virtual environment
.\venv\Scripts\Activate.ps1

# 2. Run the automated unit test suite
pytest tests/ -v

# 3. Launch the Streamlit application
streamlit run app.py
```

If port `8501` is already in use by another service on your machine, specify port `8502`:
```powershell
streamlit run app.py --server.port 8502
```

The web app will open automatically in your browser at `http://localhost:8501` (or `http://localhost:8502`).

---

## 9. Testing & Workshop Verification Guide

To test the application in a classroom or technical workshop:

### Test Case 1: Testing Supported Content (e.g. Node.js PDF)
1. Upload `nodejs.pdf`.
2. Click **🚀 Index to Cloud** and observe the progress bars.
3. In the chat box, ask:
   ```text
   What is Node.js?
   ```
4. **Expected Behavior:**
   - The model responds explaining that Node.js explores the V8 JavaScript engine, non-blocking I/O, and is commonly used as a web server to serve up websites and JSON.
   - The **Source Citations** badge displays: `nodejs.pdf (Pages 8, 10, 23)`.
   - The **Inspect Retrieved Chunks** panel shows the exact four passages retrieved from Qdrant Cloud along with their cosine similarity scores (e.g. `0.65` to `0.69`).

### Test Case 2: Testing Feature-Specific Question
1. Ask:
   ```text
   What does the Node.js module system let you do?
   ```
2. **Expected Behavior:**
   - The model answers that the module system enables loading external libraries, built-in core modules, and third-party npm modules for working with file systems, databases, and servers.
   - Citations display: `nodejs.pdf (Pages 8, 10)`.

### Test Case 3: Testing Anti-Hallucination & Refusal
1. Ask an unrelated question not mentioned anywhere in the document:
   ```text
   What is the speed of light in vacuum?
   ```
2. **Expected Behavior:**
   - The model does **not** hallucinate or provide general physics knowledge.
   - It strictly replies:
     > *"The provided document does not contain information to answer this question."*

---

## 10. Common Errors & Troubleshooting

| Error / Symptom | Root Cause | Solution |
|---|---|---|
| `QDRANT_URL points to localhost` | Local Docker address used in `.env` | Local databases are not permitted. Use your official Qdrant Cloud cluster URL (`https://...aws.cloud.qdrant.io:6333`). |
| `401 Unauthorized (Qdrant)` | Invalid or expired Qdrant API key | Navigate to Qdrant Cloud Console > **API Keys**, generate a new key, and update `QDRANT_API_KEY` in `.env`. |
| `401 Unauthorized (OpenRouter)` | Missing key or insufficient credits | Verify your key at [openrouter.ai/keys](https://openrouter.ai/keys). Ensure `OPENROUTER_MODEL=recursive/flint-v1.0:free`. |
| `Model ... is no longer available on OpenRouter (404)` | Provider retired the model identifier | The application includes an automatic fallback to `meta-llama/llama-3.1-8b-instruct`. You can also configure any active OpenRouter model in `.env`. |
| `PDF contains no extractable text` | PDF contains scanned images only | `PyPDFLoader` requires selectable digital text. Use a digital PDF or apply an OCR tool beforehand. |
| Model refuses valid questions (`Does not contain info...`) | Greedy decoding over-refusal at `temp=0.0` | Ensure `temperature=0.2` in `rag/generation.py` to allow natural synthesis while strictly bounding output to context. |
| Port 8501 is not available | Another instance of Streamlit is running | Start Streamlit on another port: `streamlit run app.py --server.port 8502`. |
