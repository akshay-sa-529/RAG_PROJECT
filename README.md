# Learn RAG — Cloud-Native Document Question Answering

A production-ready, beginner-friendly **Retrieval-Augmented Generation (RAG)** application built in Python with **Streamlit**, **LangChain**, **Qdrant Cloud**, and **OpenRouter**.

Upload any text-based PDF, index its contents into a managed cloud vector database with SHA-256 deduplication, and ask natural language questions with strict grounding and verifiable page citations.

---

## Architecture & Workflow

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

### Key Highlights
- **Human-Friendly 1-Indexed Citations:** Automatically maps PyPDFLoader's 0-indexed internal pages to natural 1-indexed page citations (e.g., `document.pdf (Pages 2, 4)`).
- **Multi-Document Isolation:** Every chunk is tagged with a SHA-256 based `doc_id`. Similarity searches are scoped with Qdrant payload filters so documents never mix.
- **Strict Grounding:** The LLM prompt explicitly restricts the model to uploaded context. If the document doesn't contain the answer, it refuses to hallucinate and responds with: *"The provided document does not contain information to answer this question."*
- **Cloud-Native Vector Store:** Connects directly to Qdrant Cloud without requiring local Docker setups.
- **Retrieval Diagnostics:** Expandable chunk inspection view showing exact passage text, page numbers, and cosine similarity scores.

---

## Project Structure

```text
RAG_PROJECT/
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
├── README.md                   # Project overview & workshop guide
└── implementation.md           # Step-by-step implementation guide
```

---

## Prerequisites & Accounts

1. **Python 3.10+**
2. **Qdrant Cloud Account (Free Tier):**
   - Create a free cluster at [cloud.qdrant.io](https://cloud.qdrant.io/).
   - Copy your **Cluster URL** (e.g. `https://your-cluster-id.region.aws.cloud.qdrant.io:6333`).
   - Generate an **API Key** under Data Access Control.
3. **OpenRouter Account:**
   - Sign up at [openrouter.ai](https://openrouter.ai/) and generate an API key at [openrouter.ai/keys](https://openrouter.ai/keys).
   - The default model is `recursive/flint-v1.0:free` (free tier).

---

## Quickstart Setup

### 1. Create and Activate Virtual Environment

**Windows PowerShell:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

*(If you encounter execution policy restrictions, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` before activating).*

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```

Edit `.env` with your credentials:
```env
QDRANT_URL=https://your-actual-cluster-url.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key_here
QDRANT_COLLECTION_NAME=learning_rag
OPENROUTER_API_KEY=sk-or-v1-your_openrouter_api_key_here
OPENROUTER_MODEL=recursive/flint-v1.0:free
```

---

## Running the Application

### 1. Run Automated Unit Tests
Verify chunking, metadata preservation, citations, and grounding rules:
```powershell
pytest tests/test_rag_pipeline.py -v
```

### 2. (Optional) Run Live Cloud Validation
Verify your live Qdrant Cloud cluster and OpenRouter connection:
```powershell
python tests/validate_cloud_setup.py
```

### 3. Launch the Streamlit Web Application
```powershell
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## Testing Scenarios

1. **Supported Question:**
   - Upload a text PDF (e.g. `nodejs.pdf` or lecture notes).
   - Click **🚀 Index to Cloud**.
   - Ask: *"What is Node.js?"*
   - Verify: The model explains the concept with exact page citations (e.g. `nodejs.pdf (Pages 8, 10)`), and the retrieved chunks match the diagnostics panel.

2. **Refusal / Anti-Hallucination:**
   - Ask an unmentioned query: *"What is the speed of light in vacuum?"*
   - Verify: Model answers *"The provided document does not contain information to answer this question."*
