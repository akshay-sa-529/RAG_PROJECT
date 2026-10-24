# PaperMind AI — Cloud-Native Document Intelligence with MCP, MongoDB & FastAPI

A production-ready, enterprise-grade **Retrieval-Augmented Generation (RAG)** application built with **FastAPI**, **Streamlit**, **LangChain**, **Model Context Protocol (MCP)**, **MongoDB**, **Qdrant Cloud**, and **OpenRouter**.

Upload any text-based PDF, execute semantic chunking via **MCP Tools**, persist structured chunks and metadata into **MongoDB**, index vectors into **Qdrant Cloud**, and ask natural language questions with an agentic **MCP Thinking & Reasoning Trace**, strict grounding, and verifiable 1-indexed page citations.

---

## 🏛️ Architecture & Workflow

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                               THE DUAL-ENGINE RAG WORKFLOW                              │
└─────────────────────────────────────────────────────────────────────────────────────────┘

[PHASE A: DOCUMENT INGESTION & MCP CHUNKING]
 1. PDF Upload ──► 2. SHA-256 Hash ──► 3. MCP Chunking Tool ──► 4. MongoDB Persistence
   (FastAPI/Streamlit)  (doc_id deduplication) (RecursiveTextSplitter)   (Collections: documents,
                                                                          chunks, thinking_sessions)
                                                   │
                                                   └──► 5. Dense Embeddings (384d) ──► 6. Qdrant Cloud
                                                           (all-MiniLM-L6-v2)             (learning_rag)

[PHASE B: MCP THINKING & GROUNDED ANSWER GENERATION]
 7. User Query ──► 8. MCP Retrieval Tool ──► 9. Agentic Thinking Loop ──► 10. OpenRouter LLM ──► 11. Grounded Answer
   (POST /query)     (Qdrant Cloud vectors       (Analyze Intent, Verify       (recursive/flint)      + 1-Indexed Page
                      + MongoDB chunks)           Grounding, Trace Steps)                             Citations & MCP Trace
```

### Key Highlights
- **Model Context Protocol (MCP) Integration:**
  - **While Chunking:** `chunk_document_mcp` splits document pages into semantic chunks, stores records into MongoDB, and indexes vectors into Qdrant Cloud.
  - **While Thinking:** The generation pipeline executes an agentic reasoning loop using `retrieve_context_mcp` to inspect evidence, trace reasoning steps, and verify grounding before answering.
  - **MCP Server Mounted:** Exposes a standard MCP SSE endpoint at `/mcp` on FastAPI for external AI clients (Claude Desktop, Cursor, Antigravity).
- **MongoDB Persistence:**
  - Persists document metadata, raw chunk records, and agent thinking/reasoning sessions.
  - Built-in automatic fallback to an in-memory store if MongoDB is offline, ensuring the app never crashes.
- **FastAPI REST API:**
  - High-performance asynchronous backend with interactive OpenAPI docs (`/docs`).
  - Endpoints for PDF upload, MCP chunking, document management, and thinking query generation.
- **Human-Friendly 1-Indexed Citations:** Automatically maps internal 0-indexed page coordinates to natural 1-indexed page citations (e.g., `document.pdf (Pages 2, 4)`).
- **Multi-Document Isolation:** Every chunk is tagged with a SHA-256 based `doc_id`. Similarity searches are strictly scoped with Qdrant payload filters.
- **Strict Grounding:** The LLM prompt restricts the model exclusively to uploaded context. If information is missing, it refuses to hallucinate: *"The provided document does not contain information to answer this question."*

---

## 📂 Project Structure

```text
RAG_PROJECT/
├── app.py                      # Streamlit UI with live MCP thinking traces & diagnostics
├── api.py                      # FastAPI REST API & mounted MCP SSE server
├── rag/                        # Core modular RAG package
│   ├── __init__.py             # Public package exports
│   ├── config.py               # Environment configuration and validation
│   ├── db.py                   # MongoDB persistence layer with resilient fallback
│   ├── mcp_tools.py            # Model Context Protocol (MCP) server & tool suite
│   ├── ingestion.py            # PDF loading, SHA-256 hashing, chunking, Qdrant upload
│   ├── retrieval.py            # Embedding cache, scoped similarity search, citations
│   └── generation.py           # Grounded prompt engineering with MCP thinking trace
├── tests/                      # Automated test suite
│   ├── __init__.py             # Test package initializer
│   ├── test_rag_pipeline.py    # Unit tests for chunking, citations, and grounding
│   ├── test_fastapi_and_mcp.py # Unit tests for FastAPI, MongoDB, and MCP tools
│   └── validate_cloud_setup.py # End-to-end live cloud infrastructure test
├── requirements.txt            # Python package dependencies
├── .env.example                # Template for environment configuration
├── .gitignore                  # Git ignore rules for secrets and virtualenvs
├── README.md                   # Project overview & system guide
└── implementation.md           # Step-by-step implementation guide
```

---

## ⚙️ Prerequisites & Accounts

1. **Python 3.10+**
2. **Qdrant Cloud Account (Free Tier):**
   - Create a free cluster at [cloud.qdrant.io](https://cloud.qdrant.io/).
   - Copy your **Cluster URL** (e.g., `https://your-cluster-id.region.aws.cloud.qdrant.io:6333`).
   - Generate an **API Key** under Data Access Control.
3. **OpenRouter Account:**
   - Sign up at [openrouter.ai](https://openrouter.ai/) and generate an API key at [openrouter.ai/keys](https://openrouter.ai/keys).
   - The default model is `recursive/flint-v1.0:free` (free tier).
4. **MongoDB (Optional / Recommended):**
   - Local MongoDB (`mongodb://localhost:27017`) or MongoDB Atlas cloud URI (`mongodb+srv://...`).
   - If offline, the application automatically uses the in-memory fallback store.

---

## 🚀 Quickstart Setup

### 1. Activate Virtual Environment

**Windows PowerShell:**
```powershell
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
# Qdrant Cloud Vector Database Configuration
QDRANT_URL=https://your-cluster-id.us-east-1-0.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key_here
QDRANT_COLLECTION_NAME=learning_rag

# OpenRouter LLM Configuration
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=recursive/flint-v1.0:free

# MongoDB Configuration (Local or MongoDB Atlas)
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB_NAME=papermind_rag

# FastAPI Backend Configuration
FASTAPI_HOST=127.0.0.1
FASTAPI_PORT=8000
FASTAPI_URL=http://127.0.0.1:8000
```

---

## 🏃 Running the Application

### 1. Run Automated Tests
Verify pipeline, citations, FastAPI endpoints, MongoDB persistence, and MCP tools:
```powershell
pytest tests/ -v
```

### 2. Launch FastAPI Backend Server
Start the high-performance FastAPI service with OpenAPI documentation and mounted MCP server:
```powershell
python api.py
```
Or with Uvicorn:
```powershell
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```
- Interactive OpenAPI Docs: `http://127.0.0.1:8000/docs`
- Health Check: `http://127.0.0.1:8000/health`
- MCP SSE Endpoint: `http://127.0.0.1:8000/mcp/sse`

### 3. Launch Streamlit UI
In a separate terminal, launch the Streamlit frontend:
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📡 FastAPI REST Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Service info and active feature flags |
| `GET` | `/health` | Deep health check (MongoDB, Qdrant, config) |
| `GET` | `/config` | Safe configuration metadata |
| `POST` | `/documents/upload` | Upload PDF, run MCP chunking, save to MongoDB & Qdrant |
| `POST` | `/documents/chunk` | Chunk raw text pages via MCP tool & save to MongoDB |
| `GET` | `/documents` | List registered documents from MongoDB |
| `GET` | `/documents/{doc_id}` | Retrieve document details by doc_id |
| `GET` | `/documents/{doc_id}/chunks` | Retrieve chunk records from MongoDB |
| `DELETE` | `/documents/{doc_id}` | Delete document and chunks from MongoDB and Qdrant |
| `POST` | `/query` | Execute MCP thinking retrieval and grounded LLM generation |
| `GET` | `/history` | Retrieve recent thinking sessions from MongoDB |
| `GET` | `/mcp/sse` | MCP Server Server-Sent Events (SSE) transport |

---

## 🧪 Testing Scenarios

1. **Document Ingestion with MCP Chunking & MongoDB Storage:**
   - Upload a text PDF (e.g. `lecture_notes.pdf`).
   - Click **Index to Cloud**.
   - Notice the MCP Chunking Tool executes, chunks are saved to MongoDB, and vectors are registered in Qdrant Cloud.

2. **Querying with Live MCP Thinking Process:**
   - Ask: *"What are the core topics discussed in this document?"*
   - Expand the **🧠 Model Thinking & Reasoning Trace (MCP-Powered)** box to view each step taken during thinking:
     1. Intent analysis
     2. MCP retrieval tool call (`retrieve_context_mcp`)
     3. Evidence evaluation with page numbers
     4. Grounding verification and citation synthesis
   - Check the final verified page citations and retrieved passage diagnostic scores.

3. **Anti-Hallucination & Refusal:**
   - Ask an unmentioned query: *"What is the speed of light in vacuum?"*
   - Verify: Model states: *"The provided document does not contain information to answer this question."*
