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
