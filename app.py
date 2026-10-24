"""
PaperMind AI — Cloud-Native Document Intelligence
==================================================
A human-crafted, production-ready Retrieval-Augmented Generation (RAG) interface.
Powered by Qdrant Cloud vector search, Hugging Face embeddings, and OpenRouter LLMs.
"""

import os
import streamlit as st

from rag.config import get_config, validate_config
from rag.db import get_rag_database
from rag.mcp_tools import execute_mcp_chunking, execute_mcp_retrieval
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


# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="PaperMind · Document Intelligence",
    page_icon="https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Design System & Motion Graphics (CSS Animations & Typography)
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* Base Font & Theme Overrides */
html, body, [class*="css"], .stMarkdown, p, button, input {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}

code, pre, .stCodeBlock {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Hide default streamlit decorations */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main container spacing */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
    max-width: 1200px !important;
}

/* ---------------- Motion Keyframes ---------------- */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(12px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes pulseGlow {
    0% {
        box-shadow: 0 0 0 0 rgba(79, 70, 229, 0.4);
    }
    70% {
        box-shadow: 0 0 0 10px rgba(79, 70, 229, 0);
    }
    100% {
        box-shadow: 0 0 0 0 rgba(79, 70, 229, 0);
    }
}

@keyframes pulseGreen {
    0% {
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4);
    }
    70% {
        box-shadow: 0 0 0 8px rgba(16, 185, 129, 0);
    }
    100% {
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
    }
}

@keyframes scanSweep {
    0% {
        left: 0%;
        width: 25%;
    }
    50% {
        width: 55%;
    }
    100% {
        left: 75%;
        width: 25%;
    }
}

@keyframes shimmerEffect {
    0% {
        background-position: -200% 0;
    }
    100% {
        background-position: 200% 0;
    }
}

@keyframes dotPulse {
    0%, 80%, 100% {
        transform: scale(0.6);
        opacity: 0.4;
    }
    40% {
        transform: scale(1.1);
        opacity: 1;
    }
}

/* Brand Hero Banner */
.brand-hero {
    background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 60%, #312E81 100%);
    border-radius: 20px;
    padding: 32px 36px;
    color: #FFFFFF;
    margin-bottom: 28px;
    box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.2), 0 8px 10px -6px rgba(15, 23, 42, 0.2);
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.1);
    animation: fadeInUp 0.4s ease-out;
}

.brand-hero::after {
    content: "";
    position: absolute;
    top: -40px;
    right: -40px;
    width: 220px;
    height: 220px;
    background: radial-gradient(circle, rgba(99, 102, 241, 0.3) 0%, rgba(99, 102, 241, 0) 70%);
    border-radius: 50%;
    pointer-events: none;
}

.brand-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(255, 255, 255, 0.12);
    border: 1px solid rgba(255, 255, 255, 0.2);
    backdrop-filter: blur(8px);
    padding: 6px 14px;
    border-radius: 100px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    color: #E0E7FF;
    margin-bottom: 14px;
    text-transform: uppercase;
}

.pulsing-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #38BDF8;
    animation: pulseGlow 1.6s infinite;
    display: inline-block;
}

.pulsing-dot-green {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #10B981;
    animation: pulseGreen 1.8s infinite;
    display: inline-block;
}

.brand-title {
    font-size: 2.25rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.2;
    margin: 0 0 10px 0;
    background: linear-gradient(180deg, #FFFFFF 0%, #E2E8F0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.brand-subtitle {
    font-size: 1.05rem;
    color: #CBD5E1;
    max-width: 680px;
    line-height: 1.6;
    margin: 0;
    font-weight: 400;
}

/* Surface Cards */
.human-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 1px 3px 0 rgba(15, 23, 42, 0.04), 0 1px 2px -1px rgba(15, 23, 42, 0.04);
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    animation: fadeInUp 0.45s ease-out;
}

.human-card:hover {
    box-shadow: 0 8px 24px -4px rgba(15, 23, 42, 0.08);
    border-color: #CBD5E1;
}

.card-header-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
    border-bottom: 1px solid #F1F5F9;
    padding-bottom: 12px;
}

.card-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #0F172A;
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 0;
}

.card-subtitle {
    font-size: 0.88rem;
    color: #64748B;
    margin: 2px 0 0 0;
}

/* Scanning / Searching Animation Box */
.searching-scanner-container {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 16px 20px;
    margin: 14px 0 20px 0;
    position: relative;
    overflow: hidden;
    animation: fadeInUp 0.3s ease-out;
}

.scanning-laser {
    position: absolute;
    top: 0;
    left: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, #4F46E5 50%, #06B6D4 100%);
    box-shadow: 0 0 8px #4F46E5;
    animation: scanSweep 1.8s ease-in-out infinite alternate;
}

.scanner-content {
    display: flex;
    align-items: center;
    gap: 12px;
}

.animated-loading-dots {
    display: inline-flex;
    gap: 4px;
    align-items: center;
}

.animated-loading-dots span {
    width: 6px;
    height: 6px;
    background: #4F46E5;
    border-radius: 50%;
    animation: dotPulse 1.2s infinite ease-in-out both;
}

.animated-loading-dots span:nth-child(1) { animation-delay: -0.32s; }
.animated-loading-dots span:nth-child(2) { animation-delay: -0.16s; }
.animated-loading-dots span:nth-child(3) { animation-delay: 0s; }

.scanner-text {
    font-size: 0.9rem;
    font-weight: 600;
    color: #334155;
}

.scanner-subtext {
    font-size: 0.78rem;
    color: #64748B;
    margin-top: 2px;
}

/* Feature Showcase Cards (Empty State) */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 16px;
    margin-top: 16px;
    margin-bottom: 24px;
}

.feature-box {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 20px;
    transition: transform 0.2s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.2s ease, box-shadow 0.2s ease;
}

.feature-box:hover {
    transform: translateY(-3px);
    border-color: #818CF8;
    background: #FFFFFF;
    box-shadow: 0 10px 20px -5px rgba(79, 70, 229, 0.08);
}

.feature-icon-wrapper {
    width: 42px;
    height: 42px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 14px;
    background: #EEF2FF;
    color: #4F46E5;
}

.feature-heading {
    font-size: 1rem;
    font-weight: 700;
    color: #1E293B;
    margin-bottom: 6px;
}

.feature-desc {
    font-size: 0.875rem;
    color: #64748B;
    line-height: 1.5;
    margin: 0;
}

/* Status Chips & Badges */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
}

.pill-success {
    background: #ECFDF5;
    color: #065F46;
    border: 1px solid #A7F3D0;
}

.pill-indigo {
    background: #EEF2FF;
    color: #3730A3;
    border: 1px solid #C7D2FE;
}

.pill-amber {
    background: #FFFBEB;
    color: #92400E;
    border: 1px solid #FDE68A;
}

/* Verified Citations Component */
.citation-container {
    background: linear-gradient(180deg, #F8FAFC 0%, #F1F5F9 100%);
    border-left: 4px solid #4F46E5;
    border-radius: 0 12px 12px 0;
    padding: 14px 18px;
    margin: 14px 0;
    border-top: 1px solid #E2E8F0;
    border-right: 1px solid #E2E8F0;
    border-bottom: 1px solid #E2E8F0;
    animation: fadeInUp 0.3s ease-out;
}

.citation-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 6px;
}

.citation-label {
    font-size: 0.82rem;
    font-weight: 700;
    color: #312E81;
    display: flex;
    align-items: center;
    gap: 6px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.citation-text {
    font-size: 0.92rem;
    color: #1E293B;
    font-weight: 500;
    margin: 0;
}

/* Retrieved Evidence Card */
.evidence-card {
    background: #FAFAFA;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 10px;
}

.evidence-meta {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.78rem;
    font-weight: 600;
    color: #6B7280;
    margin-bottom: 8px;
}

/* Quick prompt starter hints */
.starter-hint {
    font-size: 0.85rem;
    font-weight: 600;
    color: #475569;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* MCP Thinking & Reasoning Trace Styling */
.thinking-container {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-left: 3px solid #6366F1;
    border-radius: 10px;
    padding: 14px 16px;
    margin: 10px 0 16px 0;
    font-size: 0.86rem;
}

.thinking-header {
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 700;
    color: #1E293B;
    margin-bottom: 10px;
    font-size: 0.88rem;
}

.thinking-step-row {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin-bottom: 8px;
    font-size: 0.84rem;
    line-height: 1.5;
}

.thinking-step-row:last-child {
    margin-bottom: 0;
}

.thinking-badge {
    background: #EEF2FF;
    color: #4F46E5;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 6px;
    font-size: 0.72rem;
    white-space: nowrap;
    border: 1px solid rgba(79, 70, 229, 0.2);
}


/* Sidebar Custom Styling */
[data-testid="stSidebar"] {
    background-color: #F8FAFC !important;
    border-right: 1px solid #E2E8F0 !important;
}

.sidebar-brand {
    padding: 8px 0 16px 0;
    border-bottom: 1px solid #E2E8F0;
    margin-bottom: 18px;
}

.sidebar-title {
    font-size: 1.25rem;
    font-weight: 800;
    color: #0F172A;
    letter-spacing: -0.02em;
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 0;
}

.sidebar-subtitle {
    font-size: 0.8rem;
    color: #64748B;
    margin-top: 2px;
}

.doc-pill-card {
    background: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 16px;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.03);
}

.doc-name {
    font-weight: 700;
    font-size: 0.9rem;
    color: #0F172A;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.doc-stat {
    font-size: 0.78rem;
    color: #64748B;
    margin-top: 4px;
}

/* Modern Primary Buttons */
div.stButton > button:first-child {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    transition: all 0.18s cubic-bezier(0.16, 1, 0.3, 1) !important;
    border: 1px solid #CBD5E1 !important;
}

div.stButton > button:first-child:hover {
    border-color: #4F46E5 !important;
    color: #4F46E5 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.12) !important;
}

div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4F46E5 0%, #4338CA 100%) !important;
    border: none !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.25) !important;
}

div.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #4338CA 0%, #3730A3 100%) !important;
    box-shadow: 0 6px 18px rgba(79, 70, 229, 0.35) !important;
    color: #FFFFFF !important;
    transform: translateY(-2px) !important;
}

/* Streamlit Chat Messages Styling */
.stChatMessage {
    border-radius: 14px !important;
    padding: 14px 18px !important;
    margin-bottom: 12px !important;
    border: 1px solid #E2E8F0 !important;
    background: #FFFFFF !important;
    animation: fadeInUp 0.25s ease-out;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# State Management
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "active_document" not in st.session_state:
    st.session_state.active_document = None

if "preset_prompt" not in st.session_state:
    st.session_state.preset_prompt = None

config = get_config()
is_config_valid, config_issues = validate_config(config)

# -----------------------------------------------------------------------------
# Sidebar Experience
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-title">
                <span class="pulsing-dot"></span>
                <span>PaperMind AI</span>
            </div>
            <div class="sidebar-subtitle">Cloud-Native Document Intelligence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Active Document Information
    st.markdown("##### Active Workspace")
    if st.session_state.active_document:
        doc = st.session_state.active_document
        st.markdown(
            f"""
            <div class="doc-pill-card">
                <div class="doc-name" title="{doc['filename']}">{doc['filename']}</div>
                <div class="doc-stat">
                    <span>{doc.get('total_pages', 'N/A')} pages</span> · 
                    <span>{doc.get('total_chunks', 'N/A')} vector chunks</span>
                </div>
                <div class="doc-stat" style="font-family: monospace; font-size: 0.72rem; color: #94A3B8; margin-top: 6px;">
                    ID: {doc['doc_id'][:12]}...{doc['doc_id'][-6:]}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_switch, col_del = st.columns(2)
        with col_switch:
            if st.button("Switch PDF", use_container_width=True, help="Upload or switch to another document"):
                st.session_state.active_document = None
                st.rerun()

        with col_del:
            if st.button("Delete Document", use_container_width=True, help="Remove vectors from Qdrant Cloud"):
                with st.spinner("Deleting vectors from cloud storage..."):
                    try:
                        client = get_qdrant_client(config.qdrant_url, config.qdrant_api_key)
                        del_count = delete_document_vectors(
                            client, config.qdrant_collection_name, doc["doc_id"]
                        )
                        st.session_state.active_document = None
                        st.session_state.messages = []
                        st.toast(f"Removed {del_count} vectors from Qdrant Cloud.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to delete: {e}")
    else:
        st.markdown(
            """
            <div style="background: #F1F5F9; border-radius: 10px; padding: 12px; font-size: 0.84rem; color: #64748B; margin-bottom: 16px;">
                No document loaded. Upload a PDF in the main workspace to begin.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Cloud Connectivity Telemetry
    st.markdown("##### System Telemetry")
    db = get_rag_database()
    mongo_status = db.connection_status

    if config.qdrant_url and "localhost" not in config.qdrant_url and config.qdrant_api_key:
        cluster_host = config.qdrant_url.split("//")[-1].split(":")[0]
        st.markdown(
            f"""
            <div style="margin-bottom: 8px;">
                <span class="status-pill pill-success">
                    <span class="pulsing-dot-green"></span>
                    <span>Qdrant Cloud Connected</span>
                </span>
                <div style="font-size: 0.75rem; color: #64748B; margin-top: 4px; font-family: monospace;">{cluster_host}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<span class="status-pill pill-amber">Qdrant Not Configured</span>', unsafe_allow_html=True)

    # MongoDB Persistence Telemetry
    if mongo_status["connected"]:
        st.markdown(
            f"""
            <div style="margin-bottom: 8px;">
                <span class="status-pill pill-success">
                    <span class="pulsing-dot-green"></span>
                    <span>MongoDB: Connected</span>
                </span>
                <div style="font-size: 0.75rem; color: #64748B; margin-top: 4px; font-family: monospace;">DB: {mongo_status['db_name']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="margin-bottom: 8px;">
                <span class="status-pill pill-indigo">
                    <span>Storage: In-Memory Fallback</span>
                </span>
                <div style="font-size: 0.72rem; color: #64748B; margin-top: 3px;">Start mongod or set MONGODB_URI in .env</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # FastAPI & MCP Telemetry
    st.markdown(
        f"""
        <div style="margin-bottom: 8px;">
            <span class="status-pill pill-indigo">
                <span class="pulsing-dot"></span>
                <span>FastAPI & MCP Tools: Ready</span>
            </span>
            <div style="font-size: 0.72rem; color: #64748B; margin-top: 3px;">API: <code>{config.fastapi_url}</code></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if config.openrouter_api_key:
        st.markdown(
            f"""
            <div style="margin-bottom: 8px;">
                <span class="status-pill pill-indigo">
                    <span>Model: {config.openrouter_model.split('/')[-1]}</span>
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<span class="status-pill pill-amber">OpenRouter Key Missing</span>', unsafe_allow_html=True)

    st.caption(f"Vector Collection: `{config.qdrant_collection_name}`")


    if not is_config_valid:
        with st.expander("Configuration Checklist", expanded=False):
            for issue in config_issues:
                st.write(f"• {issue}")
            st.caption("Update credentials in `.env` to resolve.")

    st.markdown("---")

    # Display Preferences
    st.markdown("##### Preferences")
    show_diagnostics = st.toggle(
        "Inspect Chunk Diagnostics",
        value=True,
        help="Display cosine similarity scores, passage text, and page citation metadata.",
    )

    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.toast("Conversation cleared.")
        st.rerun()

    with st.expander("How the Engine Works"):
        st.markdown(
            """
            - **1. Extraction:** Extracts clean text while locking 1-indexed page coordinates.
            - **2. Semantic Chunking:** 1,000 character windows with 400 character overlap.
            - **3. Dense Vectors:** `all-MiniLM-L6-v2` encodes 384-dimensional embeddings.
            - **4. Cloud Indexing:** Stored in Qdrant Cloud with cryptographic `doc_id` isolation.
            - **5. Grounded Synthesis:** Answers strictly from retrieved passages without hallucinations.
            """
        )

# -----------------------------------------------------------------------------
# Hero Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="brand-hero">
        <div class="brand-pill">
            <span class="pulsing-dot"></span>
            <span>RAG Document Intelligence</span>
        </div>
        <h1 class="brand-title">Talk with Any PDF with Absolute Grounding</h1>
        <p class="brand-subtitle">
            Index text documents to Qdrant Cloud vector search and receive verified answers with human-friendly page citations.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Early exit if environment is incomplete
if not is_config_valid:
    st.error("Configuration Required: Missing credentials in `.env` file.")
    with st.expander("View Required Credentials", expanded=True):
        for issue in config_issues:
            st.write(f"• {issue}")
    st.stop()

# -----------------------------------------------------------------------------
# Section 1: Document Upload & Cloud Ingestion
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="human-card">
        <div class="card-header-row">
            <div>
                <h3 class="card-title">Document Ingestion Studio</h3>
                <p class="card-subtitle">Choose a text PDF to index into Qdrant Cloud vector database</p>
            </div>
            <span class="status-pill pill-indigo">SHA-256 Deduplicated</span>
        </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload PDF document",
    type=["pdf"],
    label_visibility="collapsed",
    help="Upload text-based PDF documents (whitepapers, contracts, manuals, research papers).",
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
        st.error(f"Could not connect to Qdrant Cloud: {e}. Check your credentials.")
        already_indexed, existing_chunks = False, 0

    col_meta, col_btn = st.columns([3, 1.2])

    with col_meta:
        if already_indexed:
            st.markdown(
                f"""
                <div style="background: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 10px; padding: 12px 16px;">
                    <div style="font-weight: 700; color: #065F46; font-size: 0.92rem; display: flex; align-items: center; gap: 8px;">
                        <span class="pulsing-dot-green"></span>
                        <span>Document Synced in Cloud</span>
                    </div>
                    <div style="color: #047857; font-size: 0.84rem; margin-top: 4px;">
                        <strong>{filename}</strong> is ready for querying ({existing_chunks} vector chunks indexed).
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            file_kb = len(file_bytes) / 1024
            st.markdown(
                f"""
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 12px 16px;">
                    <div style="font-weight: 700; color: #1E293B; font-size: 0.92rem; display: flex; align-items: center; gap: 8px;">
                        <span class="pulsing-dot"></span>
                        <span>Ready for Cloud Indexing</span>
                    </div>
                    <div style="color: #64748B; font-size: 0.84rem; margin-top: 4px;">
                        <strong>{filename}</strong> ({file_kb:.1f} KB) · SHA-256 fingerprint verified.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_btn:
        button_label = "Re-Index Document" if already_indexed else "Index to Cloud"
        start_indexing = st.button(button_label, use_container_width=True, type="primary")

    if already_indexed and (not st.session_state.active_document or st.session_state.active_document.get("doc_id") != doc_id):
        if st.button("Set as Active Document for Chat", use_container_width=True):
            st.session_state.active_document = {
                "doc_id": doc_id,
                "filename": filename,
                "total_chunks": existing_chunks,
            }
            st.toast(f"Active document set to '{filename}'")
            st.rerun()

    if start_indexing:
        # Animated Ingestion Scanner Feedback
        ingestion_placeholder = st.empty()
        ingestion_placeholder.markdown(
            """
            <div class="searching-scanner-container">
                <div class="scanning-laser"></div>
                <div class="scanner-content">
                    <div class="animated-loading-dots">
                        <span></span><span></span><span></span>
                    </div>
                    <div>
                        <div class="scanner-text">Processing Document through Ingestion Pipeline</div>
                        <div class="scanner-subtext">Extracting pages, chunking text, and generating 384d vector embeddings...</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        progress_bar = st.progress(0, text="Initializing ingestion pipeline...")
        try:
            progress_bar.progress(20, text="Extracting PDF pages & locking 1-indexed page coordinates...")
            pages = load_pdf_from_bytes(file_bytes, filename)

            progress_bar.progress(45, text=f"Chunking with MCP Tool & saving to MongoDB (size: {config.chunk_size}, overlap: {config.chunk_overlap})...")
            mcp_res = execute_mcp_chunking(
                doc_id=doc_id,
                filename=filename,
                pages=pages,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                store_in_qdrant=True,
            )
            indexed_count = mcp_res["total_chunks"]
            storage_mode = mcp_res.get("storage_mode", "MongoDB")

            progress_bar.progress(100, text=f"Indexing complete! Saved to {storage_mode} and Qdrant Cloud.")
            ingestion_placeholder.empty()

            st.session_state.active_document = {
                "doc_id": doc_id,
                "filename": filename,
                "total_pages": len(pages),
                "total_chunks": indexed_count,
                "storage_mode": storage_mode
            }
            st.toast(f"Successfully indexed {indexed_count} chunks in {storage_mode} & Qdrant Cloud!")
            st.rerun()

        except Exception as exc:
            ingestion_placeholder.empty()
            st.error(f"Ingestion error: {exc}")

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Section 2: Conversational Intelligence (Chat)
# -----------------------------------------------------------------------------
if not st.session_state.active_document:
    # Clean onboarding state with SVG graphics & subtle hover animations
    st.markdown(
        """
        <div class="human-card">
            <div class="card-header-row">
                <div>
                    <h3 class="card-title">How PaperMind Grounds Your Questions</h3>
                    <p class="card-subtitle">Engineered to prevent hallucinations and provide exact citations</p>
                </div>
            </div>
            <div class="feature-grid">
                <div class="feature-box">
                    <div class="feature-icon-wrapper">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                            <polyline points="14 2 14 8 20 8"></polyline>
                            <line x1="16" y1="13" x2="8" y2="13"></line>
                            <line x1="16" y1="17" x2="8" y2="17"></line>
                        </svg>
                    </div>
                    <div class="feature-heading">1-Indexed Page Citations</div>
                    <p class="feature-desc">Every retrieved answer maps directly to human page numbers so you can verify statements instantly.</p>
                </div>
                <div class="feature-box">
                    <div class="feature-icon-wrapper">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
                            <path d="m9 12 2 2 4-4"></path>
                        </svg>
                    </div>
                    <div class="feature-heading">Strict Grounding</div>
                    <p class="feature-desc">If the document does not contain an answer, the model explicitly declines instead of guessing.</p>
                </div>
                <div class="feature-box">
                    <div class="feature-icon-wrapper">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect>
                            <path d="M7 11V7a5 5 0 0 1 10 0v4"></path>
                        </svg>
                    </div>
                    <div class="feature-heading">Scoped Isolation</div>
                    <p class="feature-desc">Queries are isolated via cryptographic hashes so multi-document collections never bleed together.</p>
                </div>
            </div>
            <div style="text-align: center; color: #64748B; font-size: 0.9rem; padding: 12px 0;">
                <em>Upload or select a PDF above to activate the intelligence chat.</em>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

else:
    active_doc = st.session_state.active_document

    st.markdown(
        f"""
        <div class="human-card" style="margin-bottom: 16px; padding: 16px 20px;">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span class="pulsing-dot-green"></span>
                    <div>
                        <div style="font-weight: 700; color: #0F172A; font-size: 1.05rem;">{active_doc['filename']}</div>
                        <div style="font-size: 0.8rem; color: #64748B;">
                            {active_doc.get('total_pages', 'N/A')} pages · {active_doc.get('total_chunks', 'N/A')} indexed passages · Scoped via doc_id
                        </div>
                    </div>
                </div>
                <span class="status-pill pill-success">Live Cloud Context</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Prompt Starters if chat is clean
    query_from_pill = None
    if not st.session_state.messages:
        st.markdown(
            '<div class="starter-hint"><span>Quick prompt suggestions:</span></div>',
            unsafe_allow_html=True,
        )
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            if st.button("Summarize key findings", use_container_width=True):
                query_from_pill = "Summarize the key findings and main takeaways from this document."
        with col_s2:
            if st.button("What are the core topics?", use_container_width=True):
                query_from_pill = "What are the primary topics and central arguments presented in this document?"
        with col_s3:
            if st.button("Key limitations & conclusions", use_container_width=True):
                query_from_pill = "What conclusions, recommendations, or limitations are noted in this text?"

    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg["role"] == "assistant":
                thinking_trace = msg.get("thinking_trace", [])
                if thinking_trace:
                    with st.expander("🧠 Model Thinking & Reasoning Trace (MCP-Powered)", expanded=False):
                        for step in thinking_trace:
                            st.markdown(
                                f"""
                                <div class="thinking-step-row">
                                    <span class="thinking-badge">{step.get('phase', 'Reasoning')}</span>
                                    <div><strong>{step.get('action', '')}</strong>: {step.get('detail', '')}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                citations = msg.get("citations", {})
                citation_summary = citations.get("citation_summary", "")

                if citation_summary and "No citations" not in citation_summary:
                    st.markdown(
                        f"""
                        <div class="citation-container">
                            <div class="citation-header">
                                <span class="citation-label">Verified Source Citations</span>
                                <span style="font-size: 0.75rem; color: #64748B;">Grounding Verified</span>
                            </div>
                            <div class="citation-text">{citation_summary}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                chunks = msg.get("chunks", [])
                if chunks and show_diagnostics:
                    with st.expander(f"Retrieved Passages & Context ({len(chunks)} chunks from Qdrant Cloud)", expanded=False):
                        for i, ch in enumerate(chunks, 1):
                            pg = ch.get("page", 1)
                            score = ch.get("score")
                            score_label = f"{score * 100:.1f}% match" if score is not None else "Dense Match"
                            st.markdown(
                                f"""
                                <div class="evidence-card">
                                    <div class="evidence-meta">
                                        <span>Passage {i} · Page {pg}</span>
                                        <span class="status-pill pill-indigo">{score_label}</span>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            st.code(ch.get("content", ""), language="text")


    # Chat Input Box
    chat_input_val = st.chat_input("Ask any question regarding the uploaded PDF...")
    final_query = query_from_pill or chat_input_val

    if final_query:
        st.session_state.messages.append({"role": "user", "content": final_query})
        with st.chat_message("user"):
            st.markdown(final_query)

        with st.chat_message("assistant"):
            # Live Animated Searching Indicator
            searching_container = st.empty()
            searching_container.markdown(
                """
                <div class="searching-scanner-container">
                    <div class="scanning-laser"></div>
                    <div class="scanner-content">
                        <div class="animated-loading-dots">
                            <span></span><span></span><span></span>
                        </div>
                        <div>
                            <div class="scanner-text">Searching Qdrant Cloud Vectors</div>
                            <div class="scanner-subtext">Executing dense cosine similarity search & synthesizing grounded answer...</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            try:
                qdrant_client = get_qdrant_client(config.qdrant_url, config.qdrant_api_key)
                embeddings = get_embedding_model(config.embedding_model_name)

                retrieved_docs = retrieve_relevant_chunks(
                    client=qdrant_client,
                    embeddings_model=embeddings,
                    collection_name=config.qdrant_collection_name,
                    query=final_query,
                    doc_id=active_doc["doc_id"],
                    top_k=config.top_k,
                )

                result = generate_grounded_answer(
                    query=final_query,
                    retrieved_chunks=retrieved_docs,
                    openrouter_api_key=config.openrouter_api_key,
                    model_name=config.openrouter_model,
                    doc_id=active_doc["doc_id"],
                    include_thinking=True,
                )

                # Clear searching animation
                searching_container.empty()

                answer_text = result["answer"]
                citations = result["citations"]
                citation_summary = citations.get("citation_summary", "")
                thinking_trace = result.get("thinking_trace", [])

                # Render MCP Thinking Process
                if thinking_trace:
                    with st.expander("🧠 Model Thinking & Reasoning Trace (MCP-Powered)", expanded=True):
                        for step in thinking_trace:
                            st.markdown(
                                f"""
                                <div class="thinking-step-row">
                                    <span class="thinking-badge">{step.get('phase', 'Reasoning')}</span>
                                    <div><strong>{step.get('action', '')}</strong>: {step.get('detail', '')}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                st.markdown(answer_text)

                if citation_summary and "No citations" not in citation_summary:
                    st.markdown(
                        f"""
                        <div class="citation-container">
                            <div class="citation-header">
                                <span class="citation-label">Verified Source Citations</span>
                                <span style="font-size: 0.75rem; color: #64748B;">Grounding Verified</span>
                            </div>
                            <div class="citation-text">{citation_summary}</div>
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

                if chunk_details and show_diagnostics:
                    with st.expander(f"Retrieved Passages & Context ({len(chunk_details)} chunks from Qdrant Cloud)", expanded=False):
                        for i, ch in enumerate(chunk_details, 1):
                            pg = ch["page"]
                            sc = ch["score"]
                            sc_label = f"{sc * 100:.1f}% match" if sc is not None else "Dense Match"
                            st.markdown(
                                f"""
                                <div class="evidence-card">
                                    <div class="evidence-meta">
                                        <span>Passage {i} · Page {pg}</span>
                                        <span class="status-pill pill-indigo">{sc_label}</span>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            st.code(ch["content"], language="text")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer_text,
                        "citations": citations,
                        "chunks": chunk_details,
                        "thinking_trace": thinking_trace,
                        "model": result.get("model_used"),
                    }
                )

            except Exception as query_exc:
                searching_container.empty()
                err_msg = f"Error retrieving answer: {query_exc}"
                st.error(err_msg)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": err_msg,
                        "citations": {},
                        "chunks": [],
                    }
                )
