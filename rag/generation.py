"""
Generation Module for Learn RAG with MCP-Powered Thinking & Grounded Synthesis.
Connects to OpenRouter API (recursive/flint-v1.0:free by default),
orchestrates MCP tool execution during the model thinking/reasoning phase,
enforces strict context grounding, logs reasoning traces to MongoDB,
and bundles answers with exact source citations.
"""

import datetime
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from rag.retrieval import format_citations
from rag.db import get_rag_database
from rag.mcp_tools import execute_mcp_retrieval


GROUNDED_SYSTEM_PROMPT = """You are an accurate, strictly grounded AI teaching assistant for a technical RAG workshop.
Your mission is to answer questions using EXCLUSIVELY the provided context passages from the uploaded document.

CRITICAL RULES:
1. Answer the user's question using the provided document context passages. Extract, summarize, and synthesize any relevant details, explanations, features, or uses explicitly present in the text.
2. Do NOT use outside knowledge, prior assumptions, or extrapolate beyond what is explicitly stated in the context.
3. If the provided context contains NO relevant information or mentions nothing about the subject of the question, reply with:
   "The provided document does not contain information to answer this question."
4. Keep the answer direct, factual, and well-explained for students. Do not invent or hallucinate facts or citations."""


THINKING_SYSTEM_PROMPT = """You are an advanced reasoning AI evaluating retrieved document passages before formulating an answer.
You will first produce your thinking process enclosed inside <thinking> ... </thinking> tags.
In your thinking process:
- Identify what the user is asking.
- Evaluate the retrieved passages from the document: which passages contain relevant facts, and what are their page numbers?
- Check if there is enough evidence to answer factually, or if the document lacks this information.
- Note any direct quotes or details.
After closing the </thinking> tag, formulate your final answer strictly grounded in the passages."""


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
    model_name: str = "recursive/flint-v1.0:free",
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
    retrieved_chunks: Optional[List[Document]] = None,
    openrouter_api_key: str = "",
    model_name: str = "recursive/flint-v1.0:free",
    doc_id: Optional[str] = None,
    include_thinking: bool = True
) -> Dict[str, Any]:
    """
    Executes grounded answer generation with an agentic MCP thinking process:
    1. Orchestrates MCP retrieval if chunks are not provided.
    2. Builds structured thinking trace (reasoning, tool calls, grounding checks).
    3. Invokes LLM with strict grounding prompt.
    4. Persists the thinking trace, question, and answer into MongoDB.
    """
    thinking_trace: List[Dict[str, Any]] = []
    timestamp_start = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Step 1: Thinking Initialization & Analysis
    thinking_trace.append({
        "step": 1,
        "phase": "Question Analysis",
        "action": "Parse Intent",
        "detail": f"Analyzing query '{query}' to identify target information and retrieval scope.",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    })

    # Step 2: MCP Retrieval (if chunks not already supplied, or to enrich)
    if retrieved_chunks is None:
        thinking_trace.append({
            "step": 2,
            "phase": "MCP Tool Invocation",
            "action": "execute_mcp_retrieval",
            "detail": f"Calling MCP tool 'retrieve_context_mcp' scoped to doc_id='{doc_id}'",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        })
        mcp_res = execute_mcp_retrieval(query=query, doc_id=doc_id)
        retrieved_chunks = mcp_res.get("raw_docs", [])
        retrieval_source = mcp_res.get("retrieval_source", "MCP Tool")
    else:
        retrieval_source = "Pre-retrieved Vector Chunks"

    citations = format_citations(retrieved_chunks)
    pages_found = sorted(list({c.metadata.get("page", 1) for c in retrieved_chunks}))

    thinking_trace.append({
        "step": len(thinking_trace) + 1,
        "phase": "Context Inspection",
        "action": "Evaluate Evidence",
        "detail": f"Retrieved {len(retrieved_chunks)} passage(s) across page(s) {pages_found} via {retrieval_source}.",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    })

    if not retrieved_chunks:
        fallback_msg = "The provided document does not contain information to answer this question."
        thinking_trace.append({
            "step": len(thinking_trace) + 1,
            "phase": "Grounding Check",
            "action": "Declined Out-of-Context",
            "detail": "No relevant passages were found in the uploaded document. Responding with strict fallback.",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        })

        db = get_rag_database()
        db.save_thinking_session(
            query=query,
            thinking_trace=thinking_trace,
            answer=fallback_msg,
            citations=citations,
            chunks=[],
            doc_id=doc_id,
            model_used=model_name
        )

        return {
            "answer": fallback_msg,
            "citations": citations,
            "chunks": [],
            "model_used": model_name,
            "grounded": False,
            "thinking_trace": thinking_trace
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

    answer_text = response.content.strip() if hasattr(response, "content") else str(response).strip()

    # Step 4: Verification of grounded response
    is_not_found = (
        "does not contain information" in answer_text.lower()
        or "not found in the provided" in answer_text.lower()
        or "context does not mention" in answer_text.lower()
    )

    thinking_trace.append({
        "step": len(thinking_trace) + 1,
        "phase": "Synthesis & Attributions",
        "action": "Finalize Answer",
        "detail": f"Synthesized answer grounded in document context ({'Verified' if not is_not_found else 'Information missing'}). Citations assigned to pages {pages_found}.",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    })

    # Step 5: Save thinking session to MongoDB
    chunk_dicts = [
        {
            "page": doc.metadata.get("page", 1),
            "score": doc.metadata.get("similarity_score"),
            "content": doc.page_content
        }
        for doc in retrieved_chunks
    ]

    db = get_rag_database()
    db.save_thinking_session(
        query=query,
        thinking_trace=thinking_trace,
        answer=answer_text,
        citations=citations,
        chunks=chunk_dicts,
        doc_id=doc_id or (retrieved_chunks[0].metadata.get("doc_id") if retrieved_chunks else None),
        model_used=model_name
    )

    return {
        "answer": answer_text,
        "citations": citations,
        "chunks": retrieved_chunks,
        "model_used": model_name,
        "grounded": not is_not_found,
        "thinking_trace": thinking_trace
    }
