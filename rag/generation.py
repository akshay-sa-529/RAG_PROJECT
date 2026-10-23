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
    retrieved_chunks: List[Document],
    openrouter_api_key: str,
    model_name: str = "recursive/flint-v1.0:free"
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
            
    answer_text = response.content.strip() if hasattr(response, "content") else str(response).strip()
    
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
