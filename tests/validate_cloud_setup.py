"""
End-to-End Live Cloud Infrastructure & Pipeline Validation Script.
Tests Qdrant Cloud connectivity, collection readiness, and OpenRouter API access.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.documents import Document

from rag.config import get_config, validate_config
from rag.ingestion import (
    compute_document_id,
    split_document_pages,
    get_qdrant_client,
    ensure_collection_exists,
    index_chunks_to_qdrant,
    delete_document_vectors,
)
from rag.retrieval import get_embedding_model, retrieve_relevant_chunks, format_citations
from rag.generation import generate_grounded_answer


def run_cloud_validation():
    print("=" * 70)
    print("🚀 Learn RAG — Cloud Infrastructure & Pipeline Live Validation")
    print("=" * 70)

    # 1. Configuration check
    print("\n[Step 1/5] Checking Environment Configuration...")
    config = get_config()
    is_valid, issues = validate_config(config)

    if not is_valid:
        print("❌ Configuration validation failed:")
        for issue in issues:
            print(f"   • {issue}")
        print("\nPlease update your '.env' file with valid cloud credentials before proceeding.")
        sys.exit(1)

    print("✅ Configuration is valid!")
    print(f"   • Qdrant URL: {config.qdrant_url}")
    print(f"   • Collection: {config.qdrant_collection_name}")
    print(f"   • OpenRouter Model: {config.openrouter_model}")

    # 2. Qdrant Cloud connection
    print("\n[Step 2/5] Connecting to Qdrant Cloud...")
    try:
        client = get_qdrant_client(config.qdrant_url, config.qdrant_api_key)
        collections = client.get_collections().collections
        existing_names = [c.name for c in collections]
        print(f"✅ Connected to Qdrant Cloud successfully! Found {len(collections)} collections.")
        print(f"   • Collections: {', '.join(existing_names) if existing_names else 'None yet'}")
        
        ensure_collection_exists(client, config.qdrant_collection_name, vector_dim=384)
        print(f"✅ Collection '{config.qdrant_collection_name}' is ready.")
    except Exception as e:
        print(f"❌ Failed to connect to Qdrant Cloud: {e}")
        sys.exit(1)

    # 3. Embedding model check
    print("\n[Step 3/5] Initializing Embedding Model (all-MiniLM-L6-v2)...")
    try:
        embedding_model = get_embedding_model(config.embedding_model_name)
        test_vec = embedding_model.embed_query("Test connection probe")
        print(f"✅ Embedding model loaded successfully! Vector dimension: {len(test_vec)}")
    except Exception as e:
        print(f"❌ Failed to load embedding model: {e}")
        sys.exit(1)

    # 4. Ingest test document into Qdrant Cloud
    print("\n[Step 4/5] Testing Ingestion and Scoped Retrieval on Qdrant Cloud...")
    test_text_p1 = (
        "Quantum computing harnesses the phenomena of quantum mechanics such as superposition "
        "and entanglement to perform calculations exponentially faster than classical computers for certain problems."
    )
    test_text_p2 = (
        "Qubits can exist in a linear combination of states |0> and |1> simultaneously. "
        "Quantum error correction remains one of the greatest engineering hurdles in building fault-tolerant machines."
    )
    sample_docs = [
        Document(page_content=test_text_p1, metadata={"page": 1, "filename": "quantum_intro.pdf"}),
        Document(page_content=test_text_p2, metadata={"page": 2, "filename": "quantum_intro.pdf"}),
    ]
    test_doc_id = "test_probe_9999"

    try:
        chunks = split_document_pages(sample_docs, test_doc_id, "quantum_intro.pdf", chunk_size=500, chunk_overlap=100)
        indexed = index_chunks_to_qdrant(
            client=client,
            embeddings_model=embedding_model,
            collection_name=config.qdrant_collection_name,
            chunks=chunks,
            doc_id=test_doc_id,
            overwrite_existing=True,
        )
        print(f"✅ Successfully indexed {indexed} test chunks to Qdrant Cloud.")

        # Test scoped similarity retrieval
        retrieved = retrieve_relevant_chunks(
            client=client,
            embeddings_model=embedding_model,
            collection_name=config.qdrant_collection_name,
            query="What is superposition and entanglement?",
            doc_id=test_doc_id,
            top_k=2,
        )
        print(f"✅ Retrieved {len(retrieved)} relevant chunks scoped to doc_id '{test_doc_id}'.")
        citations = format_citations(retrieved)
        print(f"   • Formatted citation: {citations['citation_summary']}")
    except Exception as e:
        print(f"❌ Ingestion/retrieval test failed: {e}")
        sys.exit(1)

    # 5. OpenRouter LLM grounded response check
    print("\n[Step 5/5] Testing Grounded LLM Response via OpenRouter...")
    try:
        result = generate_grounded_answer(
            query="What phenomena does quantum computing harness?",
            retrieved_chunks=retrieved,
            openrouter_api_key=config.openrouter_api_key,
            model_name=config.openrouter_model,
        )
        print("✅ OpenRouter response received:")
        print(f"   • Model Used: {result['model_used']}")
        print(f"   • Answer: {result['answer']}")
        print(f"   • Citations: {result['citations']['citation_summary']}")
        print(f"   • Grounded: {result['grounded']}")

        # Test refusal/anti-hallucination
        refusal_result = generate_grounded_answer(
            query="What is the recipe for chocolate chip cookies?",
            retrieved_chunks=retrieved,
            openrouter_api_key=config.openrouter_api_key,
            model_name=config.openrouter_model,
        )
        print("\n✅ Anti-hallucination refusal test:")
        print(f"   • Query: 'What is the recipe for chocolate chip cookies?'")
        print(f"   • Response: {refusal_result['answer']}")
    except Exception as e:
        print(f"❌ OpenRouter LLM test failed: {e}")
        sys.exit(1)
    finally:
        # Clean up test vectors
        try:
            delete_document_vectors(client, config.qdrant_collection_name, test_doc_id)
            print(f"\n🧹 Cleaned up probe test vectors from Qdrant Cloud.")
        except Exception:
            pass

    print("\n" + "=" * 70)
    print("🎉 ALL SYSTEMS GO! Cloud infrastructure and RAG pipeline are fully operational.")
    print("=" * 70)


if __name__ == "__main__":
    run_cloud_validation()
