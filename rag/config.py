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
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "recursive/flint-v1.0:free").strip()
    
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
        issues.append("OPENROUTER_MODEL is empty. Defaulting to 'recursive/flint-v1.0:free'.")

    return len(issues) == 0, issues
