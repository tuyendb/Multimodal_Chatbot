"""
Configuration settings for the Multimodal RAG Chatbot
"""

import os
from typing import Optional
from pydantic import BaseSettings

class Settings(BaseSettings):
    # OpenAI Configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = "gpt-4-1106-preview"
    openai_embedding_model: str = "text-embedding-3-small"
    
    # Vector Store Configuration
    vector_store_type: str = os.getenv("VECTOR_STORE_TYPE", "chroma")  # chroma or pinecone
    chroma_persist_directory: str = "./chroma_db"
    pinecone_api_key: Optional[str] = os.getenv("PINECONE_API_KEY")
    pinecone_index_name: str = "multimodal-rag"
    pinecone_environment: str = "us-west1-gcp"
    
    # Application Configuration
    app_name: str = "Multimodal RAG Chatbot"
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Processing Configuration
    max_file_size_mb: int = 50
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_retrieved_docs: int = 5
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    class Config:
        env_file = ".env"

# Global settings instance
settings = Settings()