"""
Multimodal RAG Chatbot - Main Application Entry Point
FastAPI application for PDF processing and multimodal question answering
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from api.endpoints import router
from src.config import settings

# Configure logging
logging.basicConfig(level=settings.log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multimodal RAG Chatbot",
    description="API for processing PDFs and answering questions using multimodal RAG",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    logger.info("Root endpoint accessed.")
    return {"message": "Multimodal RAG Chatbot API", "version": "0.1.0"}

@app.get("/health")
async def health_check():
    logger.info("Health check endpoint accessed.")
    return {"status": "healthy"}

if __name__ == "__main__":
    logger.info(f"Starting Uvicorn server at http://{settings.api_host}:{settings.api_port}")
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )