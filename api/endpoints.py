"""
API endpoints for the Multimodal RAG Chatbot
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
import tempfile
import os
import logging
from pydantic import BaseModel

from src.rag_pipeline import get_rag_pipeline
from src.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    filter_dict: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    answer: str
    source_documents: List[Dict[str, Any]]
    question: str
    context_used: Optional[str] = None

class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    pages_processed: int
    chunks_created: int
    images_extracted: int
    tables_extracted: int
    metadata: Dict[str, Any]

@router.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and process a PDF file
    """
    logger.debug(f"Received request to upload PDF: {file.filename}")
    try:
        # Validate file
        if not file.filename.endswith('.pdf'):
            logger.warning(f"Attempted upload of non-PDF file: {file.filename}")
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Check file size
        if file.size and file.size > settings.max_file_size_mb * 1024 * 1024:
            logger.warning(f"File {file.filename} exceeds size limit ({file.size} bytes).")
            raise HTTPException(
                status_code=400, 
                detail=f"File size exceeds {settings.max_file_size_mb}MB limit"
            )
        
        temp_file_path = None
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_file_path = temp_file.name
            logger.info(f"Temporary file created at: {temp_file_path}")
        
            # Process PDF with RAG pipeline
            pipeline = get_rag_pipeline()
            result = pipeline.add_document(temp_file_path)
            
            if result["success"]:
                response = {
                    "success": True,
                    "message": "PDF processed successfully",
                    "document_id": result["document_id"],
                    "filename": file.filename,
                    "pages_processed": result["pages_processed"],
                    "chunks_created": result["chunks_created"],
                    "images_extracted": result["images_extracted"],
                    "tables_extracted": result["tables_extracted"],
                    "metadata": result.get("document_metadata", {})
                }
                
                logger.info(f"Successfully uploaded and processed PDF: {file.filename}, Document ID: {result['document_id']}")
                return response
            else:
                error_detail = result.get('error', 'Unknown error')
                logger.error(f"Failed to process PDF {file.filename}: {error_detail}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to process PDF: {error_detail}"
                )
        
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                logger.debug(f"Cleaned up temporary file: {temp_file_path}")
    
    except HTTPException:
        raise # Re-raise HTTPException to be handled by FastAPI's error handling
    except Exception as e:
        logger.exception(f"Unhandled error during PDF upload for {file.filename}") # Use exception for full traceback
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the processed documents using RAG
    """
    logger.debug(f"Received query request: {request.query[:100]}...")
    try:
        if not request.query.strip():
            logger.warning("Empty query received.")
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Process query with RAG pipeline
        pipeline = get_rag_pipeline()
        result = pipeline.query(request.query, request.filter_dict)
        
        response = QueryResponse(
            answer=result["answer"],
            source_documents=result["source_documents"],
            question=result["question"],
            context_used=result.get("context_used")
        )
        
        logger.info(f"Successfully processed query: {request.query[:50]}...")
        logger.debug(f"Query response generated for: {request.query[:50]}...")
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unhandled error processing query '{request.query}'")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/query-with-scores")
async def query_documents_with_scores(
    query: str = Form(...),
    k: Optional[int] = Form(None)
):
    """
    Query documents with detailed relevance scores
    """
    logger.debug(f"Received scored query request: {query[:100]}... with k={k}")
    try:
        if not query.strip():
            logger.warning("Empty scored query received.")
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Process query with RAG pipeline
        pipeline = get_rag_pipeline()
        result = pipeline.query_with_sources(query, k)
        
        logger.info(f"Successfully processed scored query: {query[:50]}...")
        logger.debug(f"Scored query response generated for: {query[:50]}...")
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unhandled error processing scored query '{query}'")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/documents")
async def list_documents():
    """
    List all processed documents
    """
    logger.debug("Received request to list documents.")
    try:
        pipeline = get_rag_pipeline()
        documents = pipeline.get_document_list()
        
        response = {
            "documents": documents,
            "total_count": len(documents)
        }
        
        logger.info(f"Retrieved {len(documents)} documents.")
        logger.debug("Document list response generated.")
        return response
    
    except Exception as e:
        logger.exception(f"Unhandled error listing documents.")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a processed document
    """
    logger.debug(f"Received request to delete document: {document_id}")
    try:
        if not document_id.strip():
            logger.warning("Empty document ID received for deletion.")
            raise HTTPException(status_code=400, detail="Document ID cannot be empty")
        
        # Delete document with RAG pipeline
        pipeline = get_rag_pipeline()
        result = pipeline.delete_document(document_id)
        
        if result["success"]:
            logger.info(f"Successfully deleted document: {document_id}")
            return {
                "success": True,
                "message": f"Document {document_id} deleted successfully"
            }
        else:
            logger.warning(f"Document {document_id} not found or could not be deleted.")
            raise HTTPException(
                status_code=404,
                detail=f"Document {document_id} not found or could not be deleted"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unhandled error deleting document {document_id}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/stats")
async def get_system_stats():
    """
    Get system statistics and health information
    """
    logger.debug("Received request for system statistics.")
    try:
        from src.vector_store import get_vector_store
        
        # Get vector store statistics
        vector_store = get_vector_store()
        collection_stats = vector_store.get_collection_stats()
        
        # System information
        stats = {
            "system": {
                "status": "healthy",
                "version": "0.1.0",
                "model": settings.openai_model,
                "embedding_model": settings.openai_embedding_model
            },
            "vector_store": collection_stats,
            "settings": {
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "max_retrieved_docs": settings.max_retrieved_docs,
                "max_file_size_mb": settings.max_file_size_mb
            }
        }
        
        logger.info("Successfully retrieved system statistics.")
        logger.debug("System statistics response generated.")
        return stats
    
    except Exception as e:
        logger.exception(f"Unhandled error getting system stats.")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/search-by-content-type")
async def search_by_content_type(
    query: str = Form(...),
    content_type: str = Form(...),
    k: Optional[int] = Form(None)
):
    """
    Search for specific content types (text, image, table)
    """
    logger.debug(f"Received search by content type request: query='{query[:50]}...', content_type='{content_type}', k={k}")
    try:
        if not query.strip():
            logger.warning("Empty query received for content type search.")
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        if content_type not in ['text', 'image', 'table']:
            logger.warning(f"Invalid content type received: {content_type}")
            raise HTTPException(status_code=400, detail="Invalid content type. Must be 'text', 'image', or 'table'")
        
        # Search with content type filter
        from src.vector_store import get_vector_store
        vector_store = get_vector_store()
        
        documents = vector_store.search_by_content_type(query, content_type, k)
        
        # Format response
        sources = []
        for doc in documents:
            sources.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        response = {
            "query": query,
            "content_type": content_type,
            "results": sources,
            "total_found": len(sources)
        }
        
        logger.info(f"Successfully searched for {content_type} content: {query[:50]}...")
        logger.debug(f"Search by content type response generated for: {query[:50]}...")
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unhandled error searching by content type '{content_type}' with query '{query}'")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    logger.debug("Health check endpoint accessed.")
    return {
        "status": "healthy", 
        "service": "multimodal-rag-api",
        "version": "0.1.0"
    }