"""
Vector store management using Chroma and OpenAI embeddings
Following the reference architecture for multimodal RAG
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
from uuid import uuid4
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from src.config import settings

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """
    Manages vector store operations using Chroma and OpenAI embeddings
    Following the reference approach for multimodal RAG
    """
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            openai_api_key=settings.openai_api_key
        )
        
        # Initialize Chroma client
        self.chroma_client = chromadb.PersistentClient(
            path=settings.chroma_persist_directory
        )
        
        # Collection name
        self.collection_name = "multimodal_rag_documents"
        
        # Initialize vector store
        self.vector_store = None
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """
        Initialize the Chroma vector store
        """
        try:
            self.vector_store = Chroma(
                client=self.chroma_client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings
            )
            logger.info(f"Initialized Chroma vector store: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Document], document_id: str = None) -> List[str]:
        """
        Add documents to the vector store
        
        Args:
            documents: List of LangChain Document objects
            document_id: Optional document identifier for grouping
            
        Returns:
            List of document IDs
        """
        try:
            if not documents:
                logger.warning("No documents provided to add")
                return []
            
            # Add document ID to metadata if provided
            if document_id:
                for doc in documents:
                    doc.metadata['document_id'] = document_id
            
            # Add documents to vector store
            ids = self.vector_store.add_documents(documents)
            
            logger.info(f"Added {len(documents)} documents to vector store")
            return ids
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise
    
    def add_pdf_content(self, pdf_content: Dict[str, Any], document_id: str = None) -> List[str]:
        """
        Add processed PDF content to vector store
        
        Args:
            pdf_content: Processed PDF content from pdf_processor
            document_id: Optional document identifier
            
        Returns:
            List of document IDs
        """
        try:
            # Generate document ID if not provided
            if not document_id:
                document_id = str(uuid4())
            
            # Import text chunker here to avoid circular imports
            from src.text_chunker import TextChunker
            
            # Chunk the PDF content
            chunker = TextChunker()
            documents = chunker.chunk_pdf_content(pdf_content)
            
            # Add document-level metadata
            doc_metadata = pdf_content.get('document_metadata', {})
            
            for doc in documents:
                # Enhance metadata with document information
                doc.metadata.update({
                    'document_id': document_id,
                    'document_title': doc_metadata.get('title', ''),
                    'document_author': doc_metadata.get('author', ''),
                    'document_filename': doc_metadata.get('file_name', ''),
                    'total_pages': pdf_content.get('total_pages', 0),
                    'total_images': pdf_content.get('total_images', 0),
                    'total_tables': pdf_content.get('total_tables', 0)
                })
            
            # Add to vector store
            ids = self.add_documents(documents, document_id)
            
            logger.info(f"Added PDF content with {len(documents)} chunks to vector store")
            return ids
            
        except Exception as e:
            logger.error(f"Error adding PDF content to vector store: {str(e)}")
            raise
    
    def similarity_search(self, query: str, k: int = None, filter_dict: Dict[str, Any] = None) -> List[Document]:
        """
        Perform similarity search
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional filter dictionary
            
        Returns:
            List of similar documents
        """
        try:
            k = k or settings.max_retrieved_docs
            
            # Prepare search arguments
            search_kwargs = {'k': k}
            if filter_dict:
                search_kwargs['filter'] = filter_dict
            
            # Perform similarity search
            results = self.vector_store.similarity_search(
                query,
                **search_kwargs
            )
            
            logger.info(f"Found {len(results)} documents for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = None, filter_dict: Dict[str, Any] = None) -> List[tuple]:
        """
        Perform similarity search with scores
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional filter dictionary
            
        Returns:
            List of (Document, score) tuples
        """
        try:
            k = k or settings.max_retrieved_docs
            
            # Prepare search arguments
            search_kwargs = {'k': k}
            if filter_dict:
                search_kwargs['filter'] = filter_dict
            
            # Perform similarity search with scores
            results = self.vector_store.similarity_search_with_score(
                query,
                **search_kwargs
            )
            
            logger.info(f"Found {len(results)} scored documents for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error performing similarity search with scores: {str(e)}")
            return []
    
    def search_by_content_type(self, query: str, content_type: str, k: int = None) -> List[Document]:
        """
        Search for specific content types (text, image, table)
        
        Args:
            query: Search query
            content_type: Type of content to search for
            k: Number of results to return
            
        Returns:
            List of matching documents
        """
        filter_dict = {'content_type': content_type}
        return self.similarity_search(query, k, filter_dict)
    
    def search_by_document(self, query: str, document_id: str, k: int = None) -> List[Document]:
        """
        Search within a specific document
        
        Args:
            query: Search query
            document_id: Document ID to search within
            k: Number of results to return
            
        Returns:
            List of matching documents
        """
        filter_dict = {'document_id': document_id}
        return self.similarity_search(query, k, filter_dict)
    
    def search_by_page_range(self, query: str, page_start: int, page_end: int, k: int = None) -> List[Document]:
        """
        Search within a specific page range
        
        Args:
            query: Search query
            page_start: Start page number
            page_end: End page number
            k: Number of results to return
            
        Returns:
            List of matching documents
        """
        filter_dict = {
            'page_number': {
                '$gte': page_start,
                '$lte': page_end
            }
        }
        return self.similarity_search(query, k, filter_dict)
    
    def get_document_by_id(self, document_id: str) -> List[Document]:
        """
        Retrieve all chunks for a specific document
        
        Args:
            document_id: Document ID
            
        Returns:
            List of document chunks
        """
        filter_dict = {'document_id': document_id}
        try:
            # Note: Chroma doesn't have a direct get_by_filter method
            # We'll use a broad search and filter
            all_docs = self.vector_store.get()
            
            # Filter documents by metadata
            filtered_docs = []
            for i, metadata in enumerate(all_docs.get('metadatas', [])):
                if metadata.get('document_id') == document_id:
                    # Recreate Document object
                    doc_id = all_docs['ids'][i]
                    content = all_docs['documents'][i]
                    
                    doc = Document(
                        page_content=content,
                        metadata=metadata
                    )
                    filtered_docs.append(doc)
            
            return filtered_docs
            
        except Exception as e:
            logger.error(f"Error retrieving document {document_id}: {str(e)}")
            return []
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and all its chunks from the vector store
        
        Args:
            document_id: Document ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all documents for the document ID
            docs_to_delete = self.get_document_by_id(document_id)
            
            if not docs_to_delete:
                logger.warning(f"No documents found for document_id: {document_id}")
                return True
            
            # Extract the document IDs from Chroma
            all_docs = self.vector_store.get()
            ids_to_delete = []
            
            for i, metadata in enumerate(all_docs.get('metadatas', [])):
                if metadata.get('document_id') == document_id:
                    ids_to_delete.append(all_docs['ids'][i])
            
            if ids_to_delete:
                self.vector_store.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} chunks for document {document_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store collection
        """
        try:
            collection = self.chroma_client.get_collection(self.collection_name)
            count = collection.count()
            
            # Get sample documents to analyze content types
            sample_results = collection.get(limit=1000)
            
            content_types = {}
            documents_per_doc = {}
            
            for metadata in sample_results.get('metadatas', []):
                # Count content types
                content_type = metadata.get('content_type', 'unknown')
                content_types[content_type] = content_types.get(content_type, 0) + 1
                
                # Count documents per document
                doc_id = metadata.get('document_id', 'unknown')
                documents_per_doc[doc_id] = documents_per_doc.get(doc_id, 0) + 1
            
            stats = {
                'total_chunks': count,
                'unique_documents': len(documents_per_doc),
                'content_types': content_types,
                'average_chunks_per_document': count / len(documents_per_doc) if documents_per_doc else 0,
                'collection_name': self.collection_name,
                'embedding_model': settings.openai_embedding_model
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {}
    
    def reset_collection(self) -> bool:
        """
        Reset the entire collection (use with caution)
        """
        try:
            self.chroma_client.delete_collection(self.collection_name)
            self._initialize_vector_store()
            logger.info(f"Reset collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting collection: {str(e)}")
            return False

# Global vector store instance
_vector_store_manager = None

def get_vector_store() -> VectorStoreManager:
    """
    Get or create the global vector store manager instance
    """
    global _vector_store_manager
    if _vector_store_manager is None:
        _vector_store_manager = VectorStoreManager()
    return _vector_store_manager

# Convenience functions
def add_pdf_to_vector_store(pdf_content: Dict[str, Any], document_id: str = None) -> List[str]:
    """
    Convenience function to add PDF content to vector store
    """
    vector_store = get_vector_store()
    return vector_store.add_pdf_content(pdf_content, document_id)

def search_documents(query: str, k: int = None) -> List[Document]:
    """
    Convenience function to search documents
    """
    vector_store = get_vector_store()
    return vector_store.similarity_search(query, k)