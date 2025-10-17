"""
Text chunking strategies using LangChain following the multimodal RAG reference
Supports different chunking approaches for optimal RAG performance
"""

import logging
from typing import List, Dict, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain.schema import Document
from src.config import settings

logger = logging.getLogger(__name__)

class TextChunker:
    """
    Handles text chunking using LangChain splitters
    Following the reference approach for optimal multimodal RAG
    """
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        # Initialize different splitters for different content types
        self.markdown_splitter = MarkdownTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk_pdf_content(self, pdf_content: Dict[str, Any]) -> List[Document]:
        """
        Chunk PDF content into manageable pieces for RAG
        
        Args:
            pdf_content: Processed PDF content from pdf_processor
            
        Returns:
            List of LangChain Document objects with metadata
        """
        try:
            documents = []
            
            # Process each page
            for page in pdf_content['pages']:
                page_docs = self._chunk_page_content(page, pdf_content['document_metadata'])
                documents.extend(page_docs)
            
            logger.info(f"Created {len(documents)} chunks from PDF content")
            return documents
            
        except Exception as e:
            logger.error(f"Error chunking PDF content: {str(e)}")
            raise
    
    def _chunk_page_content(self, page: Dict[str, Any], doc_metadata: Dict[str, Any]) -> List[Document]:
        """
        Chunk individual page content while preserving context
        """
        documents = []
        page_num = page['page_number']
        
        # Extract main text content
        markdown_content = page.get('markdown_content', '')
        
        if markdown_content.strip():
            # Use markdown splitter for better preservation of structure
            chunks = self.markdown_splitter.split_text(markdown_content)
            
            for i, chunk in enumerate(chunks):
                # Create rich metadata for each chunk
                metadata = {
                    # Page information
                    'page_number': page_num,
                    'chunk_index': i,
                    'total_chunks_on_page': len(chunks),
                    
                    # Document information
                    'document_title': doc_metadata.get('title', ''),
                    'document_author': doc_metadata.get('author', ''),
                    'document_filename': doc_metadata.get('file_name', ''),
                    
                    # Content type information
                    'content_type': 'text',
                    'has_images': len(page.get('page_images', [])) > 0,
                    'has_tables': len(page.get('page_tables', [])) > 0,
                    'image_count': len(page.get('page_images', [])),
                    'table_count': len(page.get('page_tables', [])),
                    
                    # Chunk information
                    'chunk_size': len(chunk),
                    'chunk_type': 'markdown',
                    
                    # Cross-modal references
                    'related_images': [img['image_index'] for img in page.get('page_images', [])],
                    'related_tables': [table['table_index'] for table in page.get('page_tables', [])]
                }
                
                # Add content summary if available
                content_summary = page.get('content_summary', '')
                if content_summary:
                    metadata['page_context'] = content_summary
                
                # Create Document object
                doc = Document(
                    page_content=chunk,
                    metadata=metadata
                )
                documents.append(doc)
        
        # Create separate documents for images if they exist
        for img in page.get('page_images', []):
            image_metadata = {
                'page_number': page_num,
                'content_type': 'image',
                'image_index': img['image_index'],
                'image_format': img['format'],
                'image_width': img['width'],
                'image_height': img['height'],
                'image_size_bytes': img['size_bytes'],
                'document_title': doc_metadata.get('title', ''),
                'document_filename': doc_metadata.get('file_name', ''),
                'chunk_type': 'image_reference'
            }
            
            # Create a text description for the image
            image_description = f"""
            Image on page {page_num}:
            - Format: {img['format']}
            - Size: {img['width']}x{img['height']} pixels
            - File size: {img['size_bytes']} bytes
            - Page context: {page.get('content_summary', 'No context available')}
            """.strip()
            
            image_doc = Document(
                page_content=image_description,
                metadata=image_metadata
            )
            documents.append(image_doc)
        
        # Create separate documents for tables if they exist
        for table in page.get('page_tables', []):
            table_metadata = {
                'page_number': page_num,
                'content_type': 'table',
                'table_index': table['table_index'],
                'table_rows': table['rows'],
                'table_columns': table['columns'],
                'table_caption': table.get('caption', ''),
                'document_title': doc_metadata.get('title', ''),
                'document_filename': doc_metadata.get('file_name', ''),
                'chunk_type': 'table_reference'
            }
            
            # Create a text description for the table
            table_description = f"""
            Table on page {page_num}:
            - Dimensions: {table['rows']} rows x {table['columns']} columns
            - Caption: {table.get('caption', 'No caption')}
            - Page context: {page.get('content_summary', 'No context available')}
            """.strip()
            
            table_doc = Document(
                page_content=table_description,
                metadata=table_metadata
            )
            documents.append(table_doc)
        
        return documents
    
    def chunk_text_list(self, text_chunks: List[str], source_metadata: Dict[str, Any] = None) -> List[Document]:
        """
        Chunk a list of text strings (simpler interface)
        """
        try:
            documents = []
            
            for i, text_chunk in enumerate(text_chunks):
                if not text_chunk.strip():
                    continue
                
                # Use recursive splitter for general text
                sub_chunks = self.recursive_splitter.split_text(text_chunk)
                
                for j, sub_chunk in enumerate(sub_chunks):
                    metadata = {
                        'source_index': i,
                        'chunk_index': j,
                        'total_chunks': len(sub_chunks),
                        'content_type': 'text',
                        'chunk_size': len(sub_chunk),
                        'chunk_type': 'recursive'
                    }
                    
                    # Add source metadata if provided
                    if source_metadata:
                        metadata.update(source_metadata)
                    
                    doc = Document(
                        page_content=sub_chunk,
                        metadata=metadata
                    )
                    documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error chunking text list: {str(e)}")
            raise
    
    def create_summary_chunks(self, documents: List[Document], max_summary_length: int = 200) -> List[Document]:
        """
        Create summary chunks for better retrieval
        """
        try:
            summary_docs = []
            
            for doc in documents:
                content = doc.page_content
                if len(content) > max_summary_length:
                    summary = content[:max_summary_length] + "..."
                else:
                    summary = content
                
                # Create enhanced metadata for summary
                summary_metadata = doc.metadata.copy()
                summary_metadata.update({
                    'is_summary': True,
                    'original_chunk_size': len(content),
                    'summary_type': 'truncated'
                })
                
                summary_doc = Document(
                    page_content=summary,
                    metadata=summary_metadata
                )
                summary_docs.append(summary_doc)
            
            return summary_docs
            
        except Exception as e:
            logger.error(f"Error creating summary chunks: {str(e)}")
            return []
    
    def filter_chunks_by_size(self, documents: List[Document], min_size: int = 50, max_size: int = 2000) -> List[Document]:
        """
        Filter chunks by size to ensure optimal retrieval
        """
        filtered_docs = []
        
        for doc in documents:
            chunk_size = len(doc.page_content)
            if min_size <= chunk_size <= max_size:
                filtered_docs.append(doc)
            else:
                logger.debug(f"Filtered out chunk of size {chunk_size} (min: {min_size}, max: {max_size})")
        
        logger.info(f"Filtered {len(documents)} chunks to {len(filtered_docs)} valid chunks")
        return filtered_docs
    
    def get_chunk_statistics(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Get statistics about the created chunks
        """
        if not documents:
            return {
                'total_chunks': 0,
                'average_chunk_size': 0,
                'min_chunk_size': 0,
                'max_chunk_size': 0,
                'content_types': {}
            }
        
        chunk_sizes = [len(doc.page_content) for doc in documents]
        content_types = {}
        
        for doc in documents:
            content_type = doc.metadata.get('content_type', 'unknown')
            content_types[content_type] = content_types.get(content_type, 0) + 1
        
        stats = {
            'total_chunks': len(documents),
            'average_chunk_size': sum(chunk_sizes) / len(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'total_characters': sum(chunk_sizes),
            'content_types': content_types,
            'pages_covered': len(set(doc.metadata.get('page_number', 0) for doc in documents))
        }
        
        return stats

# Convenience functions
def chunk_pdf_content(pdf_content: Dict[str, Any]) -> List[Document]:
    """
    Convenience function to chunk PDF content
    """
    chunker = TextChunker()
    return chunker.chunk_pdf_content(pdf_content)

def chunk_text_for_rag(texts: List[str], metadata: Dict[str, Any] = None) -> List[Document]:
    """
    Convenience function to chunk text for RAG
    """
    chunker = TextChunker()
    return chunker.chunk_text_list(texts, metadata)