"""
PDF processing utilities following the multimodal RAG reference architecture
Uses pymupdf4llm for text extraction and supports multimodal content processing
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pymupdf4llm
import fitz  # PyMuPDF
from PIL import Image
import io
import base64

logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    Handles PDF processing operations following the multimodal RAG reference
    """
    
    def __init__(self):
        self.supported_formats = ['.pdf']
    
    def extract_markdown_content(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract content as Markdown using pymupdf4llm (following reference approach)
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of pages with markdown content and metadata
        """
        try:
            # Verify file exists
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            # Use pymupdf4llm to extract markdown content (following reference)
            markdown_pages = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
            
            processed_pages = []
            
            for page_data in markdown_pages:
                page_info = {
                    'page_number': page_data.get('page', 1),
                    'markdown_content': page_data.get('text', ''),
                    'images': page_data.get('images', []),
                    'tables': page_data.get('tables', []),
                    'metadata': {
                        'chunk_type': 'page',
                        'source_page': page_data.get('page', 1)
                    }
                }
                processed_pages.append(page_info)
            
            logger.info(f"Successfully extracted markdown content from {pdf_path}")
            return processed_pages
            
        except Exception as e:
            logger.error(f"Error extracting markdown from PDF {pdf_path}: {str(e)}")
            raise
    
    def extract_images_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract images from PDF following reference approach
        """
        try:
            images = []
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                image_list = page.get_images(full=True)
                
                for img_index, img in enumerate(image_list):
                    # Get image data
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    if pix.n - pix.alpha < 4:  # Ensure it's not a CMYK image
                        img_data = pix.tobytes("png")
                        
                        # Convert to base64 for storage/processing
                        img_base64 = base64.b64encode(img_data).decode('utf-8')
                        
                        image_info = {
                            'page_number': page_num + 1,
                            'image_index': img_index,
                            'image_data': img_base64,
                            'format': 'png',
                            'width': pix.width,
                            'height': pix.height,
                            'size_bytes': len(img_data)
                        }
                        images.append(image_info)
                    
                    pix = None  # Free memory
            
            doc.close()
            logger.info(f"Extracted {len(images)} images from {pdf_path}")
            return images
            
        except Exception as e:
            logger.error(f"Error extracting images from {pdf_path}: {str(e)}")
            return []
    
    def extract_tables_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract tables from PDF using PyMuPDF table detection
        """
        try:
            tables = []
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Find tables on the page
                table_list = page.find_tables()
                
                for table_index, table in enumerate(table_list):
                    try:
                        # Extract table data
                        table_data = table.extract()
                        
                        # Convert to more usable format
                        table_info = {
                            'page_number': page_num + 1,
                            'table_index': table_index,
                            'data': table_data,
                            'rows': len(table_data),
                            'columns': len(table_data[0]) if table_data else 0,
                            'bbox': table.bbox,  # Bounding box coordinates
                            'caption': f"Table on page {page_num + 1}"
                        }
                        tables.append(table_info)
                        
                    except Exception as e:
                        logger.warning(f"Could not extract table {table_index} on page {page_num + 1}: {str(e)}")
                        continue
            
            doc.close()
            logger.info(f"Extracted {len(tables)} tables from {pdf_path}")
            return tables
            
        except Exception as e:
            logger.error(f"Error extracting tables from {pdf_path}: {str(e)}")
            return []
    
    def process_multimodal_content(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process PDF and extract all multimodal content (following reference approach)
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing text, images, and tables with their relationships
        """
        try:
            # Extract markdown content (primary text extraction)
            markdown_pages = self.extract_markdown_content(pdf_path)
            
            # Extract images separately for detailed processing
            images = self.extract_images_from_pdf(pdf_path)
            
            # Extract tables
            tables = self.extract_tables_from_pdf(pdf_path)
            
            # Create cross-references between content types
            processed_content = self._create_cross_references(markdown_pages, images, tables)
            
            # Add document metadata
            doc_metadata = self._get_document_metadata(pdf_path)
            
            result = {
                'document_metadata': doc_metadata,
                'pages': processed_content,
                'images': images,
                'tables': tables,
                'total_pages': len(markdown_pages),
                'total_images': len(images),
                'total_tables': len(tables),
                'file_path': pdf_path
            }
            
            logger.info(f"Successfully processed multimodal content from {pdf_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing multimodal content from {pdf_path}: {str(e)}")
            raise
    
    def _create_cross_references(self, pages: List[Dict], images: List[Dict], tables: List[Dict]) -> List[Dict]:
        """
        Create cross-references between different content types per page
        """
        processed_pages = []
        
        for page in pages:
            page_num = page['page_number']
            
            # Find images on this page
            page_images = [img for img in images if img['page_number'] == page_num]
            
            # Find tables on this page
            page_tables = [table for table in tables if table['page_number'] == page_num]
            
            # Create enhanced page content
            enhanced_page = {
                **page,
                'page_images': page_images,
                'page_tables': page_tables,
                'content_summary': self._create_page_summary(page, page_images, page_tables)
            }
            
            processed_pages.append(enhanced_page)
        
        return processed_pages
    
    def _create_page_summary(self, page: Dict, images: List[Dict], tables: List[Dict]) -> str:
        """
        Create a summary of page content for better retrieval
        """
        summary_parts = []
        
        # Add text content summary
        text_content = page.get('markdown_content', '')
        if text_content:
            text_preview = text_content[:200] + "..." if len(text_content) > 200 else text_content
            summary_parts.append(f"Text: {text_preview}")
        
        # Add image descriptions
        if images:
            summary_parts.append(f"Contains {len(images)} image(s)")
            for i, img in enumerate(images):
                summary_parts.append(f"Image {i+1}: {img['width']}x{img['height']} pixels")
        
        # Add table descriptions
        if tables:
            summary_parts.append(f"Contains {len(tables)} table(s)")
            for i, table in enumerate(tables):
                summary_parts.append(f"Table {i+1}: {table['rows']} rows x {table['columns']} columns")
        
        return " | ".join(summary_parts)
    
    def _get_document_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract basic document metadata
        """
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            
            doc_info = {
                'title': metadata.get('title', ''),
                'author': metadata.get('author', ''),
                'subject': metadata.get('subject', ''),
                'creator': metadata.get('creator', ''),
                'producer': metadata.get('producer', ''),
                'creation_date': metadata.get('creationDate', ''),
                'modification_date': metadata.get('modDate', ''),
                'total_pages': len(doc),
                'is_encrypted': doc.is_encrypted,
                'file_size_bytes': os.path.getsize(pdf_path),
                'file_name': os.path.basename(pdf_path)
            }
            
            doc.close()
            return doc_info
            
        except Exception as e:
            logger.warning(f"Could not extract metadata from {pdf_path}: {str(e)}")
            return {
                'file_name': os.path.basename(pdf_path),
                'file_size_bytes': os.path.getsize(pdf_path),
                'extraction_error': str(e)
            }
    
    def get_text_for_chunking(self, pdf_path: str) -> List[str]:
        """
        Get text content prepared for chunking (following reference approach)
        Returns list of text chunks per page
        """
        try:
            processed_content = self.process_multimodal_content(pdf_path)
            
            text_chunks = []
            
            for page in processed_content['pages']:
                # Combine markdown content with content summary
                markdown_text = page.get('markdown_content', '')
                content_summary = page.get('content_summary', '')
                
                # Create rich text chunk with context
                page_chunk = f"""
Page {page['page_number']} Content:

{markdown_text}

Page Context: {content_summary}
                """.strip()
                
                text_chunks.append(page_chunk)
            
            return text_chunks
            
        except Exception as e:
            logger.error(f"Error preparing text for chunking from {pdf_path}: {str(e)}")
            raise

# Convenience functions following reference pattern
def extract_pdf_content(pdf_path: str) -> Dict[str, Any]:
    """
    Main function to extract all content from PDF following reference approach
    """
    processor = PDFProcessor()
    return processor.process_multimodal_content(pdf_path)

def get_text_chunks(pdf_path: str) -> List[str]:
    """
    Get text chunks ready for RAG processing
    """
    processor = PDFProcessor()
    return processor.get_text_for_chunking(pdf_path)