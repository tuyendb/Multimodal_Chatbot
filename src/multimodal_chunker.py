import logging
from typing import List, Dict, Any
from langchain.schema import Document
from PIL import Image
import base64
import io
import os
import uuid # For generating unique chunk IDs

# Import necessary functions from other modules
# It's assumed that these modules and functions exist and are correctly implemented.
from src.vision_processor import analyze_image_with_gpt4v
from src.text_chunker import TextChunker
from src.metadata_schema import DocumentMetadata # Import the schema

logger = logging.getLogger(__name__)

def decode_base64_to_image(base64_string: str) -> Image.Image:
    """Decodes a base64 string to a PIL Image."""
    try:
        image_data = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(image_data))
    except Exception as e:
        logger.error(f"Error decoding base64 string to image: {e}")
        raise

def enhance_image_chunks_with_vision(
    documents: List[Document],
    pdf_images: List[Dict[str, Any]],
    document_id: str,
    file_name: str,
    file_type: str,
    upload_timestamp: str
) -> List[Document]:
    """
    Enhances image chunks with summaries from a vision model and populates metadata.

    Args:
        documents (List[Document]): The list of chunked documents from TextChunker.
        pdf_images (List[Dict[str, Any]]): The list of extracted image data from PDFProcessor.
        document_id (str): Unique identifier for the document.
        file_name (str): Original name of the uploaded file.
        file_type (str): Type of the file.
        upload_timestamp (str): Timestamp of upload.

    Returns:
        List[Document]: The list of documents with image chunks enhanced and metadata populated.
    """
    image_map = {(img.get('page_number'), img.get('image_index')): img for img in pdf_images}

    for doc in documents:
        if doc.metadata.get('content_type') == 'image':
            page_num = doc.metadata.get('page_number')
            img_index = doc.metadata.get('image_index')

            image_info = image_map.get((page_num, img_index))

            if image_info and 'image_data' in image_info:
                try:
                    image = decode_base64_to_image(image_info['image_data'])

                    summary_prompt = "Describe this image in detail for a RAG system. What are the key objects, scenes, and text visible in the image?"
                    summary = analyze_image_with_gpt4v(image, summary_prompt)

                    # Populate DocumentMetadata for image chunk
                    doc_metadata = DocumentMetadata(
                        document_id=document_id,
                        file_name=file_name,
                        file_type=file_type,
                        upload_timestamp=upload_timestamp,
                        page_number=page_num,
                        chunk_id=str(uuid.uuid4()), # Generate unique ID for this chunk
                        chunk_type='image_summary',
                        text_content=f"Image Summary: {summary}",
                        image_path=image_info.get('image_path'), # Assuming image_path might be available
                        # table_data=None # Not applicable for image chunks
                    )

                    doc.page_content = doc_metadata.text_content
                    doc.metadata.update(doc_metadata.dict()) # Add all metadata fields to doc.metadata
                    doc.metadata['image_base64'] = image_info['image_data'] # Keep original base64 if needed

                    logger.info(f"Enhanced image chunk on page {page_num}, index {img_index} with vision summary and metadata.")

                except Exception as e:
                    logger.error(f"Error processing image on page {page_num}, index {img_index}: {e}")
            else:
                logger.warning(f"Could not find image data for chunk on page {page_num}, index {img_index}")

    return documents

def create_multimodal_chunks(
    pdf_content: Dict[str, Any],
    document_id: str,
    file_name: str,
    file_type: str,
    upload_timestamp: str
) -> List[Document]:
    """
    Creates multimodal chunks from processed PDF content, enhancing images with AI vision and populating metadata.

    Args:
        pdf_content (Dict[str, Any]): The output from PDFProcessor.process_multimodal_content.
        document_id (str): Unique identifier for the document.
        file_name (str): Original name of the uploaded file.
        file_type (str): Type of the file.
        upload_timestamp (str): Timestamp of upload.

    Returns:
        List[Document]: A list of text and vision-enhanced image chunks with rich metadata.
    """
    text_chunker = TextChunker()
    initial_chunks = text_chunker.chunk_pdf_content(pdf_content)

    # Populate metadata for initial text chunks
    for chunk in initial_chunks:
        if chunk.metadata.get('content_type') == 'text':
            doc_metadata = DocumentMetadata(
                document_id=document_id,
                file_name=file_name,
                file_type=file_type,
                upload_timestamp=upload_timestamp,
                page_number=chunk.metadata.get('page_number'),
                chunk_id=str(uuid.uuid4()), # Generate unique ID for this chunk
                chunk_type='text',
                text_content=chunk.page_content
            )
            chunk.metadata.update(doc_metadata.dict())
        # Handle other types of initial chunks if TextChunker produces them (e.g., tables)
        # For now, assuming TextChunker primarily produces text chunks.

    if os.getenv("OPENAI_API_KEY"):
        pdf_images = pdf_content.get('images', [])
        enhanced_chunks = enhance_image_chunks_with_vision(
            initial_chunks,
            pdf_images,
            document_id,
            file_name,
            file_type,
            upload_timestamp
        )
        logger.info(f"Created and enhanced {len(enhanced_chunks)} multimodal chunks.")
        return enhanced_chunks
    else:
        logger.warning("OPENAI_API_KEY not set. Skipping vision enhancement for image chunks.")
        return initial_chunks

if __name__ == '__main__':
    # This block is for demonstration and testing purposes.
    # To run this, you would need a sample PDF and a valid OpenAI API key.

    # from src.pdf_processor import extract_pdf_content

    # # Example usage:
    # api_key = os.getenv("OPENAI_API_KEY")
    # if api_key:
    #     # Provide a path to a PDF file with images
    #     pdf_file_path = "path/to/your/document.pdf"

    #     if os.path.exists(pdf_file_path):
    #         # 1. Process the PDF to get all multimodal content
    #         print(f"Processing PDF: {pdf_file_path}")
    #         pdf_data = extract_pdf_content(pdf_file_path)

    #         # Dummy metadata for testing
    #         test_document_id = "test_doc_123"
    #         test_file_name = "test_document.pdf"
    #         test_file_type = "application/pdf"
    #         test_upload_timestamp = "2025-10-16T11:00:00Z"

    #         # 2. Create multimodal chunks with vision enhancement
    #         print("Creating multimodal chunks...")
    #         multimodal_chunks = create_multimodal_chunks(
    #             pdf_data,
    #             test_document_id,
    #             test_file_name,
    #             test_file_type,
    #             test_upload_timestamp
    #         )

    #         # 3. Display the results for image chunks
    #         print("\n--- Results for Multimodal Chunks ---")
    #         for i, chunk in enumerate(multimodal_chunks):
    #             print(f"\nChunk {i+1}:")
    #             print(f"  Page Content: {chunk.page_content[:100]}...") # Truncate for display
    #             print(f"  Metadata: {chunk.metadata}")

    #     else:
    #         print(f"Error: PDF file not found at '{pdf_file_path}'")
    # else:
    #     print("Skipping example run: OPENAI_API_KEY environment variable is not set.")

    print("Multimodal chunker script is ready to be used with enhanced metadata.")
