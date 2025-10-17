import pytest
import os
from unittest.mock import MagicMock, patch
from langchain.schema import Document
from src.multimodal_chunker import create_multimodal_chunks, enhance_image_chunks_with_vision, decode_base64_to_image
from src.metadata_schema import DocumentMetadata
from PIL import Image
import base64
import io

# Mock data
dummy_pdf_content = {
    "text_content": "This is some text from page 1. This is more text from page 1.",
    "page_map": [{'page_number': 1, 'content': 'This is some text from page 1. This is more text from page 1.'}],
    "images": [
        {
            "page_number": 1,
            "image_index": 0,
            "image_data": base64.b64encode(Image.new('RGB', (1, 1)).tobytes()).decode('utf-8'),
            "image_path": "/path/to/image1.png"
        }
    ],
    "tables": []
}

dummy_document_id = "test_doc_123"
dummy_file_name = "test_document.pdf"
dummy_file_type = "application/pdf"
dummy_upload_timestamp = "2025-10-16T11:00:00Z"

@pytest.fixture
def mock_text_chunker(monkeypatch):
    mock_chunker = MagicMock()
    # Simulate TextChunker returning a text chunk and an image placeholder chunk
    mock_chunker.chunk_pdf_content.return_value = [
        Document(page_content="This is some text from page 1.", metadata={'page_number': 1, 'content_type': 'text'}),
        Document(page_content="", metadata={'page_number': 1, 'content_type': 'image', 'image_index': 0})
    ]
    monkeypatch.setattr('src.multimodal_chunker.TextChunker', MagicMock(return_value=mock_chunker))

@pytest.fixture
def mock_vision_processor(monkeypatch):
    mock_analyze = MagicMock(return_value="A detailed summary of the image.")
    monkeypatch.setattr('src.multimodal_chunker.analyze_image_with_gpt4v', mock_analyze)

@pytest.fixture
def set_openai_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy_key")
    yield
    monkeypatch.delenv("OPENAI_API_KEY")

def test_decode_base64_to_image():
    dummy_image = Image.new('RGB', (10, 10), color = 'red')
    buffered = io.BytesIO()
    dummy_image.save(buffered, format="PNG")
    base64_string = base64.b64encode(buffered.getvalue()).decode('utf-8')

    decoded_image = decode_base64_to_image(base64_string)
    assert isinstance(decoded_image, Image.Image)
    assert decoded_image.size == (10, 10)

def test_enhance_image_chunks_with_vision(mock_vision_processor, set_openai_api_key):
    image_data_b64 = base64.b64encode(Image.new('RGB', (1, 1)).tobytes()).decode('utf-8')
    pdf_images = [
        {
            "page_number": 1,
            "image_index": 0,
            "image_data": image_data_b64,
            "image_path": "/path/to/image1.png"
        }
    ]
    documents = [
        Document(page_content="", metadata={'page_number': 1, 'content_type': 'image', 'image_index': 0})
    ]

    enhanced_docs = enhance_image_chunks_with_vision(
        documents, pdf_images, dummy_document_id, dummy_file_name, dummy_file_type, dummy_upload_timestamp
    )

    assert len(enhanced_docs) == 1
    assert "Image Summary: A detailed summary of the image." in enhanced_docs[0].page_content
    assert enhanced_docs[0].metadata['chunk_type'] == 'image_summary'
    assert enhanced_docs[0].metadata['document_id'] == dummy_document_id
    assert 'chunk_id' in enhanced_docs[0].metadata

def test_create_multimodal_chunks_with_vision(mock_text_chunker, mock_vision_processor, set_openai_api_key):
    chunks = create_multimodal_chunks(
        dummy_pdf_content, dummy_document_id, dummy_file_name, dummy_file_type, dummy_upload_timestamp
    )

    assert len(chunks) == 2 # One text, one image

    # Check text chunk metadata
    text_chunk = chunks[0]
    assert text_chunk.metadata['chunk_type'] == 'text'
    assert text_chunk.metadata['document_id'] == dummy_document_id
    assert 'chunk_id' in text_chunk.metadata

    # Check image chunk metadata and content
    image_chunk = chunks[1]
    assert image_chunk.metadata['chunk_type'] == 'image_summary'
    assert "Image Summary: A detailed summary of the image." in image_chunk.page_content
    assert image_chunk.metadata['document_id'] == dummy_document_id
    assert 'chunk_id' in image_chunk.metadata

def test_create_multimodal_chunks_no_vision_key(mock_text_chunker, monkeypatch):
    # Ensure OPENAI_API_KEY is not set
    if os.getenv("OPENAI_API_KEY"):
        monkeypatch.delenv("OPENAI_API_KEY")

    chunks = create_multimodal_chunks(
        dummy_pdf_content, dummy_document_id, dummy_file_name, dummy_file_type, dummy_upload_timestamp
    )

    assert len(chunks) == 2
    # Image chunk should not be enhanced, content_type should remain 'image' or original
    image_chunk = chunks[1]
    assert image_chunk.metadata['chunk_type'] == 'image' # Not 'image_summary'
    assert image_chunk.page_content == "" # Original empty content
    assert image_chunk.metadata['document_id'] == dummy_document_id
    assert 'chunk_id' in image_chunk.metadata
