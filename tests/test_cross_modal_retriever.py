import pytest
from unittest.mock import MagicMock
from langchain.schema import Document
from src.cross_modal_retriever import CrossModalRetriever
from src.metadata_schema import DocumentMetadata
from src.vector_store import VectorStore # Import the abstract base class
import uuid
from datetime import datetime

# Dummy data for testing
doc_id = "test_doc_1"
file_name = "test_document.pdf"
file_type = "application/pdf"
upload_ts = datetime.now().isoformat()

@pytest.fixture
def mock_vector_store():
    mock_store = MagicMock(spec=VectorStore)
    mock_store.add_documents.return_value = None
    mock_store.similarity_search.return_value = [] # Default empty return
    return mock_store

@pytest.fixture
def sample_multimodal_chunks():
    text_chunk_metadata = DocumentMetadata(
        document_id=doc_id,
        file_name=file_name,
        file_type=file_type,
        upload_timestamp=upload_ts,
        page_number=1,
        chunk_id=str(uuid.uuid4()),
        chunk_type="text",
        text_content="This is a text chunk about multimodal RAG systems."
    )
    text_doc = Document(page_content=text_chunk_metadata.text_content, metadata=text_chunk_metadata.dict())

    image_chunk_metadata = DocumentMetadata(
        document_id=doc_id,
        file_name=file_name,
        file_type=file_type,
        upload_timestamp=upload_ts,
        page_number=2,
        chunk_id=str(uuid.uuid4()),
        chunk_type="image_summary",
        text_content="Image summary: A diagram showing a RAG pipeline with text and image inputs.",
        image_path="/path/to/image.png"
    )
    image_doc = Document(page_content=image_chunk_metadata.text_content, metadata=image_chunk_metadata.dict())

    table_chunk_metadata = DocumentMetadata(
        document_id=doc_id,
        file_name=file_name,
        file_type=file_type,
        upload_timestamp=upload_ts,
        page_number=3,
        chunk_id=str(uuid.uuid4()),
        chunk_type="table",
        text_content="Table data: Column1, Column2\nValueA, ValueB",
        table_data=[["Col1", "Col2"], ["Val1", "Val2"]]
    )
    table_doc = Document(page_content=table_chunk_metadata.text_content, metadata=table_chunk_metadata.dict())

    return [text_doc, image_doc, table_doc]

def test_index_multimodal_chunks(mock_vector_store, sample_multimodal_chunks):
    retriever = CrossModalRetriever(mock_vector_store)
    retriever.index_multimodal_chunks(sample_multimodal_chunks)

    mock_vector_store.add_documents.assert_called_once_with(sample_multimodal_chunks)

def test_retrieve_relevant_content(mock_vector_store):
    query = "multimodal RAG"
    expected_docs = [
        Document(page_content="Relevant text about multimodal RAG", metadata={"chunk_type": "text"})
    ]
    mock_vector_store.similarity_search.return_value = expected_docs

    retriever = CrossModalRetriever(mock_vector_store)
    retrieved = retriever.retrieve_relevant_content(query, k=1, filters={"chunk_type": "text"})

    mock_vector_store.similarity_search.assert_called_once_with(query, k=1, filter={"chunk_type": "text"})
    assert retrieved == expected_docs

def test_retrieve_no_results(mock_vector_store):
    query = "non-existent topic"
    mock_vector_store.similarity_search.return_value = []

    retriever = CrossModalRetriever(mock_vector_store)
    retrieved = retriever.retrieve_relevant_content(query)

    assert retrieved == []

def test_index_error_handling(mock_vector_store, sample_multimodal_chunks):
    mock_vector_store.add_documents.side_effect = Exception("Indexing failed")
    retriever = CrossModalRetriever(mock_vector_store)

    with pytest.raises(Exception, match="Indexing failed"):
        retriever.index_multimodal_chunks(sample_multimodal_chunks)

def test_retrieve_error_handling(mock_vector_store):
    mock_vector_store.similarity_search.side_effect = Exception("Retrieval failed")
    retriever = CrossModalRetriever(mock_vector_store)

    with pytest.raises(Exception, match="Retrieval failed"):
        retriever.retrieve_relevant_content("query")
