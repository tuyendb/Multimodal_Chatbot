import pytest
from unittest.mock import MagicMock, patch
from langchain.schema import Document
from src.workflow_graph import WorkflowGraph, GraphState
from src.vector_store import VectorStore
from src.metadata_schema import DocumentMetadata
import uuid
from datetime import datetime

# Dummy data for testing
doc_id = "test_doc_wf_1"
file_name = "test_workflow.pdf"
file_type = "application/pdf"
upload_ts = datetime.now().isoformat()
query_text = "What is the main topic?"

@pytest.fixture
def mock_components(monkeypatch):
    # Mock PDFProcessor
    mock_pdf_processor = MagicMock()
    mock_pdf_processor.process_multimodal_content.return_value = {
        "text_content": "Workflow text content.",
        "page_map": [{'page_number': 1, 'content': 'Workflow text content.'}],
        "images": [],
        "tables": []
    }
    monkeypatch.setattr('src.workflow_graph.PDFProcessor', MagicMock(return_value=mock_pdf_processor))

    # Mock ImageExtractor (not directly used in workflow_graph, but good practice)
    mock_image_extractor = MagicMock()
    monkeypatch.setattr('src.workflow_graph.ImageExtractor', MagicMock(return_value=mock_image_extractor))

    # Mock TableExtractor (not directly used in workflow_graph, but good practice)
    mock_table_extractor = MagicMock()
    monkeypatch.setattr('src.workflow_graph.TableExtractor', MagicMock(return_value=mock_table_extractor))

    # Mock TextChunker (its create_multimodal_chunks is mocked separately)
    mock_text_chunker = MagicMock()
    monkeypatch.setattr('src.workflow_graph.TextChunker', MagicMock(return_value=mock_text_chunker))

    # Mock create_multimodal_chunks function
    mock_multimodal_chunks = [
        Document(
            page_content="Chunk 1 content",
            metadata=DocumentMetadata(
                document_id=doc_id, file_name=file_name, file_type=file_type,
                upload_timestamp=upload_ts, page_number=1, chunk_id=str(uuid.uuid4()),
                chunk_type="text", text_content="Chunk 1 content"
            ).dict()
        )
    ]
    monkeypatch.setattr('src.workflow_graph.create_multimodal_chunks', MagicMock(return_value=mock_multimodal_chunks))

    # Mock VectorStore
    mock_vector_store = MagicMock(spec=VectorStore)
    monkeypatch.setattr('src.workflow_graph.VectorStore', MagicMock(return_value=mock_vector_store))

    # Mock CrossModalRetriever
    mock_cross_modal_retriever = MagicMock()
    mock_cross_modal_retriever.index_multimodal_chunks.return_value = None
    mock_cross_modal_retriever.retrieve_relevant_content.return_value = [
        Document(page_content="Retrieved relevant document content.", metadata={"source": "test"})
    ]
    monkeypatch.setattr('src.workflow_graph.CrossModalRetriever', MagicMock(return_value=mock_cross_modal_retriever))

    return {
        "pdf_processor": mock_pdf_processor,
        "create_multimodal_chunks": monkeypatch.get_original('src.workflow_graph.create_multimodal_chunks'), # Keep original for patching
        "cross_modal_retriever": mock_cross_modal_retriever,
        "vector_store": mock_vector_store
    }

def test_workflow_graph_initialization(mock_components):
    vector_store_instance = MagicMock(spec=VectorStore)
    graph = WorkflowGraph(vector_store_instance)
    assert graph.pdf_processor is not None
    assert graph.cross_modal_retriever is not None
    assert graph.workflow is not None

def test_extract_pdf_content_node(mock_components):
    graph = WorkflowGraph(mock_components["vector_store"])
    initial_state = GraphState(pdf_path="/fake/path/doc.pdf", document_id="", file_name="", file_type="", upload_timestamp="", query="", pdf_content={}, multimodal_chunks=[], retrieved_documents=[], answer="", error="")
    result = graph.extract_pdf_content_node(initial_state)
    assert "pdf_content" in result
    assert result["pdf_content"]["text_content"] == "Workflow text content."
    mock_components["pdf_processor"].process_multimodal_content.assert_called_once_with("/fake/path/doc.pdf")

def test_create_chunks_node(mock_components):
    graph = WorkflowGraph(mock_components["vector_store"])
    initial_state = GraphState(
        pdf_path="", document_id=doc_id, file_name=file_name, file_type=file_type,
        upload_timestamp=upload_ts, query="",
        pdf_content={
            "text_content": "Some text", "page_map": [], "images": [], "tables": []
        },
        multimodal_chunks=[], retrieved_documents=[], answer="", error=""
    )
    with patch('src.workflow_graph.create_multimodal_chunks', return_value=mock_components["create_multimodal_chunks"].return_value) as mock_create_chunks:
        result = graph.create_chunks_node(initial_state)
        assert "multimodal_chunks" in result
        assert len(result["multimodal_chunks"]) == 1
        mock_create_chunks.assert_called_once_with(
            initial_state["pdf_content"],
            doc_id, file_name, file_type, upload_ts
        )

def test_index_chunks_node(mock_components):
    graph = WorkflowGraph(mock_components["vector_store"])
    mock_multimodal_chunks = [
        Document(page_content="Test chunk", metadata={"chunk_id": "123"})
    ]
    initial_state = GraphState(
        pdf_path="", document_id="", file_name="", file_type="", upload_timestamp="", query="",
        pdf_content={}, multimodal_chunks=mock_multimodal_chunks, retrieved_documents=[], answer="", error=""
    )
    result = graph.index_chunks_node(initial_state)
    assert result == {}
    mock_components["cross_modal_retriever"].index_multimodal_chunks.assert_called_once_with(mock_multimodal_chunks)

def test_retrieve_content_node(mock_components):
    graph = WorkflowGraph(mock_components["vector_store"])
    initial_state = GraphState(
        pdf_path="", document_id="", file_name="", file_type="", upload_timestamp="",
        query=query_text, pdf_content={}, multimodal_chunks=[], retrieved_documents=[], answer="", error=""
    )
    result = graph.retrieve_content_node(initial_state)
    assert "retrieved_documents" in result
    assert len(result["retrieved_documents"]) == 1
    mock_components["cross_modal_retriever"].retrieve_relevant_content.assert_called_once_with(query_text)

def test_generate_answer_node(mock_components):
    graph = WorkflowGraph(mock_components["vector_store"])
    mock_retrieved_docs = [
        Document(page_content="This is the retrieved content for the answer.", metadata={})
    ]
    initial_state = GraphState(
        pdf_path="", document_id="", file_name="", file_type="", upload_timestamp="",
        query=query_text, pdf_content={}, multimodal_chunks=[], retrieved_documents=mock_retrieved_docs, answer="", error=""
    )
    result = graph.generate_answer_node(initial_state)
    assert "answer" in result
    assert "Based on the retrieved documents" in result["answer"]
    assert query_text in result["answer"]

def test_run_workflow_success(mock_components):
    vector_store_instance = MagicMock(spec=VectorStore)
    graph = WorkflowGraph(vector_store_instance)

    initial_state = GraphState(
        pdf_path="/fake/path/doc.pdf",
        document_id=doc_id,
        file_name=file_name,
        file_type=file_type,
        upload_timestamp=upload_ts,
        query=query_text,
        pdf_content={}, multimodal_chunks=[], retrieved_documents=[], answer="", error=""
    )

    # Patch create_multimodal_chunks for the full workflow run
    with patch('src.workflow_graph.create_multimodal_chunks', return_value=mock_components["create_multimodal_chunks"].return_value):
        final_state = graph.run_workflow(initial_state)

    assert final_state["answer"] != ""
    assert final_state["pdf_content"]["text_content"] == "Workflow text content."
    assert len(final_state["multimodal_chunks"]) == 1
    assert len(final_state["retrieved_documents"]) == 1

    # Verify calls to key components
    mock_components["pdf_processor"].process_multimodal_content.assert_called_once()
    mock_components["cross_modal_retriever"].index_multimodal_chunks.assert_called_once()
    mock_components["cross_modal_retriever"].retrieve_relevant_content.assert_called_once()
