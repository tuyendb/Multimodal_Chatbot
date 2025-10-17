import pytest
from src.metadata_schema import DocumentMetadata, MetadataSchema

def test_document_metadata_creation():
    metadata = DocumentMetadata(
        document_id="doc123",
        file_name="report.pdf",
        file_type="application/pdf",
        upload_timestamp="2025-10-16T10:00:00Z",
        page_number=1,
        chunk_id="chunk_abc",
        chunk_type="text",
        text_content="This is a test text chunk."
    )

    assert metadata.document_id == "doc123"
    assert metadata.file_name == "report.pdf"
    assert metadata.file_type == "application/pdf"
    assert metadata.upload_timestamp == "2025-10-16T10:00:00Z"
    assert metadata.page_number == 1
    assert metadata.chunk_id == "chunk_abc"
    assert metadata.chunk_type == "text"
    assert metadata.text_content == "This is a test text chunk."
    assert metadata.image_path is None
    assert metadata.table_data is None

def test_document_metadata_optional_fields():
    metadata = DocumentMetadata(
        document_id="doc124",
        file_name="image.png",
        file_type="image/png",
        upload_timestamp="2025-10-16T10:05:00Z",
        chunk_id="chunk_def",
        chunk_type="image",
        image_path="/path/to/image.png"
    )

    assert metadata.page_number is None
    assert metadata.text_content is None
    assert metadata.image_path == "/path/to/image.png"

def test_metadata_schema_get_schema():
    schema_manager = MetadataSchema()
    schema = schema_manager.get_document_metadata_schema()

    assert schema == DocumentMetadata

def test_document_metadata_json_output():
    metadata = DocumentMetadata(
        document_id="doc125",
        file_name="table.pdf",
        file_type="application/pdf",
        upload_timestamp="2025-10-16T10:10:00Z",
        page_number=2,
        chunk_id="chunk_ghi",
        chunk_type="table",
        table_data=[["Col1", "Col2"], ["Val1", "Val2"]]
    )

    json_output = metadata.json()
    assert '"document_id": "doc125"' in json_output
    assert '"table_data": [["Col1", "Col2"], ["Val1", "Val2"]]' in json_output
