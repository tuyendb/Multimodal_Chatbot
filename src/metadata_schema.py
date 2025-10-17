from pydantic import BaseModel, Field
from typing import List, Optional

class DocumentMetadata(BaseModel):
    document_id: str = Field(..., description="Unique identifier for the document.")
    file_name: str = Field(..., description="Original name of the uploaded file.")
    file_type: str = Field(..., description="Type of the file (e.g., 'application/pdf').")
    upload_timestamp: str = Field(..., description="Timestamp of when the document was uploaded.")
    page_number: Optional[int] = Field(None, description="Page number if the chunk originates from a specific page.")
    chunk_id: str = Field(..., description="Unique identifier for the chunk.")
    chunk_type: str = Field(..., description="Type of content in the chunk (e.g., 'text', 'image', 'table').")
    text_content: Optional[str] = Field(None, description="Extracted text content of the chunk.")
    image_path: Optional[str] = Field(None, description="Path to the extracted image file if applicable.")
    table_data: Optional[List[List[str]]] = Field(None, description="Structured data for tables, if applicable.")
    # Add more fields as needed for cross-modal linking, e.g., bounding box coordinates, relationships

class MetadataSchema:
    def __init__(self):
        pass

    def get_document_metadata_schema(self):
        """
        Returns the Pydantic model for document metadata.
        """
        return DocumentMetadata

if __name__ == '__main__':
    # Example usage
    metadata_schema = MetadataSchema()
    schema = metadata_schema.get_document_metadata_schema()
    print("Metadata Schema Definition:")
    print(schema.schema_json(indent=2))

    # Example of creating a metadata instance
    example_metadata = DocumentMetadata(
        document_id="doc_123",
        file_name="example.pdf",
        file_type="application/pdf",
        upload_timestamp="2025-10-16T10:00:00Z",
        page_number=1,
        chunk_id="chunk_abc",
        chunk_type="text",
        text_content="This is some text from the document."
    )
    print("\nExample Metadata Instance:")
    print(example_metadata.json(indent=2))
