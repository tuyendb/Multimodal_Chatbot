import operator
from typing import List, Dict, Any, TypedDict
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

# Import necessary components from our modules
from src.pdf_processor import PDFProcessor
from src.image_extractor import ImageExtractor
from src.table_extractor import TableExtractor
from src.text_chunker import TextChunker
from src.multimodal_chunker import create_multimodal_chunks
from src.vector_store import VectorStore # Assuming a concrete implementation like ChromaVectorStore
from src.cross_modal_retriever import CrossModalRetriever
from src.metadata_schema import DocumentMetadata

# Define a State for our LangGraph workflow
class GraphState(TypedDict):
    """
    Represents the state of our multimodal RAG workflow.
    """
    pdf_path: str
    document_id: str
    file_name: str
    file_type: str
    upload_timestamp: str
    pdf_content: Dict[str, Any]
    multimodal_chunks: List[Document]
    retrieved_documents: List[Document]
    query: str
    answer: str
    error: str

class WorkflowGraph:
    def __init__(self, vector_store: VectorStore):
        self.pdf_processor = PDFProcessor()
        self.image_extractor = ImageExtractor()
        self.table_extractor = TableExtractor()
        self.text_chunker = TextChunker()
        self.cross_modal_retriever = CrossModalRetriever(vector_store)
        self.workflow = self._build_workflow()

    def _build_workflow(self):
        workflow = StateGraph(GraphState)

        # Define nodes for each processing step
        workflow.add_node("extract_pdf_content", self.extract_pdf_content_node)
        workflow.add_node("create_chunks", self.create_chunks_node)
        workflow.add_node("index_chunks", self.index_chunks_node)
        workflow.add_node("retrieve_content", self.retrieve_content_node)
        workflow.add_node("generate_answer", self.generate_answer_node)

        # Define the entry point
        workflow.set_entry_point("extract_pdf_content")

        # Define edges (transitions)
        workflow.add_edge("extract_pdf_content", "create_chunks")
        workflow.add_edge("create_chunks", "index_chunks")
        workflow.add_edge("index_chunks", "retrieve_content")
        workflow.add_edge("retrieve_content", "generate_answer")
        workflow.add_edge("generate_answer", END)

        return workflow.compile()

    def extract_pdf_content_node(self, state: GraphState) -> Dict[str, Any]:
        print("Executing extract_pdf_content_node")
        pdf_path = state["pdf_path"]
        # Assuming PDFProcessor.process_multimodal_content exists and combines all extractions
        pdf_content = self.pdf_processor.process_multimodal_content(pdf_path)
        return {"pdf_content": pdf_content}

    def create_chunks_node(self, state: GraphState) -> Dict[str, Any]:
        print("Executing create_chunks_node")
        pdf_content = state["pdf_content"]
        document_id = state["document_id"]
        file_name = state["file_name"]
        file_type = state["file_type"]
        upload_timestamp = state["upload_timestamp"]

        multimodal_chunks = create_multimodal_chunks(
            pdf_content,
            document_id,
            file_name,
            file_type,
            upload_timestamp
        )
        return {"multimodal_chunks": multimodal_chunks}

    def index_chunks_node(self, state: GraphState) -> Dict[str, Any]:
        print("Executing index_chunks_node")
        multimodal_chunks = state["multimodal_chunks"]
        self.cross_modal_retriever.index_multimodal_chunks(multimodal_chunks)
        return {}

    def retrieve_content_node(self, state: GraphState) -> Dict[str, Any]:
        print("Executing retrieve_content_node")
        query = state["query"]
        retrieved_documents = self.cross_modal_retriever.retrieve_relevant_content(query)
        return {"retrieved_documents": retrieved_documents}

    def generate_answer_node(self, state: GraphState) -> Dict[str, Any]:
        print("Executing generate_answer_node")
        retrieved_documents = state["retrieved_documents"]
        query = state["query"]

        # This is a placeholder for actual answer generation using an LLM
        # In a real scenario, you would feed the query and retrieved_documents to an LLM.
        context = "\n\n".join([doc.page_content for doc in retrieved_documents])
        answer = f"Based on the retrieved documents, for the query '{query}', the relevant information is: {context[:200]}... (Full answer generation not implemented yet)"
        return {"answer": answer}

    def run_workflow(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the LangGraph workflow with the given initial state.
        """
        # The .stream() method allows for incremental processing, but .invoke() is simpler for full run.
        final_state = self.workflow.invoke(initial_state)
        return final_state

if __name__ == '__main__':
    # This is a simplified example and requires actual implementations of
    # PDFProcessor, ImageExtractor, TableExtractor, TextChunker, VectorStore
    # and a configured OpenAI API key for vision processing.

    # Dummy VectorStore for demonstration
    class DummyVectorStore(VectorStore):
        def __init__(self):
            self.documents = []

        def add_documents(self, docs: List[Document]):
            self.documents.extend(docs)
            print(f"[DummyVectorStore] Added {len(docs)} documents.")

        def similarity_search(self, query: str, k: int = 5, filter: Optional[Dict[str, Any]] = None) -> List[Document]:
            print(f"[DummyVectorStore] Searching for '{query}' with filter {filter}")
            # For demo, return some dummy documents that might match.
            results = []
            for doc in self.documents:
                if query.lower() in doc.page_content.lower():
                    results.append(doc)
            return results[:k]

    # Initialize with a dummy vector store
    dummy_vector_store_instance = DummyVectorStore()
    workflow_manager = WorkflowGraph(dummy_vector_store_instance)

    # Example initial state for running the workflow
    # In a real scenario, pdf_path would point to an actual PDF file.
    # document_id, file_name, file_type, upload_timestamp would come from an upload process.
    initial_state = GraphState(
        pdf_path="dummy_path.pdf", # Replace with a real PDF path for actual testing
        document_id="doc_456",
        file_name="example_doc.pdf",
        file_type="application/pdf",
        upload_timestamp="2025-10-16T12:00:00Z",
        query="What is multimodal RAG?",
        pdf_content={}, # Will be populated by extract_pdf_content_node
        multimodal_chunks=[], # Will be populated by create_chunks_node
        retrieved_documents=[], # Will be populated by retrieve_content_node
        answer="",
        error=""
    )

    print("\nRunning the LangGraph workflow...")
    # try:
    #     final_state = workflow_manager.run_workflow(initial_state)
    #     print("\nWorkflow finished.")
    #     print(f"Final Answer: {final_state.get('answer')}")
    # except Exception as e:
    #     print(f"Workflow encountered an error: {e}")

    print("WorkflowGraph class created. Requires full integration with other modules for live testing.")
