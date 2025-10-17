"""
RAG (Retrieval-Augmented Generation) Pipeline
Following the reference architecture for multimodal RAG with OpenAI
"""

import logging
from typing import List, Dict, Any, Optional
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import Document, HumanMessage, SystemMessage
from src.config import settings
from src.vector_store import get_vector_store
from src.pdf_processor import extract_pdf_content

logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline for question answering
    Following the multimodal RAG reference approach
    """
    
    def __init__(self):
        logger.debug("Initializing RAGPipeline...")
        # Initialize OpenAI chat model
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0,
            openai_api_key=settings.openai_api_key
        )
        logger.debug(f"Initialized ChatOpenAI with model: {settings.openai_model}")
        
        # Get vector store instance
        self.vector_store = get_vector_store()
        logger.debug(f"Initialized VectorStore of type: {settings.vector_store_type}")
        
        # Initialize prompts
        self._initialize_prompts()
        
        # Initialize QA chain
        self.qa_chain = None
        self._initialize_qa_chain()
        logger.info("RAGPipeline initialized successfully.")
    
    def _initialize_prompts(self):
        """
        Initialize RAG prompts following best practices
        """
        logger.debug("Initializing RAG prompts...")
        # System prompt for the assistant
        self.system_prompt = """
        You are a helpful AI assistant that answers questions based on the provided document context.
        Your task is to accurately answer questions using only the information from the retrieved document chunks.
        
        Guidelines:
        1. Use only the information provided in the context
        2. If the context doesn't contain the answer, say "I don't have enough information to answer this question"
        3. Provide specific details and citations when possible
        4. If there are images or tables mentioned in the context, acknowledge them
        5. Be accurate and comprehensive in your answers
        6. Format your response clearly with bullet points when appropriate
        """
        
        # QA prompt template
        self.qa_prompt_template = """
        Context: {context}

        Question: {question}

        Based on the provided context, please answer the question. If the context doesn't contain enough information to answer the question, please say so.
        
        Answer:
        """
        
        self.qa_prompt = PromptTemplate(
            template=self.qa_prompt_template,
            input_variables=["context", "question"]
        )
        
        # Chat prompt for more sophisticated interactions
        self.chat_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", """
            Context Information:
            {context}
            
            Question: {question}
            
            Please provide a comprehensive answer based on the context above.
            """)
        ])
        logger.debug("RAG prompts initialized.")
    
    def _initialize_qa_chain(self):
        """
        Initialize the RetrievalQA chain
        """
        logger.debug("Initializing RetrievalQA chain...")
        try:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.vector_store.as_retriever(
                    search_kwargs={"k": settings.max_retrieved_docs}
                ),
                chain_type_kwargs={"prompt": self.qa_prompt},
                return_source_documents=True
            )
            logger.info("Initialized RAG QA chain")
            
        except Exception as e:
            logger.exception(f"Error initializing QA chain.")
            raise
    
    def query(self, question: str, filter_dict: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Query the RAG pipeline
        
        Args:
            question: User's question
            filter_dict: Optional filter for retrieval
            
        Returns:
            Dictionary containing answer and source documents
        """
        logger.debug(f"Received query: {question[:100]}... with filter: {filter_dict}")
        try:
            if not question.strip():
                logger.warning("Empty question received for query.")
                return {
                    "answer": "Please provide a valid question.",
                    "source_documents": [],
                    "question": question
                }
            
            # Retrieve relevant documents
            if filter_dict:
                source_docs = self.vector_store.similarity_search(
                    question, 
                    k=settings.max_retrieved_docs, 
                    filter_dict=filter_dict
                )
                logger.debug(f"Retrieved documents with filter: {filter_dict}")
            else:
                source_docs = self.vector_store.similarity_search(
                    question, 
                    k=settings.max_retrieved_docs
                )
                logger.debug("Retrieved documents without filter.")
            
            if not source_docs:
                logger.info(f"No relevant documents found for question: {question[:50]}...")
                return {
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "source_documents": [],
                    "question": question
                }
            
            # Generate answer using the LLM
            context = self._format_context(source_docs)
            answer = self._generate_answer(question, context)
            
            # Prepare source information
            sources = self._format_sources(source_docs)
            
            result = {
                "answer": answer,
                "source_documents": sources,
                "question": question,
                "context_used": context[:500] + "..." if len(context) > 500 else context
            }
            
            logger.info(f"Generated answer for question: {question[:50]}...")
            logger.debug(f"Query result for {question[:50]}...: {result['answer'][:100]}...")
            return result
            
        except Exception as e:
            logger.exception(f"Error processing query '{question}'.")
            return {
                "answer": f"An error occurred while processing your question: {str(e)}",
                "source_documents": [],
                "question": question
            }
    
    def _format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into context string
        """
        logger.debug(f"Formatting context for {len(documents)} documents.")
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            metadata = doc.metadata
            
            # Create context header
            source_info = []
            if metadata.get('document_title'):
                source_info.append(f"Document: {metadata['document_title']}")
            if metadata.get('page_number'):
                source_info.append(f"Page {metadata['page_number']}")
            if metadata.get('content_type'):
                source_info.append(f"Type: {metadata['content_type']}")
            
            source_str = " | ".join(source_info)
            
            # Add content
            content = doc.page_content
            
            context_parts.append(f"""
            [Source {i}] {source_str}
            {content}
            """)
        
        logger.debug("Context formatted.")
        return "\n".join(context_parts)
    
    def _generate_answer(self, question: str, context: str) -> str:
        """
        Generate answer using the LLM
        """
        logger.debug(f"Generating answer for question: {question[:50]}...")
        try:
            # Use chat prompt for better formatting
            messages = self.chat_prompt.format_messages(
                context=context,
                question=question
            )
            
            response = self.llm(messages)
            logger.info(f"Answer generated for question: {question[:50]}...")
            return response.content
            
        except Exception as e:
            logger.exception(f"Error generating answer for question '{question}'. Attempting fallback.")
            # Fallback to simpler prompt
            try:
                formatted_prompt = self.qa_prompt.format(
                    context=context,
                    question=question
                )
                response = self.llm(formatted_prompt)
                logger.info(f"Fallback answer generated for question: {question[:50]}...")
                return response
            except Exception as fallback_error:
                logger.exception(f"Fallback answer generation also failed for question '{question}'.")
                return "I apologize, but I encountered an error while generating an answer to your question."
    
    def _format_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Format source documents for response
        """
        logger.debug(f"Formatting sources for {len(documents)} documents.")
        sources = []
        
        for doc in documents:
            metadata = doc.metadata
            
            source = {
                "content": doc.page_content,
                "metadata": {
                    "page_number": metadata.get('page_number'),
                    "document_title": metadata.get('document_title', 'Unknown'),
                    "content_type": metadata.get('content_type', 'text'),
                    "document_id": metadata.get('document_id'),
                    "chunk_size": len(doc.page_content)
                }
            }
            
            # Add additional metadata if available
            if metadata.get('has_images'):
                source["metadata"]["has_images"] = True
            if metadata.get('has_tables'):
                source["metadata"]["has_tables"] = True
            if metadata.get('related_images'):
                source["metadata"]["related_images"] = metadata['related_images']
            if metadata.get('related_tables'):
                source["metadata"]["related_tables"] = metadata['related_tables']
            
            sources.append(source)
        
        logger.debug("Sources formatted.")
        return sources
    
    def query_with_sources(self, question: str, k: int = None) -> Dict[str, Any]:
        """
        Query with detailed source information including scores
        """
        logger.debug(f"Received query with sources: {question[:100]}... with k={k}")
        try:
            k = k or settings.max_retrieved_docs
            
            # Get documents with scores
            docs_with_scores = self.vector_store.similarity_search_with_score(
                question, 
                k=k
            )
            logger.debug(f"Retrieved {len(docs_with_scores)} documents with scores.")
            
            if not docs_with_scores:
                logger.info(f"No relevant documents found with scores for question: {question[:50]}...")
                return {
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "source_documents": [],
                    "question": question
                }
            
            # Separate documents and scores
            documents = [doc for doc, score in docs_with_scores]
            scores = [score for doc, score in docs_with_scores]
            
            # Generate answer
            context = self._format_context(documents)
            answer = self._generate_answer(question, context)
            
            # Format sources with scores
            sources = []
            for i, (doc, score) in enumerate(docs_with_scores):
                source = self._format_sources([doc])[0]
                source["relevance_score"] = float(score)
                source["rank"] = i + 1
                sources.append(source)
            
            result = {
                "answer": answer,
                "source_documents": sources,
                "question": question,
                "retrieval_stats": {
                    "total_retrieved": len(documents),
                    "average_score": sum(scores) / len(scores) if scores else 0,
                    "best_score": min(scores) if scores else 0
                }
            }
            
            logger.info(f"Generated answer with sources for question: {question[:50]}...")
            logger.debug(f"Query with sources result for {question[:50]}...: {result['answer'][:100]}...")
            return result
            
        except Exception as e:
            logger.exception(f"Error in query_with_sources for question '{question}'. Falling back to simple query.")
            return self.query(question)  # Fallback to simple query
    
    def add_document(self, pdf_path: str, document_id: str = None) -> Dict[str, Any]:
        """
        Add a PDF document to the RAG pipeline
        
        Args:
            pdf_path: Path to PDF file
            document_id: Optional document identifier
            
        Returns:
            Status and document information
        """
        logger.debug(f"Attempting to add document: {pdf_path} with ID: {document_id}")
        try:
            # Extract content from PDF
            pdf_content = extract_pdf_content(pdf_path)
            logger.debug(f"Extracted content from PDF: {pdf_path}")
            
            # Add to vector store
            doc_ids = self.vector_store.add_pdf_content(pdf_content, document_id)
            logger.debug(f"Added PDF content to vector store. Generated chunk IDs: {doc_ids}")
            
            result = {
                "success": True,
                "document_id": document_id or doc_ids[0] if doc_ids else None,
                "chunks_created": len(doc_ids),
                "pages_processed": pdf_content.get('total_pages', 0),
                "images_extracted": pdf_content.get('total_images', 0),
                "tables_extracted": pdf_content.get('total_tables', 0),
                "document_metadata": pdf_content.get('document_metadata', {})
            }
            
            logger.info(f"Successfully added document {pdf_path} to RAG pipeline. Document ID: {result['document_id']}")
            return result
            
        except Exception as e:
            logger.exception(f"Error adding document {pdf_path}.")
            return {
                "success": False,
                "error": str(e),
                "document_id": document_id
            }
    
    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """
        Delete a document from the RAG pipeline
        """
        logger.debug(f"Attempting to delete document with ID: {document_id}")
        try:
            success = self.vector_store.delete_document(document_id)
            
            result = {
                "success": success,
                "document_id": document_id
            }
            
            if success:
                logger.info(f"Successfully deleted document {document_id} from RAG pipeline")
            else:
                logger.warning(f"Failed to delete document {document_id} from RAG pipeline. Document might not exist.")
            
            return result
            
        except Exception as e:
            logger.exception(f"Error deleting document {document_id}.")
            return {
                "success": False,
                "error": str(e),
                "document_id": document_id
            }
    
    def get_document_list(self) -> List[Dict[str, Any]]:
        """
        Get list of all documents in the RAG pipeline
        """
        logger.debug("Attempting to retrieve document list.")
        try:
            # Get collection statistics
            stats = self.vector_store.get_collection_stats()
            logger.debug("Retrieved vector store collection stats.")
            
            # This is a simplified implementation
            # In a real implementation, you might want to maintain a separate document registry
            result = {
                "total_documents": stats.get('unique_documents', 0),
                "total_chunks": stats.get('total_chunks', 0),
                "content_types": stats.get('content_types', {}),
                "average_chunks_per_document": stats.get('average_chunks_per_document', 0)
            }
            
            logger.info("Successfully retrieved document list.")
            return [result]  # Return as list for consistency
            
        except Exception as e:
            logger.exception(f"Error getting document list.")
            return []

# Global RAG pipeline instance
_rag_pipeline = None

def get_rag_pipeline() -> RAGPipeline:
    """
    Get or create the global RAG pipeline instance
    """
    global _rag_pipeline
    if _rag_pipeline is None:
        logger.debug("Creating new RAGPipeline instance.")
        _rag_pipeline = RAGPipeline()
    logger.debug("Returning RAGPipeline instance.")
    return _rag_pipeline

# Convenience functions
def ask_question(question: str, filter_dict: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Convenience function to ask a question
    """
    logger.debug(f"Convenience function ask_question called for: {question[:50]}...")
    pipeline = get_rag_pipeline()
    return pipeline.query(question, filter_dict)

def add_pdf_document(pdf_path: str, document_id: str = None) -> Dict[str, Any]:
    """
    Convenience function to add a PDF document
    """
    logger.debug(f"Convenience function add_pdf_document called for: {pdf_path}")
    pipeline = get_rag_pipeline()
    return pipeline.add_document(pdf_path, document_id)