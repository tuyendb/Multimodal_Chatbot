import os
import base64
import uuid
import json
import logging
from typing import List, Dict, Tuple
from collections import defaultdict

import pymupdf4llm
import numpy as np
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from langchain_upstage import UpstageDocumentParseLoader
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_experimental.open_clip import OpenCLIPEmbeddings
import tempfile
import atexit
import shutil
from PIL import Image
from io import BytesIO
import chromadb

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImageDescriptionGenerator:
    """Generate rich descriptions for images with retry logic"""
    
    def __init__(self):
        self.model = ChatOpenAI(
            api_key=os.getenv("OPEN_ROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model="openai/gpt-4o-mini",
            temperature=0
        )
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def generate_description(self, img_base64: str) -> Dict:
        """Generate description and classify image type"""
        try:
            message = HumanMessage(
                content=[
                    {"type": "text", 
                     "text": """Analyze this image and provide:

1. **Image Type** (one of): chart, table, diagram, photo, mixed, decorative
2. **Description**: 
   - If decorative/non-informational: output '<---image--->'
   - For charts: State chart type, all axis labels, data points, trends
   - For tables: Convert to markdown table with exact data
   - For diagrams: Describe components and relationships
   - For photos: List visible objects and context
   
3. **Has Text**: true/false (does image contain readable text?)
4. **Key Information**: Extract any critical numbers, labels, or facts

Return JSON format:
{
    "type": "chart|table|diagram|photo|mixed|decorative",
    "description": "detailed description",
    "has_text": true/false,
    "key_info": ["fact1", "fact2"]
}

Rules:
* Be factual and precise
* Include all numerical values exactly
* No interpretation or analysis"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                    },
                ]
            )
            
            response = self.model.invoke([message])
            
            # Parse JSON response
            try:
                result = json.loads(response.content)
            except json.JSONDecodeError:
                # Fallback if not JSON
                result = {
                    "type": "mixed",
                    "description": response.content,
                    "has_text": False,
                    "key_info": []
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate description: {e}")
            return {
                "type": "unknown",
                "description": "<---image----->",
                "has_text": False,
                "key_info": []
            }


class HybridImageEmbedder:
    """Create hybrid embeddings combining visual and semantic information"""
    
    def __init__(self, visual_weight=0.7, semantic_weight=0.3):
        self.clip = OpenCLIPEmbeddings(
            model_name="ViT-B-32",
            checkpoint="openai"
        )
        self.text_embedder = OpenAIEmbeddings()
        self.visual_weight = visual_weight
        self.semantic_weight = semantic_weight
    
    def embed_images(self, image_paths: List[str], descriptions: List[str]) -> Tuple[List, List, List]:
        """
        Returns: (hybrid_embeddings, visual_embeddings, semantic_embeddings)
        """
        logger.info(f"Creating hybrid embeddings for {len(image_paths)} images...")
        
        # Visual embeddings from CLIP
        visual_embs = self.clip.embed_image(image_paths)
        
        # Semantic embeddings from descriptions
        semantic_embs = self.text_embedder.embed_documents(descriptions)
        
        # Normalize embeddings
        visual_embs = [np.array(emb) / (np.linalg.norm(emb) + 1e-8) for emb in visual_embs]
        semantic_embs = [np.array(emb) / (np.linalg.norm(emb) + 1e-8) for emb in semantic_embs]
        
        # Create hybrid embeddings by concatenating (instead of adding)
        # This preserves all information from both modalities
        hybrid_embs = []
        for v_emb, s_emb in zip(visual_embs, semantic_embs):
            # Concatenate visual (512) + semantic (1536) = 2048 dimensions
            combined = np.concatenate([v_emb, s_emb])
            hybrid_embs.append(combined.tolist())
        
        # Convert to lists for storage
        visual_embs = [emb.tolist() for emb in visual_embs]
        semantic_embs = [emb.tolist() for emb in semantic_embs]
        
        return hybrid_embs, visual_embs, semantic_embs


def extract_section_context(text: str, max_chars=200) -> str:
    """Extract section title or surrounding context"""
    lines = text.split('\n')
    
    # Look for headings (lines with #, or all caps, or short lines)
    for line in lines[:5]:  # Check first 5 lines
        line = line.strip()
        if line.startswith('#') or (line.isupper() and len(line) < 100):
            return line
    
    # Return first meaningful text
    for line in lines:
        line = line.strip()
        if len(line) > 20:
            return line[:max_chars]
    
    return ""


def create_enriched_image_docs(docs: List, desc_generator: ImageDescriptionGenerator) -> Tuple[List, List, List]:
    """
    Create enriched image documents with rich metadata
    Returns: (image_documents, image_paths, image_descriptions)
    """
    logger.info("Creating enriched image documents...")
    
    image_documents = []
    image_paths = []
    image_descriptions = []
    temp_dir = tempfile.mkdtemp()
    atexit.register(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
    
    for doc in docs:
        if 'base64_encodings' in doc.metadata and len(doc.metadata['base64_encodings']) > 0:
            # Extract section context from surrounding text
            section_context = extract_section_context(doc.page_content)
            
            for idx, img_base64 in enumerate(doc.metadata['base64_encodings']):
                # Generate rich description
                desc_data = desc_generator.generate_description(img_base64)
                
                # Save image to temp file
                img_data = base64.b64decode(img_base64)
                image = Image.open(BytesIO(img_data))
                img_path = os.path.join(temp_dir, f"img_{uuid.uuid4().hex}.jpg")
                image.save(img_path)
                
                image_paths.append(img_path)
                image_descriptions.append(desc_data['description'])
                
                # Create document with rich metadata
                new_doc = Document(
                    page_content=desc_data['description'],
                    metadata={
                        "page": f"{doc.metadata.get('page', 'unknown')}",
                        "type": "image",
                        "image_type": desc_data['type'],
                        "has_text": desc_data['has_text'],
                        "key_info": json.dumps(desc_data['key_info']),
                        "section_context": section_context,
                        "surrounding_text": doc.page_content[:300],  # First 300 chars
                        "image_base64": img_base64,
                        "image_path": img_path,
                        "doc_id": str(uuid.uuid4())
                    }
                )
                
                image_documents.append(new_doc)
                
                logger.info(f"  [OK] Image {len(image_documents)}: {desc_data['type']} from page {doc.metadata.get('page')}")
    
    return image_documents, image_paths, image_descriptions


def smart_chunk_with_context(md_text: List, image_docs: List, 
                            chunk_size=1000, chunk_overlap=200) -> List[Document]:
    """
    Smart chunking that preserves image-text relationships
    """
    logger.info("Smart chunking with context preservation...")
    
    # Organize by page
    page_contents = defaultdict(lambda: {"text": [], "images": []})
    
    # Collect text by page
    for text_item in md_text:
        page = int(text_item['metadata']['page'])
        page_contents[page]["text"].append(text_item['text'])
    
    # Collect images by page
    for img_doc in image_docs:
        page = int(img_doc.metadata['page'])
        page_contents[page]["images"].append(img_doc)
    
    # Create chunks with image context
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    for page in sorted(page_contents.keys()):
        page_data = page_contents[page]
        
        # Combine text for this page
        page_text = '\n\n'.join(page_data["text"])
        
        # Chunk the text
        text_chunks = text_splitter.split_text(page_text)
        
        # Create documents with image references
        image_ids = [img.metadata['doc_id'] for img in page_data["images"]]
        
        for i, chunk in enumerate(text_chunks):
            # Add image context if images exist on this page
            if page_data["images"]:
                image_context = "\n\n[Images on this page]:\n"
                for img_doc in page_data["images"]:
                    image_context += f"- {img_doc.metadata['image_type']}: {img_doc.page_content[:100]}...\n"
                
                chunk_with_context = chunk + image_context
            else:
                chunk_with_context = chunk
            
            doc = Document(
                page_content=chunk_with_context,
                metadata={
                    "page": page,
                    "chunk_index": i,
                    "type": "text",
                    "related_images": json.dumps(image_ids),  # Link to images on same page
                    "doc_id": str(uuid.uuid4()),
                    "source": md_text[0]['metadata'].get('file_path', 'unknown')
                }
            )
            all_chunks.append(doc)
    
    logger.info(f"[OK] Created {len(all_chunks)} smart chunks")
    return all_chunks


def validate_embeddings(embeddings: List) -> bool:
    """Validate embedding quality"""
    for i, emb in enumerate(embeddings):
        emb_array = np.array(emb)
        if np.isnan(emb_array).any() or np.isinf(emb_array).any():
            logger.error(f"Invalid embedding at index {i}: contains NaN or Inf")
            return False
        if np.linalg.norm(emb_array) < 0.01:
            logger.warning(f"Near-zero embedding at index {i}")
    return True


def main(file_path: str, persist_dir="./exp3/chroma_db"):
    load_dotenv(override=True)

    logger.info("="*60)
    logger.info("IMPROVED MULTIMODAL RAG INGESTION PIPELINE")
    logger.info("="*60)

    # Initialize components
    desc_generator = ImageDescriptionGenerator()
    hybrid_embedder = HybridImageEmbedder(visual_weight=0.7, semantic_weight=0.3)

    # Step 1: Extract text
    logger.info("\nSTEP 1: Extracting text from PDF...")
    md_text = pymupdf4llm.to_markdown(
        doc=file_path,
        page_chunks=True,
        show_progress=True
    )
    logger.info(f"[OK] Extracted {len(md_text)} pages")

    # Step 2: Load documents with images
    logger.info("\nSTEP 2: Loading images from PDF...")
    loader = UpstageDocumentParseLoader(
        file_path,
        split="page",
        output_format="markdown",
        base64_encoding=["figure", "chart", "table"]
    )
    docs = loader.load_and_split()
    logger.info(f"[OK] Loaded {len(docs)} pages")

    # Step 3: Create enriched image documents
    logger.info("\nSTEP 3: Creating enriched image documents...")
    image_docs, image_paths, image_descriptions = create_enriched_image_docs(
        docs, desc_generator
    )
    logger.info(f"[OK] Created {len(image_docs)} enriched image documents")

    # Step 4: Smart chunking
    logger.info("\nSTEP 4: Smart chunking with context preservation...")
    text_chunks = smart_chunk_with_context(md_text, image_docs)
    logger.info(f"[OK] Created {len(text_chunks)} context-aware chunks")

    # Step 5: Create text embeddings
    logger.info("\nSTEP 5: Creating text embeddings...")
    text_embedder = OpenAIEmbeddings()
    
    text_vectorstore = Chroma.from_documents(
        documents=text_chunks,
        embedding=text_embedder,
        collection_name="text_chunks",
        persist_directory=persist_dir
    )
    logger.info("[OK] Text chunks indexed")

    # Step 6: Create hybrid image embeddings
    logger.info("\nSTEP 6: Creating hybrid image embeddings...")
    hybrid_embs, visual_embs, semantic_embs = hybrid_embedder.embed_images(
        image_paths, image_descriptions
    )
    
    # Validate embeddings
    if not validate_embeddings(hybrid_embs):
        raise ValueError("Invalid embeddings detected!")
    
    logger.info("[OK] Hybrid embeddings created and validated")

    # Step 7: Index images with rich metadata
    logger.info("\nSTEP 7: Indexing images with rich metadata...")
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    
    # Clean old collection
    try:
        chroma_client.delete_collection(name="image_vectors")
        logger.info("  [OK] Deleted old image collection")
    except:
        pass
    
    image_collection = chroma_client.create_collection(
        name="image_vectors",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Add images with all embeddings and metadata
    for i, img_doc in enumerate(image_docs):
        image_collection.add(
            ids=[img_doc.metadata['doc_id']],
            embeddings=[hybrid_embs[i]],
            metadatas=[{
                "page": img_doc.metadata['page'],
                "type": img_doc.metadata['type'],
                "image_type": img_doc.metadata['image_type'],
                "has_text": img_doc.metadata['has_text'],
                "section_context": img_doc.metadata['section_context'],
                "image_base64": img_doc.metadata['image_base64'],
                "description": img_doc.page_content,
                "visual_embedding": json.dumps(visual_embs[i]),  # Store for potential re-ranking
                "semantic_embedding": json.dumps(semantic_embs[i]),
                "key_info": img_doc.metadata.get('key_info', '[]')
            }],
            documents=[img_doc.page_content]
        )
    
    logger.info(f"[OK] Indexed {len(image_docs)} images with hybrid embeddings")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"INGESTION COMPLETED SUCCESSFULLY!")
    logger.info(f"{'='*60}")
    logger.info(f"Text chunks: {len(text_chunks)} (with image context)")
    logger.info(f"Images: {len(image_docs)} (hybrid embeddings)")
    logger.info(f"Database: {persist_dir}")
    logger.info(f"Features:")
    logger.info(f"   - Smart chunking with context preservation")
    logger.info(f"   - Hybrid visual + semantic embeddings")
    logger.info(f"   - Rich metadata (image type, key info, section context)")
    logger.info(f"   - Error handling with retry logic")
    logger.info(f"   - Embedding validation")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    file_path = "Document 173322.pdf"
    main(file_path)