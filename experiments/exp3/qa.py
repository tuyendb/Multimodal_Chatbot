import base64
import os
import json
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
import chromadb


class HybridImageEmbedder:
    """Create hybrid embeddings combining visual and semantic information - same as ingest_data.py"""

    def __init__(self, visual_weight=0.7, semantic_weight=0.3):
        self.clip = OpenCLIPEmbeddings(
            model_name="ViT-B-32",
            checkpoint="openai"
        )
        self.text_embedder = OpenAIEmbeddings()
        self.visual_weight = visual_weight
        self.semantic_weight = semantic_weight

    def embed_query(self, text: str) -> List[float]:
        """Create hybrid embedding for text query"""
        # Visual embedding from CLIP
        visual_emb = np.array(self.clip.embed_query(text))

        # Semantic embedding from OpenAI
        semantic_emb = np.array(self.text_embedder.embed_query(text))

        # Normalize embeddings
        visual_emb = visual_emb / (np.linalg.norm(visual_emb) + 1e-8)
        semantic_emb = semantic_emb / (np.linalg.norm(semantic_emb) + 1e-8)

        # Create hybrid embedding (concatenate like in ingest_data.py)
        combined = np.concatenate([visual_emb, semantic_emb])
        return combined.tolist()

    def embed_image(self, image_paths: List[str]) -> List[List[float]]:
        """Create hybrid embedding for image query"""
        # Visual embedding from CLIP
        visual_embs = self.clip.embed_image(image_paths)

        # For images, we need text descriptions for semantic embedding
        # Since we don't have descriptions for query images, use visual only
        # But pad to 2048 dimensions to match stored embeddings
        hybrid_embs = []
        for visual_emb in visual_embs:
            visual_emb = np.array(visual_emb) / (np.linalg.norm(visual_emb) + 1e-8)
            # Pad with zeros to match 2048 dimensions (512 visual + 1536 semantic)
            padding = np.zeros(1536)  # Semantic part
            combined = np.concatenate([visual_emb, padding])
            hybrid_embs.append(combined.tolist())

        return hybrid_embs

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class QueryIntent:
    """Enhanced query intent classification"""
    needs_images: bool
    needs_text: bool
    visual_emphasis: float  # 0-1: 0=text only, 1=image only
    query_type: str  # factual, visual_identification, data_extraction, visual_similar, cross_reference, visual_analysis, visual_first
    cross_modal: bool  # Needs both text and images together
    analysis_depth: str  # "specific" or "contextual"
    data_extraction: bool  # Extracting numbers/data from visuals
    priority_visual: bool  # Visual info is primary focus
    search_strategy: str  # "text_first", "image_first", "balanced"


class QueryClassifier:
    """Classify query intent to optimize retrieval"""
    
    def __init__(self):
        self.model = ChatOpenAI(
            api_key=os.getenv("OPEN_ROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model="openai/gpt-4o-mini",
            temperature=0
        )
    
    def classify(self, question: str, has_user_image: bool = False) -> QueryIntent:
        """Enhanced query classification for optimal retrieval"""
        logger.info("🧠 Enhanced query classification...")

        prompt = f"""Classify this query for optimal multimodal retrieval:

Question: {question}
User provided image: {has_user_image}

Analyze and classify:

1. **Query Type** (choose one):
   - factual: Text-based factual questions, no visuals needed
   - visual_identification: "Show me/what is X" without user image, needs both text+images
   - data_extraction: Extract specific numbers/data from charts/tables
   - visual_similar: User provided image + "find similar", "what is this", "identify"
   - cross_reference: User image + comparison/analysis questions
   - visual_analysis: Complex analysis requiring deep visual+text understanding
   - visual_first: User image identification questions - SEARCH IMAGES FIRST!

2. **Search Strategy**:
   - text_first: Search text, then get linked images (default for most)
   - image_first: Search images by similarity, then get text context (for visual_similar, visual_first)
   - balanced: Equal emphasis on both modalities

3. **Content Needs**:
   - needs_images: true/false
   - needs_text: true/false
   - cross_modal: true if both text and images needed together
   - data_extraction: true if extracting specific numbers/data

4. **Retrieval Priority**:
   - visual_emphasis: 0.0-1.0 (how much weight on visuals)
   - analysis_depth: "specific" (exact data) or "contextual" (broader understanding)
   - priority_visual: true if visual info is primary focus

Return JSON:
{{
    "needs_images": true/false,
    "needs_text": true/false,
    "visual_emphasis": 0.0-1.0,
    "query_type": "factual|visual_identification|data_extraction|visual_similar|cross_reference|visual_analysis|visual_first",
    "search_strategy": "text_first|image_first|balanced",
    "cross_modal": true/false,
    "analysis_depth": "specific|contextual",
    "data_extraction": true/false,
    "priority_visual": true/false,
    "reasoning": "brief explanation of classification"
}}

Examples:
- "What is the company's revenue strategy?" → {{"query_type": "factual", "needs_images": false, "needs_text": true, "search_strategy": "text_first"}}
- "Show me the revenue chart" → {{"query_type": "visual_identification", "needs_images": true, "needs_text": true, "search_strategy": "text_first", "priority_visual": true}}
- "What was Q2 revenue exactly?" → {{"query_type": "data_extraction", "needs_images": true, "needs_text": true, "search_strategy": "text_first", "data_extraction": true}}
- [User image] + "What is this image?" → {{"query_type": "visual_first", "needs_images": true, "needs_text": true, "search_strategy": "image_first", "priority_visual": true}}
- [User image] + "Find similar charts" → {{"query_type": "visual_similar", "needs_images": true, "needs_text": false, "search_strategy": "image_first"}}
- [User image] + "How does this compare to our performance?" → {{"query_type": "cross_reference", "needs_images": true, "needs_text": true, "search_strategy": "balanced"}}
- "Analyze the company's market position across all metrics" → {{"query_type": "visual_analysis", "needs_images": true, "needs_text": true, "search_strategy": "balanced", "analysis_depth": "contextual"}}
"""
        
        response = self.model.invoke(prompt)
        logger.info(f"🔍 Model response: {response.content}")

        # Clean response content to handle common issues
        content = response.content.strip()
        if content.startswith('```json'):
            content = content[7:]  # Remove ```json
        if content.endswith('```'):
            content = content[:-3]  # Remove ```
        content = content.strip()

        if not content:
            logger.error("❌ Empty response from model")
            raise ValueError("Empty response from model")

        print(f"🔍 Raw response: '{response.content}'")
        print(f"🔍 Cleaned content: '{content}'")
        result = json.loads(content)

        logger.info(f"  ✓ Intent: {result['query_type']}")
        logger.info(f"  🎯 Visual Emphasis: {result['visual_emphasis']:.2f}")
        logger.info(f"  📋 Needs: Text={result['needs_text']}, Images={result['needs_images']}")
        logger.info(f"  🔗 Cross-modal: {result.get('cross_modal', False)}")
        logger.info(f"  📊 Data Extraction: {result.get('data_extraction', False)}")
        logger.info(f"  📝 Analysis Depth: {result.get('analysis_depth', 'N/A')}")
        logger.info(f"  🎨 Priority Visual: {result.get('priority_visual', False)}")
        logger.info(f"  💭 {result.get('reasoning', 'N/A')}")

        return QueryIntent(
            needs_images=result['needs_images'],
            needs_text=result['needs_text'],
            visual_emphasis=result['visual_emphasis'],
            query_type=result['query_type'],
            cross_modal=result.get('cross_modal', False),
            analysis_depth=result.get('analysis_depth', 'contextual'),
            data_extraction=result.get('data_extraction', False),
            priority_visual=result.get('priority_visual', False),
            search_strategy=result.get('search_strategy', 'text_first')
        )


class AdaptiveThreshold:
    """Calculate adaptive threshold based on distance distribution"""
    
    @staticmethod
    def calculate(distances: List[float], base_threshold=0.35) -> float:
        """
        Calculate adaptive threshold using gap detection
        """
        if not distances or len(distances) < 2:
            return base_threshold
        
        distances = sorted(distances)
        
        # Find significant gap in distances
        for i in range(len(distances) - 1):
            gap = distances[i+1] - distances[i]
            if gap > 0.1:  # Significant gap threshold
                threshold = (distances[i] + distances[i+1]) / 2
                logger.info(f"  📊 Adaptive threshold: {threshold:.3f} (gap detected)")
                return threshold
        
        # Fallback: mean + 0.5*std
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        threshold = min(mean_dist + 0.5 * std_dist, base_threshold)
        
        logger.info(f"  📊 Adaptive threshold: {threshold:.3f} (statistical)")
        return threshold


class MultimodalReranker:
    """Re-rank retrieved documents for better relevance"""
    
    def __init__(self):
        self.model = ChatOpenAI(
            api_key=os.getenv("OPEN_ROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model="openai/gpt-4o-mini",
            temperature=0
        )
    
    def rerank_images(self, question: str, image_candidates: List[Dict], 
                      top_k: int = 3) -> List[Dict]:
        """Re-rank images using LLM-based relevance scoring"""
        if not image_candidates:
            return []
        
        logger.info(f"🔄 Re-ranking {len(image_candidates)} images...")
        
        scored_images = []
        
        for img in image_candidates:
            # Quick relevance check using description only (faster)
            prompt = f"""Rate the relevance (0-10) of this image to the question.
Consider:
- Content relevance
- Information value
- Direct answer capability

Question: {question}

Image Information:
- Type: {img.get('image_type', 'unknown')}
- Description: {img.get('description', 'No description')}
- Key Info: {img.get('key_info', [])}
- Section Context: {img.get('section_context', 'N/A')}

Output ONLY a number 0-10 (no explanation):"""
            
            response = self.model.invoke(prompt)
            score = float(response.content.strip())
            score = max(0, min(10, score))  # Clamp to 0-10
            
            scored_images.append((img, score))
            logger.info(f"  📊 Page {img.get('page')}: {img.get('image_type')} - Score: {score:.1f}/10")
        
        # Sort by score (descending) and return top_k
        scored_images.sort(key=lambda x: x[1], reverse=True)
        top_images = [img for img, score in scored_images[:top_k]]
        
        logger.info(f"  ✓ Selected top {len(top_images)} images after re-ranking")
        return top_images
    
    def rerank_text(self, question: str, text_candidates: List[Document], 
                    top_k: int = 5) -> List[Document]:
        """Re-rank text chunks (simplified version)"""
        if not text_candidates:
            return []
        
        logger.info(f"🔄 Re-ranking {len(text_candidates)} text chunks...")
        
        # Simple keyword-based scoring (can be improved with CrossEncoder)
        question_words = set(question.lower().split())
        
        scored_docs = []
        for doc in text_candidates:
            content_words = set(doc.page_content.lower().split())
            overlap = len(question_words & content_words)
            score = overlap / (len(question_words) + 1e-8)
            scored_docs.append((doc, score))
        
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, score in scored_docs[:top_k]]
        
        logger.info(f"  ✓ Selected top {len(top_docs)} text chunks")
        return top_docs


class ContextOptimizer:
    """Optimize context to fit within token budget"""
    
    TEXT_TOKENS_PER_CHAR = 0.25  # Approximate
    IMAGE_TOKENS = 800  # Per image
    
    @staticmethod
    def optimize(text_results: List[Document], image_results: List[Dict],
                 max_tokens: int = 8000) -> Dict:
        """
        Pack context intelligently within token budget
        Priority: Top images first, then text, then additional images
        """
        logger.info(f"🎯 Optimizing context (budget: {max_tokens} tokens)...")
        
        budget = max_tokens
        context = {'text': [], 'images': []}
        
        # Priority 1: Top 2 most relevant images
        priority_images = image_results[:2]
        for img in priority_images:
            if budget > ContextOptimizer.IMAGE_TOKENS:
                context['images'].append(img)
                budget -= ContextOptimizer.IMAGE_TOKENS
                logger.info(f"  ✓ Added priority image from page {img.get('page')} ({ContextOptimizer.IMAGE_TOKENS} tokens)")
        
        # Priority 2: Fill with text chunks
        for doc in text_results:
            chunk_tokens = len(doc.page_content) * ContextOptimizer.TEXT_TOKENS_PER_CHAR
            if budget > chunk_tokens:
                context['text'].append(doc)
                budget -= chunk_tokens
                logger.info(f"  ✓ Added text chunk from page {doc.metadata.get('page')} ({int(chunk_tokens)} tokens)")
            else:
                break
        
        # Priority 3: Add more images if budget allows
        for img in image_results[2:]:
            if budget > ContextOptimizer.IMAGE_TOKENS:
                context['images'].append(img)
                budget -= ContextOptimizer.IMAGE_TOKENS
                logger.info(f"  ✓ Added additional image from page {img.get('page')}")
        
        logger.info(f"  📊 Final context: {len(context['text'])} text chunks, {len(context['images'])} images")
        logger.info(f"  📊 Estimated tokens used: ~{max_tokens - budget}/{max_tokens}")
        
        return context


def encode_image(image_path: str) -> str:
    """Encode image from file to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_adaptive_k(intent: QueryIntent, base_k: int = 10) -> Dict[str, int]:
    """Calculate adaptive k values based on query type - use actual values, not limited by base_k"""

    if intent.query_type == "factual":
        return {"text": 5, "images": 0}
    elif intent.query_type == "visual_identification":
        return {"text": 3, "images": 4}
    elif intent.query_type == "data_extraction":
        return {"text": 5, "images": 5}
    elif intent.query_type == "visual_similar":
        return {"text": 2, "images": 10}
    elif intent.query_type == "cross_reference":
        return {"text": 5, "images": 8}
    elif intent.query_type == "visual_analysis":
        return {"text": 6, "images": 6}
    elif intent.query_type == "visual_first":
        return {"text": 3, "images": 8}
    else:
        return {"text": base_k, "images": base_k}


def retrieve_linked_images(text_chunk: Document, image_collection) -> List[Dict]:
    """Retrieve images specifically linked to a text chunk via related_images metadata"""
    try:
        # Get image IDs from text chunk metadata
        related_ids = json.loads(text_chunk.metadata.get('related_images', '[]'))
        if not related_ids:
            return []

        logger.info(f"  🔗 Text chunk page {text_chunk.metadata.get('page')} has {len(related_ids)} linked images")

        # Retrieve specific images by ID
        results = image_collection.get(
            ids=related_ids,
            include=['metadatas', 'documents']
        )

        linked_images = []
        for i, img_id in enumerate(related_ids):
            if i < len(results['metadatas']):
                meta = results['metadatas'][i]
                linked_images.append({
                    'page': meta.get('page', '?'),
                    'image_type': meta.get('image_type', 'unknown'),
                    'type': meta.get('type', 'image'),  # Use stored type
                    'base64': meta.get('image_base64', ''),
                    'description': meta.get('description', ''),
                    'section_context': meta.get('section_context', ''),
                    'key_info': json.loads(meta.get('key_info', '[]')),
                    'has_text': meta.get('has_text', False),  # Use stored has_text
                    'visual_embedding': json.loads(meta.get('visual_embedding', '[]')),  # Use stored visual embedding
                    'semantic_embedding': json.loads(meta.get('semantic_embedding', '[]')),  # Use stored semantic embedding
                    'distance': 0.0,  # Perfect match since linked
                    'similarity': 100.0,
                    'source': 'linked_text',
                    'doc_id': img_id
                })

        return linked_images

    except Exception as e:
        logger.warning(f"  ⚠️  Failed to retrieve linked images: {e}")
        return []


def retrieve_text(vectorstore, question: str, intent: QueryIntent) -> List[Document]:
    """Enhanced text retrieval with adaptive k based on query type"""
    if not intent.needs_text:
        logger.info("  ↳ Skipping text retrieval (not needed)")
        return []

    # Get adaptive k based on query type
    adaptive_k = get_adaptive_k(intent)
    text_k = adaptive_k["text"]

    logger.info(f"🔍 Retrieving text (k={text_k}) for {intent.query_type} query...")
    results = vectorstore.similarity_search(question, k=text_k)

    logger.info(f"  ✓ Found {len(results)} text chunks")
    for i, doc in enumerate(results[:3], 1):
        preview = doc.page_content[:80].replace('\n', ' ')
        logger.info(f"    {i}. Page {doc.metadata.get('page', '?')}: {preview}...")

    return results


def get_text_from_images(image_results: List[Dict], text_vectorstore, question: str, intent: QueryIntent) -> List[Document]:
    """Get text context from pages that contain relevant images"""
    if not image_results:
        return []

    logger.info(f"🔍 Getting text context from {len(image_results)} images...")

    # Collect unique pages from images
    pages = set()
    for img in image_results:
        if 'page' in img:
            pages.add(img['page'])

    # Get text chunks from these pages
    all_text_results = []
    for page in pages:
        # Search for text chunks from this specific page
        page_filter = f"page {page}"
        results = text_vectorstore.similarity_search(
            f"{question} {page_filter}",
            k=min(3, len(pages))  # Distribute across pages
        )
        all_text_results.extend(results)

    # Limit total results
    adaptive_k = get_adaptive_k(intent)
    max_text = adaptive_k["text"]
    all_text_results = all_text_results[:max_text]

    logger.info(f"  ✓ Found {len(all_text_results)} text chunks from image pages")
    return all_text_results


def retrieve_images(image_collection, question: str, image_path: Optional[str],
                    intent: QueryIntent, clip_embeddings, hybrid_embeddings, text_results: List[Document] = None) -> List[Dict]:
    """Enhanced image retrieval with multiple strategies"""
    if not intent.needs_images:
        logger.info("  ↳ Skipping image retrieval (not needed)")
        return []

    # Get adaptive k
    adaptive_k = get_adaptive_k(intent)
    image_k = adaptive_k["images"]

    logger.info(f"🔍 Retrieving images (k={image_k}) for {intent.query_type} query...")

    # Strategy 1: Use linked images from text chunks (Primary for most queries)
    if text_results and intent.query_type in ["visual_questions", "data_extraction", "analysis"]:
        logger.info(f"  🔗 Strategy 1: Using linked images from text chunks...")
        all_linked_images = []

        for doc in text_results:
            linked_images = retrieve_linked_images(doc, image_collection)
            all_linked_images.extend(linked_images)

        if all_linked_images:
            logger.info(f"  ✅ Found {len(all_linked_images)} linked images from text chunks")
            return all_linked_images[:image_k]

    # Strategy 2: Vector similarity search (Fallback or primary for image-only queries)
    if intent.query_type in ["visual_similar", "cross_reference", "visual_first"] or not text_results:
        logger.info(f"  🔍 Strategy 2: Using vector similarity search...")
        return vector_image_search(image_collection, question, image_path, image_k, intent, hybrid_embeddings)

    # Strategy 3: Page-based retrieval (Additional context for analysis)
    if intent.query_type in ["visual_analysis", "analysis"] and text_results:
        logger.info(f"  📄 Strategy 3: Page-based image retrieval for context...")
        return page_based_image_retrieval(image_collection, text_results, image_k)

    return []


def vector_image_search(image_collection, question: str, image_path: Optional[str],
                       k: int, intent: QueryIntent, hybrid_embeddings) -> List[Dict]:
    """Vector similarity search using hybrid embeddings"""
    logger.info(f"  🔍 Vector search mode: {'image-to-image' if image_path else 'text-to-image'}")

    # Determine query embedding using hybrid embeddings (2048 dimensions)
    if image_path:
        query_embedding = hybrid_embeddings.embed_image([image_path])[0]
        logger.info(f"  📷 Using user image for similarity search (2048-dim hybrid)")
    else:
        query_embedding = hybrid_embeddings.embed_query(question)
        logger.info(f"  📝 Using text query for image search (2048-dim hybrid)")

    # Query image collection
    image_results = image_collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=['metadatas', 'documents', 'distances']
    )
    
    # Parse results with adaptive threshold
    retrieved_images = []
    
    if image_results['ids'][0]:
        distances = image_results['distances'][0]
        threshold = AdaptiveThreshold.calculate(distances, base_threshold=0.35)
        
        for i in range(len(image_results['ids'][0])):
            meta = image_results['metadatas'][0][i]
            distance = distances[i]
            similarity = (1 - distance) * 100
            
            # Filter by adaptive threshold
            if distance > threshold:
                logger.info(f"  ⚠️  Filtered out: Page {meta.get('page')} - "
                          f"Distance {distance:.3f} > {threshold:.3f} (sim: {similarity:.1f}%)")
                continue
            
            # Parse stored data including unused metadata
            retrieved_images.append({
                'page': meta.get('page', '?'),
                'image_type': meta.get('image_type', 'unknown'),
                'type': meta.get('type', 'image'),  # Use stored type
                'base64': meta.get('image_base64', ''),
                'description': meta.get('description', ''),
                'section_context': meta.get('section_context', ''),
                'key_info': json.loads(meta.get('key_info', '[]')),
                'has_text': meta.get('has_text', False),  # Use stored has_text
                'visual_embedding': json.loads(meta.get('visual_embedding', '[]')),  # Use stored visual embedding
                'semantic_embedding': json.loads(meta.get('semantic_embedding', '[]')),  # Use stored semantic embedding
                'distance': distance,
                'similarity': similarity,
                'source': 'vector_search',
                'doc_id': image_results['ids'][0][i]
            })

        logger.info(f"  ✓ Found {len(retrieved_images)} images after filtering")
        for i, img in enumerate(retrieved_images[:3], 1):
            logger.info(f"    {i}. Page {img['page']} - {img['image_type']} - {img['similarity']:.1f}% similar")

    return retrieved_images


def page_based_image_retrieval(image_collection, text_results: List[Document], k: int) -> List[Dict]:
    """Retrieve additional images from same pages as text chunks for broader context"""
    try:
        # Get pages from text chunks
        pages = set()
        for doc in text_results:
            pages.add(int(doc.metadata.get('page', 0)))

        if not pages:
            return []

        logger.info(f"  📄 Retrieving images from pages: {sorted(pages)}")

        # Get all images from these pages
        all_images = []
        for page in pages:
            page_images = image_collection.get(
                where={"page": str(page)},
                include=['metadatas', 'documents']
            )

            for i, meta in enumerate(page_images['metadatas']):
                all_images.append({
                    'page': meta.get('page', '?'),
                    'image_type': meta.get('image_type', 'unknown'),
                    'type': meta.get('type', 'image'),  # Use stored type
                    'base64': meta.get('image_base64', ''),
                    'description': page_images['documents'][i],
                    'section_context': meta.get('section_context', ''),
                    'key_info': json.loads(meta.get('key_info', '[]')),
                    'has_text': meta.get('has_text', False),  # Use stored has_text
                    'visual_embedding': json.loads(meta.get('visual_embedding', '[]')),  # Use stored visual embedding
                    'semantic_embedding': json.loads(meta.get('semantic_embedding', '[]')),  # Use stored semantic embedding
                    'distance': 0.0,
                    'similarity': 100.0,
                    'source': 'page_context',
                    'doc_id': page_images['ids'][i] if 'ids' in page_images else f"page_{page}_{i}"
                })

        logger.info(f"  ✓ Found {len(all_images)} images from related pages")
        return all_images[:k]

    except Exception as e:
        logger.warning(f"  ⚠️  Failed page-based retrieval: {e}")
        return []


def generate_answer(question: str, context: Dict, intent: QueryIntent, 
                   user_image_path: Optional[str] = None) -> str:
    """Generate answer using multimodal context"""
    logger.info("🤖 Generating answer...")
    
    model = ChatOpenAI(
        api_key=os.getenv("OPEN_ROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        model="openai/gpt-4o-mini",
        temperature=0,
        max_tokens=1000
    )
    
    # Build message content
    content = [{"type": "text", "text": f"""Answer this question based on the provided context.

Question: {question}

Instructions:
- Use ONLY the provided context (text and images)
- Be precise and factual
- If context doesn't contain the answer, say "I cannot answer based on the provided context"
- For data questions, extract exact numbers/values
- For visual questions, describe what you see in the images
- Reference page numbers when possible

Context:"""}]
    
    # Add text context
    if context['text']:
        text_content = "\n\nTEXTUAL INFORMATION:\n"
        for i, doc in enumerate(context['text'], 1):
            text_content += f"\n--- Text Chunk {i} (Page {doc.metadata.get('page', '?')}) ---\n"
            text_content += doc.page_content + "\n"
        content.append({"type": "text", "text": text_content})
    
    # Add image context
    if context['images']:
        image_content = "\n\nVISUAL INFORMATION:\n"
        for i, img in enumerate(context['images'], 1):
            image_content += f"\n--- Image {i} (Page {img['page']}, Type: {img['image_type']}, Has Text: {img.get('has_text', False)}) ---\n"
            image_content += f"Description: {img['description']}\n"
            if img['key_info']:
                image_content += f"Key Information: {', '.join(img['key_info'])}\n"
            if img['section_context']:
                image_content += f"Section Context: {img['section_context']}\n"

            # Add embedding info for advanced analysis
            if img.get('visual_embedding') and img.get('semantic_embedding'):
                image_content += f"Embedding Info: Visual vector available ({len(img['visual_embedding'])}d), Semantic vector available ({len(img['semantic_embedding'])}d)\n"

            # Add actual image
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img['base64']}"}
            })
        
        content.append({"type": "text", "text": image_content})
    
    # Add user image if provided
    if user_image_path:
        user_image_b64 = encode_image(user_image_path)
        content.append({
            "type": "image_url", 
            "image_url": {"url": f"data:image/jpeg;base64,{user_image_b64}"}
        })
        content.append({
            "type": "text",
            "text": f"\n\nUSER PROVIDED IMAGE: Analyze this image in relation to the question."
        })
    
    content.append({"type": "text", "text": "\n\nBased on all the context above, provide your answer:"})
    
    messages = [HumanMessage(content=content)]
    response = model.invoke(messages)
    return response.content


def ask_question(question: str, persist_dir: str = "./exp3/chroma_db",
                user_image_path: Optional[str] = None) -> str:
    """Main QA pipeline"""
    logger.info("="*60)
    logger.info(f"🚀 MULTIMODAL RAG QA SYSTEM")
    logger.info(f"📋 Question: {question}")
    if user_image_path:
        logger.info(f"📷 User Image: {user_image_path}")
    logger.info("="*60)
    
    # Load environment
    load_dotenv(override=True)

    # Initialize components
    text_vectorstore = Chroma(
        persist_directory=persist_dir,
        collection_name="text_chunks",
        embedding_function=OpenAIEmbeddings()
    )

    chroma_client = chromadb.PersistentClient(path=persist_dir)
    image_collection = chroma_client.get_collection(name="image_vectors")

    # Initialize both CLIP and Hybrid embeddings
    clip_embeddings = OpenCLIPEmbeddings(
        model_name="ViT-B-32",
        checkpoint="openai"
    )
    hybrid_embedder = HybridImageEmbedder(visual_weight=0.7, semantic_weight=0.3)

    # Initialize processing components
    classifier = QueryClassifier()
    reranker = MultimodalReranker()

    # Step 1: Classify query intent
    intent = classifier.classify(question, bool(user_image_path))
    logger.info(f"🔍 Search Strategy: {intent.search_strategy}")

    # Step 2: Adaptive retrieval based on search strategy
    if intent.search_strategy == "image_first":
        # Image-first search: Search images, then get text context
        logger.info("🔍 Using IMAGE-FIRST search strategy")
        image_results = retrieve_images(
            image_collection, question, user_image_path, intent, clip_embeddings, hybrid_embedder, None
        )
        # Get text context from images
        text_results = get_text_from_images(image_results, text_vectorstore, question, intent)
    else:
        # Text-first or balanced: Default strategy
        logger.info("🔍 Using TEXT-FIRST search strategy")
        # Step 2: Retrieve text content first
        text_results = retrieve_text(text_vectorstore, question, intent)

        # Step 3: Retrieve images with text-aware strategy
        image_results = retrieve_images(
            image_collection, question, user_image_path, intent, clip_embeddings, hybrid_embedder, text_results
        )

    # Step 4: Re-rank for better relevance with adaptive top_k
    adaptive_k = get_adaptive_k(intent)
    rerank_text_k = min(3, adaptive_k["text"])  # Rerank to max 3 or available
    rerank_image_k = min(3, adaptive_k["images"])  # Rerank to max 3 or available

    if text_results:
        text_results = reranker.rerank_text(question, text_results, top_k=rerank_text_k)
    if image_results:
        image_results = reranker.rerank_images(question, image_results, top_k=rerank_image_k)

    # Step 5: Optimize context within token budget
    context = ContextOptimizer.optimize(text_results, image_results, max_tokens=8000)

    # Step 6: Generate answer
    answer = generate_answer(question, context, intent, user_image_path)

    logger.info("="*60)
    logger.info("✅ ANSWER GENERATED")
    logger.info("="*60)
    return answer


if __name__ == "__main__":
    # 3 Sample questions for testing different query types

    print("🚀 Multimodal RAG QA System - Testing Mode")
    print("=" * 60)

    # Sample 1: Text-only factual query
    print("\n📝 SAMPLE 1: Text-only factual query")
    print("-" * 40)
    question1 = "Khi trời lạnh hoặc ở nhiệt độ thấp thì cần phải khởi động bộ điều khiển như thế nào?, trả lời bằng tiếng Việt"
    print(f"❓ Question: {question1}")
    print("🔄 Processing...")
    answer1 = ask_question(question1)
    print(f"💡 Answer:\n{answer1}\n")

    # Sample 2: Visual question (text + expected images)
    print("📊 SAMPLE 2: Visual question (needs both text and images)")
    print("-" * 60)
    question2 = "Ảnh này là gì vậy, bạn biết không?"
    image_path2 = "Screenshot 2025-10-17 170459.png"
    print(f"❓ Question: {question2}")
    print("🔄 Processing...")
    answer2 = ask_question(
        question=question2,
        user_image_path=image_path2
    )
    print(f"💡 Answer:\n{answer2}\n")

    # Sample 3: Analysis query (deep analysis with both modalities)
    print("📈 SAMPLE 3: Deep analysis query (comprehensive multimodal)")
    print("-" * 60)
    question3 = "What is amount of grease of E-axis"
    print(f"❓ Question: {question3}")
    print("🔄 Processing...")
    answer3 = ask_question(question3)
    print(f"💡 Answer:\n{answer3}\n")

    print("=" * 60)
    print("✅ Testing completed!")
    print("\n🎯 You can now ask your own questions below (or 'quit' to exit)")
    print("-" * 50)

    # Interactive mode
    while True:
        question = input("\n❓ Ask a question: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break

        if not question:
            continue

        print("\n🔄 Processing...")
        answer = ask_question(question)

        print(f"\n💡 Answer:\n{answer}\n")
        print("-" * 50)