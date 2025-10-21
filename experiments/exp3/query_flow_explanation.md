# Query and Search Flow Explanation - QA System

## Overview
The `qa.py` file implements a sophisticated multimodal RAG (Retrieval-Augmented Generation) system that can answer questions using both text and images from documents. The system uses adaptive retrieval strategies based on query intent analysis.

## Main Query Flow

### 1. Query Entry Point
```python
ask_question(question: str, persist_dir: str = "./exp3/chroma_db",
            user_image_path: Optional[str] = None,
            k_text: int = 3, k_image: int = 3) -> str
```

### 2. Query Intent Classification
**Location**: `QueryClassifier.classify()` (lines 96-169)

The system first analyzes the user's question to understand what type of information is needed:

**Query Types**:
- **factual**: Text-based questions, no visuals needed
- **visual_questions**: User wants to "see" or "show" something
- **data_extraction**: Extract numbers/data from charts/tables
- **image_only**: User provided image, wants similar/find context
- **cross_reference**: User image + text, wants comparison
- **analysis**: Complex analysis requiring both text and visuals

**Key Parameters Determined**:
- `needs_images`: Whether visual content is required
- `needs_text`: Whether text content is required
- `visual_emphasis`: 0.0-1.0 weighting for visual vs text
- `cross_modal`: Whether both text and images needed together
- `analysis_depth`: "specific" (exact data) or "contextual" (broader understanding)
- `data_extraction`: Whether extracting specific numbers/data
- `priority_visual`: Whether visual info is primary focus

### 3. Adaptive Retrieval Strategy
**Location**: `get_adaptive_k()` (lines 335-351)

Based on query intent, the system determines how many text chunks and images to retrieve:

| Query Type | Text K | Image K | Rationale |
|------------|---------|----------|-----------|
| factual | min(5, base_k) | 0 | Text-only questions |
| visual_questions | min(3, base_k) | min(3, base_k) | Balance of text and visuals |
| data_extraction | min(5, base_k) | min(5, base_k) | More context for data |
| image_only | 2 | min(8, base_k) | Focus on images |
| cross_reference | min(5, base_k) | min(10, base_k) | Max context for comparison |
| analysis | min(8, base_k) | min(6, base_k) | Deep analysis needs more context |

## Search Strategies

### 1. Text Retrieval
**Location**: `retrieve_text()` (lines 398-416)

**Process**:
1. Uses semantic similarity search with OpenAI embeddings
2. Retrieves `k` text chunks based on adaptive strategy
3. Logs results with page numbers and content previews

**Vector Store**: ChromaDB with `text_chunks` collection
**Embedding**: OpenAI text embeddings (1536 dimensions)

### 2. Image Retrieval
**Location**: `retrieve_images()` (lines 419-455)

The system uses **three different strategies** based on query type:

#### Strategy 1: Linked Images from Text Chunks (Primary)
**When Used**: `visual_questions`, `data_extraction`, `analysis` queries
**How**:
- Finds images linked to retrieved text chunks via `related_images` metadata
- Most effective strategy for contextual queries

#### Strategy 2: Vector Similarity Search (Fallback/Primary)
**When Used**: `image_only`, `cross_reference` queries, or when no text results
**How**:
- Uses hybrid embeddings combining CLIP (visual) + OpenAI (semantic)
- 2048-dimensional vectors (512 visual + 1536 semantic)
- Supports both text-to-image and image-to-image search

#### Strategy 3: Page-Based Retrieval (Additional Context)
**When Used**: `analysis` queries when text chunks are available
**How**:
- Retrieves all images from pages containing relevant text chunks
- Provides broader contextual information

### 3. Hybrid Image Embeddings
**Location**: `HybridImageEmbedder` class (lines 18-62)

**Components**:
- **Visual Embedding**: CLIP ViT-B-32 model (512 dimensions)
- **Semantic Embedding**: OpenAI text embeddings (1536 dimensions)
- **Combination**: Concatenated for 2048-dimensional hybrid vectors

**Weighting**: 70% visual, 30% semantic (configurable)

## Re-ranking System

### 1. Image Re-ranking
**Location**: `MultimodalReranker.rerank_images()` (lines 213-253)

**Process**:
1. Uses LLM to score each image's relevance to the question (0-10 scale)
2. Considers content relevance, information value, and direct answer capability
3. Sorts by score and returns top-k results
4. Logs scoring results for transparency

### 2. Text Re-ranking
**Location**: `MultimodalReranker.rerank_text()` (lines 255-277)

**Process**:
1. Simple keyword-based scoring (can be enhanced with CrossEncoder)
2. Calculates word overlap between question and text chunks
3. Sorts by relevance score

## Context Optimization

### Token Budget Management
**Location**: `ContextOptimizer.optimize()` (lines 280-326)

**Budget**: 8000 tokens total
**Cost Estimates**:
- Text: 0.25 tokens per character
- Images: 800 tokens per image

**Priority Strategy**:
1. **Priority 1**: Top 2 most relevant images
2. **Priority 2**: Fill with text chunks
3. **Priority 3**: Add more images if budget allows

## Answer Generation

### Multimodal Answer Generation
**Location**: `generate_answer()` (lines 568-643)

**Model**: GPT-4o-mini via OpenRouter API
**Max Tokens**: 1000

**Input Construction**:
1. **System Prompt**: Instructions for using provided context only
2. **Text Context**: Formatted text chunks with page numbers
3. **Image Context**: Descriptions, key info, and actual base64 images
4. **User Image**: If provided, included for analysis
5. **Final Prompt**: "Based on all the context above, provide your answer:"

**Content Structure**:
- Uses LangChain's `HumanMessage` with mixed content types
- Supports both text and `image_url` content types
- Images encoded as base64 data URLs

## Adaptive Threshold System

### Distance-based Filtering
**Location**: `AdaptiveThreshold.calculate()` (lines 175-199)

**Process**:
1. Finds significant gaps in similarity distances
2. Uses gap detection or statistical mean + 0.5*std
3. Filters out results below relevance threshold
4. Logs filtering decisions for debugging

## Key Features

### 1. Cross-modal Retrieval
- Links text chunks to related images via metadata
- Enables coordinated retrieval of related content
- Supports queries requiring both text and visual understanding

### 2. Multiple Search Strategies
- Semantic text search
- Hybrid image similarity search
- Page-based contextual retrieval
- Metadata-driven linked content retrieval

### 3. Adaptive Processing
- Query-specific retrieval strategies
- Dynamic k-values based on intent
- Budget-aware context optimization
- Threshold-based result filtering

### 4. Comprehensive Logging
- Step-by-step process logging
- Relevance scoring transparency
- Filtering decisions explanation
- Token usage tracking

## Error Handling Improvements

All try-except blocks have been removed to ensure errors surface clearly for debugging. This allows:
- Precise error identification
- Stack trace visibility
- E troubleshooting and fixes
- No silent failures with default fallbacks

## Configuration Points

### Key Parameters:
- `k_text`, `k_image`: Default retrieval amounts (currently 3)
- `max_tokens`: Context budget (8000)
- `visual_weight`, `semantic_weight`: Hybrid embedding weights (0.7, 0.3)
- `temperature`: Model determinism (0 for consistency)
- `max_tokens`: Response length (1000)

### Strategy Selection:
The system automatically selects optimal strategies based on query classification, ensuring efficient and relevant information retrieval for each question type.