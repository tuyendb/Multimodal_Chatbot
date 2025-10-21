# ENRICHED TEXT CHUNK EXAMPLE

## Original Text Chunk:
```
Revenue increased by 25% compared to previous year. The following chart shows quarterly performance. Our strategic initiatives have started to bear fruit, particularly in the digital transformation segment.
```

## Enriched Text Chunk (after adding image context):
```
Revenue increased by 25% compared to previous year. The following chart shows quarterly performance. Our strategic initiatives have started to bear fruit, particularly in the digital transformation segment.

[Images on this page]:
- chart: Bar chart showing quarterly revenue: Q1: $2.5M, Q2: $3.1M, Q3: $2.8M, Q4: $3.5M with growth trend...
- table: Product revenue breakdown: Software $1.2M, Hardware $0.8M, Services $0.5M...
```

## Document Metadata:
```python
Document(
    page_content="Revenue increased by 25% compared to previous year. The following chart shows quarterly performance. Our strategic initiatives have started to bear fruit, particularly in the digital transformation segment.\n\n[Images on this page]:\n- chart: Bar chart showing quarterly revenue: Q1: $2.5M, Q2: $3.1M, Q3: $2.8M, Q4: $3.5M with growth trend...\n- table: Product revenue breakdown: Software $1.2M, Hardware $0.8M, Services $0.5M...\n",
    metadata={
        "page": 0,                    # Same page as images
        "chunk_index": 1,             # Chunk position on page
        "type": "text",
        "related_images": json.dumps(["img_001", "img_002"]),  # Direct link!
        "doc_id": "chunk_uuid_123",
        "source": "Document 173322.pdf"
    }
)
```

# RETRIEVAL BENEFITS

## User Query: "Revenue Q2 là bao nhiêu?"

### Traditional RAG (Without Context):
```
Text Chunk Found: "Revenue increased by 25% compared to previous year. The following chart shows quarterly performance..."
→ LLM không biết Q2 revenue là bao nhiêu
→ Cần separate image search → Complex orchestration
```

### Enhanced RAG (With Context):
```
Text Chunk Found: "Revenue increased by 25% compared to previous year. The following chart shows quarterly performance.\n\n[Images on this page]:\n- chart: Bar chart showing quarterly revenue: Q1: $2.5M, Q2: $3.1M, Q3: $2.8M, Q4: $3.5M with growth trend..."
→ LLM thấy ngay Q2: $3.1M trong text context
→ Answer trực tiếp, chính xác, fast
```

# ADVANCED BENEFITS

## 1. Semantic Coherence:
- Text và descriptions "sống cùng nhau"
- LLM hiểu mối quan hệ giữa text và visual information

## 2. Retrieval Efficiency:
- Single search thay vì multiple searches
- Context preserved trong embedding space

## 3. Cross-Reference Capability:
- Metadata `related_images` cho phép direct image lookup
- Flexible: có thể retrieve images riêng nếu cần

## 4. Error Resilience:
- Nếu image retrieval fails, vẫn có descriptions trong text
- Multiple pathways để get information

# TECHNICAL IMPLEMENTATION DETAILS

## Chunk Size Considerations:
```python
chunk_size=1000, chunk_overlap=200
# → Mỗi chunk ~1000 chars + image context (~300 chars)
# → Total ~1300 chars per enriched chunk
```

## Memory Usage:
- Text chunks: Increased by ~30% due to image context
- Benefit: Improved retrieval accuracy outweighs memory cost

## Search Performance:
- Single vector search retrieves rich context
- Eliminates need for complex multi-modal orchestration
- Faster response times