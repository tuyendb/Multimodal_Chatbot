import os
import base64
import uuid

import pymupdf4llm
from dotenv import load_dotenv

from collections import defaultdict
from langchain_upstage import UpstageDocumentParseLoader
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_experimental.open_clip import OpenCLIPEmbeddings
import tempfile
from PIL import Image
from io import BytesIO
import chromadb


def create_image_descriptions_and_save(docs):
    """
    Tạo descriptions cho ảnh VÀ lưu ảnh ra file tạm để embed bằng CLIP
    Returns: (image_docs_with_descriptions, image_paths_for_clip)
    """
    model = ChatOpenAI(model="gpt-4o-mini")
    
    image_documents = []
    image_paths = []
    temp_dir = tempfile.mkdtemp()
    
    for doc in docs:
        if 'base64_encodings' in doc.metadata and len(doc.metadata['base64_encodings']) > 0:
            for idx, img_base64 in enumerate(doc.metadata['base64_encodings']):
                # Tạo description cho ảnh (để LLM hiểu context)
                message = HumanMessage(
                    content=[
                        {"type": "text", 
                         "text": """Describe only the factual content visible in the image:

1. If decorative/non-informational: output '<---image--->'

2. For content images:
- General Images: List visible objects, text, and measurable attributes
- Charts/Infographics: State all numerical values and labels present
- Tables: Convert to markdown table format with exact data

Rules:
* Include only directly observable information
* Use original numbers and text without modification
* Avoid any interpretation or analysis
* Preserve all labels and measurements exactly as shown"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                        },
                    ]
                )
                
                response = model.invoke([message])
                
                # Lưu ảnh ra file tạm để CLIP embed
                img_data = base64.b64decode(img_base64)
                image = Image.open(BytesIO(img_data))
                img_path = os.path.join(temp_dir, f"img_{uuid.uuid4().hex}.jpg")
                image.save(img_path)
                image_paths.append(img_path)
                
                # Tạo document với description VÀ path
                new_doc = Document(
                    page_content=response.content,  # Description để LLM hiểu
                    metadata={
                        "page": f"{doc.metadata.get('page', 'unknown')}",
                        "type": "image",
                        "image_base64": img_base64,  # Giữ ảnh gốc để trả về
                        "image_path": img_path,  # Path để CLIP embed
                        "doc_id": str(uuid.uuid4())
                    }
                )
                
                image_documents.append(new_doc)
    
    return image_documents, image_paths


def merge_text_and_images(md_text, image_description_docs):
    """Merge text và image descriptions theo page"""
    page_contents = defaultdict(list)
    page_metadata = {}
    
    for text_item in md_text:
        page = int(text_item['metadata']['page'])
        page_contents[page].append(text_item['text'])
        if page not in page_metadata:
            page_metadata[page] = {
                'source': text_item['metadata']['file_path'],
                'page': page,
                'type': 'text',
                'doc_id': str(uuid.uuid4())
            }
    
    for img_doc in image_description_docs:
        page = int(img_doc.metadata['page'])
        page_contents[page].append(f"\n[IMAGE DESCRIPTION]\n{img_doc.page_content}\n")
    
    merged_docs = []
    for page in sorted(page_contents.keys()):
        full_content = '\n\n'.join(page_contents[page])
        
        doc = Document(
            page_content=full_content,
            metadata=page_metadata[page]
        )
        merged_docs.append(doc)
    
    return merged_docs


def main(file_path: str):
    load_dotenv(override=True)

    # Extract text
    print("📄 Extracting text from PDF...")
    md_text = pymupdf4llm.to_markdown(
        doc=file_path,
        page_chunks=True,
        show_progress=True
    )

    print(f"✓ Extracted {len(md_text)} pages")
    print(f"Preview: {md_text[0]['text'][:200]}...")
    print("#"*50)
    
    # Load documents with images
    print("🖼️  Loading images from PDF...")
    loader = UpstageDocumentParseLoader(
        file_path, 
        split="page", 
        output_format="markdown",
        base64_encoding=["figure", "chart", "table"]
    )
    docs = loader.load_and_split()

    # Tạo image descriptions + save images
    print("📝 Creating image descriptions and saving images...")
    image_docs, image_paths = create_image_descriptions_and_save(docs)
    
    print(f"✓ Created {len(image_docs)} image documents")
    print("#"*50)

    # Merge text và image descriptions
    print("🔗 Merging text and images by page...")
    merged_documents = merge_text_and_images(md_text, image_docs)

    # Split text documents
    print("✂️  Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_documents(merged_documents)
    print(f"✓ Created {len(text_chunks)} text chunks")

    # ⭐ Sử dụng CLIP embeddings cho cả text VÀ ảnh
    print("\n🤖 Initializing CLIP embeddings...")
    clip_embeddings = OpenCLIPEmbeddings(
        model_name="ViT-B-32",  # Model nhẹ, nhanh
        checkpoint="openai"
    )
    
    # Embed text chunks bằng CLIP (text embeddings)
    print(f"📊 Embedding {len(text_chunks)} text chunks with CLIP...")
    text_vectorstore = Chroma.from_documents(
        documents=text_chunks,
        embedding=clip_embeddings,  # CLIP có thể embed text
        collection_name="text_chunks",
        persist_directory="./exp2/chroma_db"
    )
    print("✓ Text chunks indexed")
    
    # ⭐ Embed ảnh bằng CLIP (image embeddings)
    print(f"\n🖼️  Embedding {len(image_docs)} images with CLIP...")
    
    # Tạo embeddings cho ảnh
    image_embeddings = clip_embeddings.embed_image(image_paths)
    print("✓ Image embeddings created")
    
    # Tạo vector store riêng cho ảnh với CLIP embeddings
    print("💾 Saving image vectors to Chroma...")
    chroma_client = chromadb.PersistentClient(path="./exp2/chroma_db")
    
    # Tạo collection cho ảnh
    try:
        chroma_client.delete_collection(name="image_vectors")
    except:
        pass
    
    image_collection = chroma_client.create_collection(
        name="image_vectors",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Add image embeddings vào collection
    for i, (img_doc, img_emb) in enumerate(zip(image_docs, image_embeddings)):
        image_collection.add(
            ids=[img_doc.metadata['doc_id']],
            embeddings=[img_emb],
            metadatas=[{
                "page": img_doc.metadata['page'],
                "type": img_doc.metadata['type'],
                "image_base64": img_doc.metadata['image_base64'],
                "description": img_doc.page_content
            }],
            documents=[img_doc.page_content]
        )
    
    print(f"✓ Image vectors indexed")
    
    print(f"\n{'='*60}")
    print(f"✅ INDEXING COMPLETED!")
    print(f"{'='*60}")
    print(f"📊 Text chunks: {len(text_chunks)} (collection: text_chunks)")
    print(f"🖼️  Images: {len(image_docs)} (collection: image_vectors)")
    print(f"💾 Database: ./exp2/chroma_db")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    file_path = "Document 173322.pdf"
    
    main(file_path)
