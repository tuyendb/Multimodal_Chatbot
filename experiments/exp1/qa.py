import base64
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_core.messages import HumanMessage
import chromadb


def encode_image(image_path):
    """Encode ảnh từ file thành base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def ask_question(question: str, image_path: str = None, k: int = 3):
    """
    Hỏi đáp với Multimodal RAG - Gửi cả text và images vào context
    
    Args:
        question: Câu hỏi
        image_path: Đường dẫn ảnh người dùng gửi (nếu có)
        k: Số documents cần retrieve
    
    Returns:
        answer: Câu trả lời từ AI
    """
    load_dotenv(override=True)
    
    print(f"\n{'='*60}")
    print(f"❓ Question: {question}")
    if image_path:
        print(f"🖼️  User Image: {image_path}")
    print(f"{'='*60}\n")
    
    # Initialize CLIP embeddings
    print("🤖 Loading CLIP model...")
    clip_embeddings = OpenCLIPEmbeddings(
        model_name="ViT-B-32",
        checkpoint="openai"
    )
    
    # Load text vector store
    print("📚 Loading text database...")
    text_vectorstore = Chroma(
        collection_name="text_chunks",
        embedding_function=clip_embeddings,
        persist_directory="./chroma_db"
    )
    
    # Load image vector store
    print("🖼️  Loading image database...")
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    image_collection = chroma_client.get_collection(name="image_vectors")
    
    # Initialize model
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Search text documents
    print(f"\n🔍 Searching text documents...")
    text_results = text_vectorstore.similarity_search(question, k=k)
    print(f"✓ Found {len(text_results)} relevant text chunks")
    for i, doc in enumerate(text_results[:3]):
        print(f"  📄 {i+1}. Page {doc.metadata.get('page', '?')} - {doc.page_content[:80]}...")
    
    # Search images
    retrieved_images = []
    if image_path:
        # Search by user image
        print(f"\n🎨 Embedding user image with CLIP...")
        user_image_embedding = clip_embeddings.embed_image([image_path])[0]
        
        print(f"🔍 Searching similar images in document...")
        image_results = image_collection.query(
            query_embeddings=[user_image_embedding],
            n_results=k,
            include=['metadatas', 'documents', 'distances']  # ⭐ Include all data
        )
    else:
        # Search by text query
        print(f"\n🔍 Searching images by text query...")
        text_embedding = clip_embeddings.embed_query(question)
        
        image_results = image_collection.query(
            query_embeddings=[text_embedding],
            n_results=k,
            include=['metadatas', 'documents', 'distances']
        )
    
    # Parse image results
    if image_results['ids'][0]:
        for i in range(len(image_results['ids'][0])):
            meta = image_results['metadatas'][0][i]
            
            # ⭐ Lấy base64 từ documents (đã lưu khi index)
            # Hoặc từ metadata nếu bạn lưu ở đó
            base64_data = image_results['documents'][0][i]  # Giả sử lưu base64 trong documents
            
            retrieved_images.append({
                'page': meta.get('page', '?'),
                'base64': base64_data,
                'description': meta.get('description', ''),
                'distance': image_results['distances'][0][i] if 'distances' in image_results else None
            })
        
        print(f"✓ Found {len(retrieved_images)} similar images")
        for i, img in enumerate(retrieved_images):
            dist = img.get('distance', 0)
            similarity = (1 - dist) * 100 if dist is not None else 0
            print(f"  🖼️  {i+1}. Page {img['page']} - Similarity: {similarity:.1f}%")
    
    # ⭐ Build multimodal context
    print(f"\n📝 Building multimodal context...")
    
    system_prompt = """You are a helpful AI assistant with access to both text and images from a document.

Rules:
- Analyze BOTH text content and images provided
- Compare images visually when user asks about similarities
- Reference specific page numbers
- If context doesn't contain the answer, say so clearly
- Be precise and detailed in your analysis"""
    
    # Start with system prompt and text context
    message_content = [
        {"type": "text", "text": system_prompt}
    ]
    
    # Add text chunks
    if text_results:
        text_context = "\n\n".join([
            f"[Text from page {doc.metadata.get('page', '?')}]:\n{doc.page_content}"
            for doc in text_results
        ])
        message_content.append({
            "type": "text",
            "text": f"\n\nText Context:\n{text_context}"
        })
    
    # ⭐ Add retrieved images with their descriptions
    if retrieved_images:
        message_content.append({
            "type": "text",
            "text": "\n\nRelevant Images from Document:"
        })
        
        for i, img_data in enumerate(retrieved_images):
            # ⭐ Add both page number AND description for context
            page = img_data.get('page', '?')
            desc = img_data.get('description', 'No description available')
            
            message_content.append({
                "type": "text",
                "text": f"\n[Image {i+1} from page {page}]\nDescription: {desc}"
            })
            
            # ⭐ Then add the actual image for visual analysis
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_data['base64']}"}
            })
    
    # Add user's question
    message_content.append({
        "type": "text",
        "text": f"\n\nUser's Question: {question}"
    })
    
    # ⭐ Add user's uploaded image if provided
    if image_path:
        user_image_base64 = encode_image(image_path)
        message_content.append({
            "type": "text",
            "text": "\n[User's Uploaded Image]:"
        })
        message_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{user_image_base64}"}
        })
    
    message_content.append({
        "type": "text",
        "text": "\nProvide a detailed answer based on the context above:"
    })
    
    # Generate answer
    print(f"🤔 Generating answer with {len(retrieved_images)} images...\n")
    final_message = HumanMessage(content=message_content)
    response = model.invoke([final_message])
    
    print(f"{'='*60}")
    print(f"💬 Answer:\n")
    print(response.content)
    print(f"\n{'='*60}\n")
    
    return response.content


if __name__ == "__main__":
    
    print("🚀 Multimodal RAG with Full Image Context")
    print("="*60)
    
    # Example usage
    print("\n📌 EXAMPLE: Compare uploaded chart with document charts")
    print("""
    # User uploads a chart and asks:
    ask_question(
        question="Is there a similar chart in the document? What are the differences?",
        image_path="path/to/user_chart.jpg",
        k=3  # Retrieve top 3 similar images
    )
    
    # Now LLM can SEE:
    # - User's uploaded chart
    # - 3 most similar charts from document
    # - Text context around those charts
    # → Can do visual comparison!
    """)
    
    print("\n" + "="*60)
    print("💡 INTERACTIVE MODE")
    print("="*60)
    print("Commands:")
    print("  • Type question for text+image search")
    print("  • IMAGE:/path/to/image.jpg|Your question - to upload an image")
    print("  • 'quit' to exit")
    print("="*60 + "\n")
    
    while True:
        try:
            user_input = input("❓ Your question: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if not user_input:
                continue
            
            if user_input.startswith("IMAGE:"):
                parts = user_input[6:].split("|", 1)
                if len(parts) == 2:
                    img_path, question = parts
                    ask_question(question.strip(), img_path.strip())
                else:
                    print("❌ Format: IMAGE:/path/to/image.jpg|Your question")
            else:
                ask_question(user_input)
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
