import os
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from PIL import Image
import numpy as np

def example_image_embedding():
    """Example of embedding an image using OpenCLIP"""
    
    # Initialize CLIP embeddings model
    clip_embeddings = OpenCLIPEmbeddings(
        model_name="ViT-B-32",  # Lightweight model
        checkpoint="laion2b_s34b_b79k"
    )
    
    # Path to the image to embed
    image_path = "download.jpg"  # Replace with actual image path
    
    # Check if image file exists
    if not os.path.exists(image_path):
        print(f"❌ Image not found at: {image_path}")
        print("📝 Please provide a valid image path")
        return
    
    try:
        # Embed the image
        print(f"🖼️  Embedding image: {image_path}")
        image_embedding = clip_embeddings.embed_image([image_path])
        
        # Result is a list of embeddings
        embedding_vector = image_embedding[0]
        
        print(f"✅ Embedding completed!")
        print(f"📏 Vector size: {len(embedding_vector)}")
        print(f"🔢 First 10 values: {embedding_vector[:10]}")
        print(f"📊 Min/max values: {np.min(embedding_vector):.4f} / {np.max(embedding_vector):.4f}")
        
        # Example: embed text samples for comparison
        text_samples = [
            "A beautiful landscape photo",
            "A technical diagram",  
            "A person portrait"
        ]
        
        print(f"\n📝 Embedding text samples for comparison...")
        text_embeddings = clip_embeddings.embed_documents(text_samples)
        
        # Calculate similarity between image and text
        for i, text in enumerate(text_samples):
            similarity = np.dot(embedding_vector, text_embeddings[i]) / (
                np.linalg.norm(embedding_vector) * np.linalg.norm(text_embeddings[i])
            )
            print(f"📊 Similarity with '{text}': {similarity:.4f}")
            
    except Exception as e:
        print(f"❌ Error embedding image: {e}")

if __name__ == "__main__":
    example_image_embedding()