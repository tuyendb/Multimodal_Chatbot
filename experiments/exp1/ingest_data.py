import os
import base64

import pymupdf4llm
from dotenv import load_dotenv

from collections import defaultdict
from langchain_upstage import UpstageDocumentParseLoader
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


def create_image_descriptions(docs):
    # Initialize the Gemini model
    model = ChatOpenAI(model="gpt-4o-mini")
    
    new_documents = []
    
    for doc in docs:
        # Check if base64_encodings exist in metadata
        if 'base64_encodings' in doc.metadata and len(doc.metadata['base64_encodings']) > 0:
            for idx, img_base64 in enumerate(doc.metadata['base64_encodings']):
                # Create a message containing the image
                message = HumanMessage(
                    content=[
                        {"type": "text", 
                         "text": """
                                    Describe only the factual content visible in the image:

                                    1. If decorative/non-informational: output '<---image--->'

                                    2. For content images:
                                    - General Images: List visible objects, text, and measurable attributes
                                    - Charts/Infographics: State all numerical values and labels present
                                    - Tables: Convert to markdown table format with exact data

                                    Rules:
                                    * Include only directly observable information
                                    * Use original numbers and text without modification
                                    * Avoid any interpretation or analysis
                                    * Preserve all labels and measurements exactly as shown
                                """
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                        },
                    ]
                )
                
                # Obtain image description from the model
                response = model.invoke([message])
                
                # Create a new Document
                new_doc = Document(
                    page_content=response.content,
                    metadata={
                        "page": f"{doc.metadata.get('page', 'unknown')}"
                    }
                )
                
                new_documents.append(new_doc)
    
    return new_documents


def merge_text_and_images(md_text, image_description_docs):
    # Create a dictionary to collect data by page
    page_contents = defaultdict(list)
    page_metadata = {}
    
    # Process md_text
    for text_item in md_text:
        # Standardize page numbers to integer
        page = int(text_item['metadata']['page'])
        page_contents[page].append(text_item['text'])
        # Save metadata for each page
        if page not in page_metadata:
            page_metadata[page] = {
                'source': text_item['metadata']['file_path'],
                'page': page
            }
    
    # Process image_description_docs
    for img_doc in image_description_docs:
        # Standardize page numbers to integer
        page = int(img_doc.metadata['page'])
        page_contents[page].append(img_doc.page_content)
    
    # Create the final list of Document objects
    merged_docs = []
    for page in sorted(page_contents.keys()):
        # Combine all content of the page into a single string
        full_content = '\n\n'.join(page_contents[page])
        
        # Create a Document object
        doc = Document(
            page_content=full_content,
            metadata=page_metadata[page]
        )
        merged_docs.append(doc)
    
    return merged_docs


def main(file_path: str):
    load_dotenv(override=True)

    md_text = pymupdf4llm.to_markdown(
        doc=file_path,  # The file, either as a file path or a PyMuPDF Document.
        page_chunks=True,  # If True, output is a list of page-specific dictionaries.
        show_progress=True  # Displays a progress bar during processing.
        # pages=[0, 1, 2],  - Optional, specify 0-based page numbers to process.
        # hdr_info=False,  - Optional, disables header detection logic.
        # write_images=True,  - Saves images found in the document as files.
        # embed_images=True,  - Embeds images directly as base64 in markdown.
        # image_size_limit=0.05,  - Exclude small images below this size threshold.
        # dpi=150,  - Image resolution in dots per inch, if write_images=True.
        # image_path="output_images",  - Directory to save images if write_images=True.
        # image_format="png",  - Image file format, e.g., "png" or "jpg".
        # force_text=True,  - Include text overlapping images/graphics.
        # margins=0,  - Specify page margins for text extraction.
        # page_width=612,  - Desired page width for reflowable documents.
        # page_height=None,  - Desired page height for reflowable documents.
        # table_strategy="lines_strict",  - Strategy for table detection.
        # graphics_limit=5000,  - Limit the number of vector graphics processed.
        # ignore_code=False,  - If True, avoids special formatting for mono-spaced text.
        # extract_words=False,  - Adds word-level data to each page dictionary.
    )

    print(md_text[0]['text'])
    print("#"*50)
    
    loader = UpstageDocumentParseLoader(
            file_path, split="page", 
            output_format="markdown",
            base64_encoding=["figure", "chart", "table"]
        )
    docs = loader.load_and_split()

    for i, j in enumerate(docs[:5]):
        bs_encoding = j.metadata['base64_encodings']

        if len(bs_encoding) > 0:
            print(f"📄 **Page {i+1}**\n{'='*20}")
            print(f"📝 **Page Content Preview:** {j.page_content[:100]}...")
            print(f"🔑 **Metadata Keys:** {', '.join(j.metadata.keys())}")
            print(f"🖼️ **Base64 Encoding (Preview):** {bs_encoding[0][:10]}...")
            print("\n")

    image_description_docs = create_image_descriptions(docs)

    print("#"*50)

    for doc in image_description_docs[:4]:
        print(f"📄 **Page {doc.metadata['page']}**\n{'='*20}")
        print(f"Description: {doc.page_content}")
        print("---")

    merged_documents = merge_text_and_images(md_text, image_description_docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(merged_documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma.from_documents(
        documents=all_splits,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )


if __name__ == "__main__":
    file_path = "Document 173322.pdf"
    
    main(file_path)
