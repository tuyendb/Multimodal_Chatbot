import fitz  # PyMuPDF
from PIL import Image
import io
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_images_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extracts images from a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                               contains image data ('image'), page number ('page_num'),
                               and image format ('format').
                               Returns an empty list if no images are found or
                               if an error occurs.
    """
    extracted_images = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Error opening PDF file at {pdf_path}: {e}")
        return extracted_images

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        image_list = page.get_images(full=True)

        if not image_list:
            logger.info(f"No images found on page {page_num + 1}")
            continue

        logger.info(f"Found {len(image_list)} images on page {page_num + 1}")

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            try:
                pil_image = Image.open(io.BytesIO(image_bytes))
                extracted_images.append({
                    "image": pil_image,
                    "page_num": page_num + 1,
                    "format": image_ext
                })
            except Exception as e:
                logger.error(f"Error processing image {img_index + 1} on page {page_num + 1}: {e}")

    logger.info(f"Successfully extracted {len(extracted_images)} images from {pdf_path}")
    doc.close()
    return extracted_images

if __name__ == '__main__':
    # This is an example of how to use the function.
    # You would need a PDF with images to test this.
    
    # Example usage:
    # pdf_file = "path/to/your/document.pdf"
    # images = extract_images_from_pdf(pdf_file)
    # if images:
    #     for idx, img_data in enumerate(images):
    #         img_data["image"].save(f"extracted_image_{idx}.{img_data['format']}")
    #         print(f"Saved image {idx} from page {img_data['page_num']}")
    
    print("Image extractor script is ready to be used.")
