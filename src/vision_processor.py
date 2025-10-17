import openai
import base64
import os
import logging
from PIL import Image
import io
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for OpenAI API key
# In a production environment, it's recommended to use a more secure way to manage API keys.
if "OPENAI_API_KEY" in os.environ:
    openai.api_key = os.environ["OPENAI_API_KEY"]
else:
    logger.warning("OPENAI_API_KEY environment variable not set. Some features may not work.")

def encode_image_to_base64(image: Image.Image) -> str:
    """Encodes a PIL Image to a base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def analyze_image_with_gpt4v(image: Image.Image, prompt: str) -> str:
    """
    Analyzes an image using GPT-4 Vision and a text prompt.

    Args:
        image (Image.Image): The image to analyze.
        prompt (str): The text prompt to guide the analysis.

    Returns:
        str: The analysis result from the model. Returns an error message
             if the API call fails or the API key is not set.
    """
    if not openai.api_key:
        error_msg = "Error: OPENAI_API_KEY not configured."
        logger.error(error_msg)
        return error_msg

    base64_image = encode_image_to_base64(image)

    try:
        # Note: The OpenAI Python library version and API structure may change.
        # This code is based on a version that supports the chat completions endpoint with image inputs.
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error calling OpenAI Vision API: {e}")
        return f"Error analyzing image: {e}"

if __name__ == '__main__':
    # This is an example of how to use the function.
    # You would need a sample image and a valid OpenAI API key to test this.
    
    # Example usage:
    # if openai.api_key:
    #     try:
    #         # Create a dummy image for testing
    #         dummy_image = Image.new('RGB', (100, 100), color = 'blue')
    #         prompt_text = "What is the dominant color of this image?"
    #         analysis = analyze_image_with_gpt4v(dummy_image, prompt_text)
    #         print(f"Analysis result: {analysis}")
    #     except Exception as e:
    #         print(f"An error occurred during the example run: {e}")
    # else:
    #     print("Skipping example run as OPENAI_API_KEY is not set.")

    print("Vision processor script is ready to be used.")
