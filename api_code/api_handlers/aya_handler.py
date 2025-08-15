# api_handlers/aya_handler.py
import base64
import time
from pathlib import Path
import cohere
import threading

thread_local = {}

def get_api_key_path():
    """Returns the default path for the Cohere API key."""
    return "api_key/cohere_key.txt"

def get_client(api_key):
    """Creates or retrieves the Cohere client for the current thread."""
    thread_id = str(threading.get_ident())
    if thread_id not in thread_local:
        thread_local[thread_id] = cohere.ClientV2(api_key=api_key)
    return thread_local[thread_id]

def encode_image_to_base64(image_path):
    """Encodes an image file to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_image_media_type(image_path):
    """Determines the media type (MIME type) of an image based on its extension."""
    ext = Path(image_path).suffix.lower()
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    elif ext == ".png":
        return "image/png"
    elif ext == ".webp":
        return "image/webp"
    elif ext == ".gif":
        return "image/gif"
    else:
        print(f"Warning: Unknown image type for {image_path}, defaulting to image/jpeg.")
        return "image/jpeg"

def make_api_call(api_key, model, prompt, image_path, temperature):
    """Makes an API call to the Cohere Aya Vision model."""
    client = get_client(api_key)
    base64_image = encode_image_to_base64(image_path)
    media_type = get_image_media_type(image_path)
    base64_image_url = f"data:{media_type};base64,{base64_image}"

    for attempt in range(3):
        try:
            # Use the correct Cohere Aya Vision format based on documentation
            response = client.chat(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": base64_image_url},
                            },
                        ],
                    }
                ],
                temperature=temperature,
            )
            
            # Extract response text using the correct format
            response_text = response.message.content[0].text
            
            # Convert response to match the expected format (similar to Claude handler)
            # Handle usage object properly to avoid JSON serialization errors
            usage_info = {}
            if hasattr(response, 'usage') and response.usage:
                try:
                    usage_info = {
                        "input_tokens": getattr(response.usage, 'input_tokens', 0),
                        "output_tokens": getattr(response.usage, 'output_tokens', 0),
                        "total_tokens": getattr(response.usage, 'total_tokens', 0)
                    }
                except:
                    usage_info = {}
            
            return {
                "content": [{"type": "text", "text": response_text}],
                "model": model,
                "usage": usage_info,
                "id": getattr(response, 'id', ''),
                "role": "assistant",
                "stop_reason": getattr(response, 'finish_reason', None)
            }
            
        except Exception as e:
            print(f"Attempt {attempt+1} failed for {image_path}: {e}")
            if attempt < 2:
                print(f"Retrying in 60 seconds...")
                time.sleep(60)
            else:
                raise

def process_item(item, api_key, model, temperature, prompt_lang):
    """
    Processes a single item by making an API call and appending the response.
    This function follows the same structure as the Claude handler.
    """
    item_id = item['ID']
    image_path = item['image_path']
    
    prompt_key = f"{prompt_lang}_prompt"
    if prompt_key not in item:
        print(f"ERROR: Item {item_id} is missing '{prompt_key}'. Skipping.")
        return None
        
    prompt_text = item[prompt_key]
    
    try:
        response_dict = make_api_call(api_key, model, prompt_text, image_path, temperature)
        item_with_response = item.copy()
        item_with_response['api_response'] = response_dict
        print(f"Successfully processed {item_id}")
        return item_with_response
    except Exception as e:
        print(f"Failed to process {item_id}: {e}")
        return None