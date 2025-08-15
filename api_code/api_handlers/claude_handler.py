# api_handlers/claude_handler.py
import base64
import time
from pathlib import Path
from anthropic import Anthropic
import threading

thread_local = {}

def get_api_key_path():
    """Returns the default path for the Claude API key."""
    return "api_key/claude_key.txt"

def get_client(api_key):
    """Creates or retrieves the Anthropic client for the current thread."""
    thread_id = str(threading.get_ident())
    if thread_id not in thread_local:
        thread_local[thread_id] = Anthropic(api_key=api_key)
    return thread_local[thread_id]

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_image_media_type(image_path):
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
    client = get_client(api_key)
    base64_image = encode_image_to_base64(image_path)
    media_type = get_image_media_type(image_path)

    for attempt in range(3):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=2048, # Keeping max_tokens or expose as an argument if needed
                temperature=temperature,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": base64_image,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
            )
            return response.model_dump()
        except Exception as e:
            print(f"Attempt {attempt+1} failed for {image_path}: {e}")
            if attempt < 2:  
                print(f"Retrying in 60 seconds...")
                time.sleep(60)
            else:
                raise

def process_item(item, api_key, model, temperature, prompt_lang):
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