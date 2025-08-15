# api_handlers/gemini_handler.py
import time
import threading
from PIL import Image
import google.generativeai as genai


thread_local = {}

def get_api_key_path():
    """Returns the default path for the Gemini API key."""
    return "api_key/gemini_key.txt"

def get_client(api_key):
    """Creates or retrieves the Gemini client for the current thread."""
    thread_id = str(threading.get_ident())
    if thread_id not in thread_local:
        genai.configure(api_key=api_key)
        thread_local[thread_id] = genai
    return thread_local[thread_id]

def make_api_call(api_key, model, prompt, image_path, temperature):
    client = get_client(api_key)
    image = Image.open(image_path)
    
    for attempt in range(3):
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=temperature
            )
            # Use GenerativeModel instead of client.models.generate_content
            model_instance = client.GenerativeModel(model)
            response = model_instance.generate_content(
                contents=[image, prompt],
                generation_config=generation_config
            )
            
            # Normalize the response to resemble other APIs
            response_text = ""
            try:
                response_text = response.text
            except (ValueError, IndexError):
                 # Handle cases where the response is blocked or has no text
                if response.candidates:
                    response_text = f"Blocked: {response.prompt_feedback.block_reason.name}"
                else:
                    response_text = "No content generated."

            return {"content": [{"type": "text", "text": response_text}], "model": model, "usage": {}}
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