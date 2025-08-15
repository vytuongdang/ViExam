# api_handlers/openai_handler.py
import base64
import time
import threading
from openai import OpenAI

thread_local = {}

def get_api_key_path():
    """Returns the default path for the OpenAI API key."""
    return "api_key/openai_key.txt"

def get_client(api_key):
    """Creates or retrieves the OpenAI client for the current thread."""
    thread_id = str(threading.get_ident())
    if thread_id not in thread_local:
        thread_local[thread_id] = OpenAI(api_key=api_key)
    return thread_local[thread_id]

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def make_api_call(api_key, model, prompt, image_path, temperature):
    client = get_client(api_key)
    base64_image = encode_image_to_base64(image_path)
    
    for attempt in range(3):
        try:
            if 'o3' in model:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ]
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
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