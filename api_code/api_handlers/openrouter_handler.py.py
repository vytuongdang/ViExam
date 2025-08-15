
import base64
import json
import time
import requests 
from urllib.parse import urlparse

def get_api_key_path():
    """Returns the default path for the OpenRouter API key file."""
    return "api_key/openrouter_key.txt"

def is_url(path):
    """Checks if a string is a valid URL."""
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except:
        return False

def encode_image_to_base64(image_path):
    """Encodes a local image file to a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"ERROR: Image file not found at {image_path}")
        raise

def make_api_call(api_key, model, prompt, image_path, temperature, ignore_providers=None):
    """Makes an API call to OpenRouter using requests."""
    # Handle image (URL or local file)
    if is_url(image_path):
        image_content = {'type': 'image_url', 'image_url': {'url': image_path}}
    else:
        base64_image = encode_image_to_base64(image_path)
        image_content = {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{base64_image}'}}
    
    # Build headers
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
        "HTTP-Referer": "https://github.com/your-repo", 
        "X-Title": "xxxxx",
    }

    # Build payload
    payload = {
        'model': model,
        'temperature': temperature,
        'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': prompt}, image_content]}]
    }
    
    # Add provider preferences to the payload 
    if ignore_providers:
        provider_prefs = {'ignore': [p.strip() for p in ignore_providers.split(',')]}
        payload['provider'] = provider_prefs

    # Retry logic and API call
    for attempt in range(3):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                data=json.dumps(payload)
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt+1} failed for {image_path} with model {model}: {e}")
            if attempt < 2: 
                print("Retrying in 60 seconds...")
                time.sleep(60)
            else:
                raise

def process_item(item, api_key, model, temperature, prompt_lang, ignore_providers=None):
    """Main processing function called by main_api.py"""
    item_id = item['ID']
    image_path = item['image_path']
    
    prompt_key = f"{prompt_lang}_prompt"
    prompt_text = item.get(prompt_key)
    
    if not prompt_text:
        print(f"ERROR: Item {item_id} is missing '{prompt_key}'. Skipping.")
        return None
        
    print(f"Processing {item_id} with model {model}...")
    try:
        response_dict = make_api_call(api_key, model, prompt_text, image_path, temperature, ignore_providers)
        item_with_response = item.copy()
        item_with_response['api_response'] = response_dict
        print(f"Successfully processed {item_id}")
        return item_with_response
    except Exception as e:
        print(f"Failed to process {item_id} after all retries: {e}")
        return None