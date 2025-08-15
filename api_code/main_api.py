# main_api.py
import argparse
import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Import handlers from the api_handlers package
from api_handlers import openai_handler, claude_handler, gemini_handler, aya_handler, openrouter_handler

thread_local = threading.local()


def read_api_key(key_path):
    try:
        with open(key_path, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"API key not found at {key_path}. Please make sure the file exists.")

def ensure_dir_exists(file_path):
    directory = os.path.dirname(file_path)
    if directory:
        Path(directory).mkdir(parents=True, exist_ok=True)

def load_existing_results(output_file):
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: {output_file} exists but is not valid JSON. Starting fresh.")
    return []

def get_processed_ids(results):
    return {item['ID'] for item in results if 'ID' in item}

def get_output_filename(input_file, base_output_dir, model_folder, prompt_lang):
    input_basename = os.path.basename(input_file)
    input_name, extension = os.path.splitext(input_basename)
    output_filename = f"{input_name}_{prompt_lang}{extension}"
    return os.path.join(base_output_dir, model_folder, output_filename)

def save_results(results, output_file, lock):
    with lock:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

def get_model_provider(model_name):
    """Determines the API provider based on the model name."""
    model_lower = model_name.lower()
    if 'gpt' in model_lower or 'o3' in model_lower:
        return 'openai'
    elif 'claude' in model_lower:
        return 'claude'
    elif 'gemini' in model_lower or 'gemma' in model_lower:
        return 'gemini'
    elif 'aya' in model_lower or 'c4ai-aya-vision' in model_lower:
        return 'aya'
    elif 'qwen' in model_lower or 'deepseek' in model_lower or 'mistral' in model_lower or 'llama' in model_lower:
        return 'openrouter' 
    else:
        # Mặc định là openrouter nếu không khớp, vì nó hỗ trợ nhiều model
        print(f"Warning: Could not determine provider for '{model_name}'. Defaulting to 'openrouter'.")
        return 'openrouter'

# ==============================================================================
# MAIN FUNCTION
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Unified API caller for Vision-Language Models.")
    # Common arguments
    parser.add_argument("--model", type=str, required=True, help="Model to use (e.g., 'gpt-4o', 'claude-3-sonnet-20240229', 'qwen/qwen-vl-plus')")
    parser.add_argument("--input-file", type=str, default="dataset/metadata/text_only_vqa.json", help="Input JSON file path")
    parser.add_argument("--output-file", type=str, default=None, help="Optional: Manually specify output file path.")
    parser.add_argument("--output-dir", type=str, default="results_ocr", help="Base directory for saving result files.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for the model")
    parser.add_argument("--test", type=int, default=None, metavar="N", help="Test mode with N samples.")
    parser.add_argument("--concurrency", type=int, default=300, help="Number of concurrent API calls")
    parser.add_argument("--prompt_language", type=str, default="vn", choices=['vn', 'en'], help="Choose prompt language ('vn' for Vietnamese or 'en' for English)")

  
    parser.add_argument(
        "--openrouter-ignore-providers", 
        type=str, 
        default=None,
        help="For OpenRouter only: Comma-separated list of providers to ignore (e.g., 'anthropic,google')"
    )

    args = parser.parse_args()

    # 1. Determine the provider and corresponding handler
    provider = get_model_provider(args.model)
    if provider == 'openai':
        handler = openai_handler
    elif provider == 'claude':
        handler = claude_handler
    elif provider == 'gemini':
        handler = gemini_handler
    elif provider == 'aya':
        handler = aya_handler
    elif provider == 'openrouter':
        handler = openrouter_handler
    
    print(f"Detected model provider: {provider.upper()}")

    
    if provider == 'openrouter' and args.openrouter_ignore_providers:
        print(f"OpenRouter: Ignoring providers -> {args.openrouter_ignore_providers}")


    # 2. Get API key path and read the key
    api_key_path = handler.get_api_key_path()
    api_key = read_api_key(api_key_path)

    # 3. Create the output file path
    model_folder_name = args.model.replace("/", "_") # Thay / bằng _ để tên thư mục hợp lệ
    if args.output_file is None:
        output_file = get_output_filename(args.input_file, args.output_dir, model_folder_name, args.prompt_language)
    else:
        output_file = args.output_file
    
    ensure_dir_exists(output_file)

    # 4. Load data and prepare for processing
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if args.test is not None:
        print(f"--- Running in TEST MODE with {args.test} samples ---")
        data = data[:args.test]

    results = load_existing_results(output_file)
    processed_ids = get_processed_ids(results)
    
    print(f"Found {len(results)} existing results. Continuing from where we left off.")
    print(f"Using model: {args.model}")
    print(f"Output file: {output_file}")
    
    results_lock = threading.Lock()
    save_lock = threading.Lock()
    
    items_to_process = [item for item in data if item['ID'] not in processed_ids]
    print(f"Found {len(items_to_process)} new items to process out of {len(data)} total.")
    
    # 5. Main processing loop
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
       
        future_to_item = {}
        for item in items_to_process:
           
            process_kwargs = {
                'item': item,
                'api_key': api_key,
                'model': args.model,
                'temperature': args.temperature,
                'prompt_lang': args.prompt_language
            }
            
            # If provider is 'openrouter', add 'ignore_providers' in dictionary
            if provider == 'openrouter':
                process_kwargs['ignore_providers'] = args.openrouter_ignore_providers
            
            future = executor.submit(handler.process_item, **process_kwargs)
            future_to_item[future] = item
       
        for i, future in enumerate(as_completed(future_to_item)):
            item = future_to_item[future]
            try:
                result = future.result()
                if result:
                    with results_lock:
                        results.append(result)
                    
                    if (i + 1) % 5 == 0 or i == len(items_to_process) - 1:
                        save_results(results, output_file, save_lock)
                        print(f"  → Saved progress ({len(results)}/{len(data)} items)")
            except Exception as exc:
                print(f"Item {item['ID']} generated an exception: {exc}")
    
    save_results(results, output_file, save_lock)
    print(f"All done! Processed {len(items_to_process)} new items. Total results: {len(results)}. Saved to {output_file}")

if __name__ == "__main__":
    main()