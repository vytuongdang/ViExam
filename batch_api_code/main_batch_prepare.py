# batch_api_code/batch_prepare_api.py

import argparse
import os

from handlers import batch_openai_handler, batch_claude_handler

def get_model_provider(model_name):
    """Xác định nhà cung cấp API dựa vào tên model."""
    model_lower = model_name.lower()
    if 'gpt' in model_lower or 'o3' in model_lower:
        return 'openai'
    elif 'claude' in model_lower:
        return 'claude'
    else:
        raise ValueError(f"Unknown model provider for batch processing: {model_name}")

def main():
    parser = argparse.ArgumentParser(description="Prepare .jsonl files for a batch job.")
    parser.add_argument("--model", type=str, required=True, help="Model to use (e.g., 'gpt-4o', 'claude-3-sonnet-20240229')")
    parser.add_argument("--input-file", type=str, required=True, help="Input JSON file path with all requests.")
    parser.add_argument("--output-dir", type=str, default="batches", help="Base directory to store prepared batch files.")
    parser.add_argument("--prompt_language", type=str, default="vn", choices=['vn', 'en'], help="Choose prompt language.")
    
    batch_split_group = parser.add_mutually_exclusive_group()
    batch_split_group.add_argument("--num-batches", type=int, help="Split data into N equal-sized batches.")
    batch_split_group.add_argument("--max-batch-size", type=float, default=None, metavar="MB", help="Split into batches not exceeding a max size in MB.")

    parser.add_argument("--test", type=int, default=None, metavar="N", help="Test mode with N samples.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for the model.")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens for Claude model responses.")

    args = parser.parse_args()
    
    if args.max_batch_size is None and args.num_batches is None:
        print("INFO: No split method chosen. Defaulting to max batch size of 185MB.")
        args.max_batch_size = 185

    provider = get_model_provider(args.model)
    print(f"--- Starting Batch Preparation for {provider.upper()} ---")
    
    handler = None
    if provider == 'openai':
        batch_openai_handler.execute_batches(args, summary, batch_files_to_process)
    else:
        raise NotImplementedError(f"Batch execution is not implemented for provider")


if __name__ == "__main__":
    main()