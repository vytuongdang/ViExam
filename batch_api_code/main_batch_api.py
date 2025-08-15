# batch_api_code/batch_call_api.py

import argparse
import json
import os


from handlers import batch_openai_handler

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
    parser = argparse.ArgumentParser(description="Submit a prepared batch job and monitor it.")
    parser.add_argument("--batch-run-dir", type=str, required=True, help="Directory containing the prepared batch files and summary.")
    parser.add_argument("--output-dir", type=str, default="results", help="Base directory to save final combined results.")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of batch files to process concurrently.")
    parser.add_argument("--check-interval", type=int, default=20, help="Interval in seconds to check batch status.")
    parser.add_argument("--test", type=int, default=None, metavar="N", help="Test 'call' by submitting only the first N prepared batch files.")

    args = parser.parse_args()

    summary_path = os.path.join(args.batch_run_dir, "batch_run_summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Cannot find batch_run_summary.json in {args.batch_run_dir}")
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    model_name = summary.get('model')
    if not model_name:
        raise ValueError("Model name not found in batch_run_summary.json")

    provider = get_model_provider(model_name)
    print(f"--- Starting Batch Execution for {provider.upper()} ---")

    batch_files_to_process = summary.get('batch_files', [])
    
    if args.test is not None:
        if args.test > 0:
            print(f"--- RUNNING IN TEST MODE: Submitting only the first {args.test} of {len(batch_files_to_process)} batch file(s). ---")
            batch_files_to_process = batch_files_to_process[:args.test]
        else:
            print("Test value is 0 or less, no batches will be processed.")
            batch_files_to_process = []
    
    if not batch_files_to_process:
        print("No batch files to process. Exiting.")
        return

    handler = None
    if provider == 'openai':
        batch_openai_handler.execute_batches(args, summary, batch_files_to_process)
    else:
        raise NotImplementedError(f"Batch execution is not implemented for provider")

if __name__ == "__main__":
    main()