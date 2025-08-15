# batch_api_code/handlers/batch_openai_handler.py
import json
import os
import base64
import time
import datetime
import math
from openai import OpenAI


def ensure_dir_exists(directory):
    os.makedirs(directory, exist_ok=True)

def read_api_key(key_path):
    with open(key_path, 'r') as f:
        return f.read().strip()

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def create_openai_request(item, model, temperature, prompt_lang):

    image_path = item['image_path']
    base64_image = encode_image_to_base64(image_path)
    prompt_key = f"{prompt_lang}_prompt"
    prompt_text = item.get(prompt_key, item.get('prompt', ''))


    body = {
        "model": model,
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ]
    }


    if 'o3' not in model:
        body["temperature"] = temperature

    return {
        "custom_id": item['ID'],
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body
    }

def estimate_request_size(item, prompt_lang):
    base_size = 500
    prompt_key = f"{prompt_lang}_prompt"
    prompt_text = item.get(prompt_key, item.get('prompt', ''))
    prompt_size = len(prompt_text.encode('utf-8'))
    image_size = 0
    if item.get('image_path') and os.path.exists(item['image_path']):
        image_size = os.path.getsize(item['image_path']) * 4/3
    return base_size + prompt_size + image_size

def create_size_based_batches(data, max_batch_size_mb, prompt_lang):
    max_batch_size_bytes = max_batch_size_mb * 1024 * 1024
    batches, current_batch, current_batch_size = [], [], 0
    
    for item in data:
        item_size = estimate_request_size(item, prompt_lang)
        if item_size > max_batch_size_bytes:
            print(f"Warning: Item {item.get('ID')} size ({item_size/1e6:.2f}MB) exceeds max batch size. Creating a separate batch.")
            if current_batch:
                batches.append(current_batch)
            batches.append([item])
            current_batch, current_batch_size = [], 0
            continue
        
        if current_batch and (current_batch_size + item_size) > max_batch_size_bytes:
            batches.append(current_batch)
            current_batch, current_batch_size = [], 0
        
        current_batch.append(item)
        current_batch_size += item_size
        
    if current_batch:
        batches.append(current_batch)
    return batches
    
def prepare_batches(args):
    """Main function to prepare batch files for OpenAI."""
    input_file_name = os.path.splitext(os.path.basename(args.input_file))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_run_dir = os.path.join(args.output_dir, args.model, f"{input_file_name}_{args.prompt_language}_{timestamp}")
    ensure_dir_exists(batch_run_dir)

    with open(args.input_file, 'r') as f:
        data = json.load(f)
    if args.test:
        data = data[:args.test]

    if args.max_batch_size is not None:
        print(f"Splitting data into batches with max size of {args.max_batch_size}MB...")
        data_batches = create_size_based_batches(data, args.max_batch_size, args.prompt_language)
    else:
        print(f"Splitting data into {args.num_batches} equal-sized batches...")
        num_batches = args.num_batches
        batch_size = max(1, math.ceil(len(data) / num_batches))
        data_batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    batch_files = []
    for i, batch_data in enumerate(data_batches):
        batch_file_path = os.path.join(batch_run_dir, f"batch_{i+1}.jsonl")
        with open(batch_file_path, 'w') as f:
            for item in batch_data:
                try:
                    request = create_openai_request(item, args.model, args.temperature, args.prompt_language)
                    f.write(json.dumps(request) + '\n')
                except Exception as e:
                    print(f"Skipping item {item.get('ID')} due to error: {e}")
        batch_files.append(batch_file_path)
        print(f"Created batch file: {batch_file_path}")

    summary = {
        "batch_run_dir": batch_run_dir,
        "batch_files": batch_files,
        "model": args.model,
        "prompt_language": args.prompt_language,
        "input_file": args.input_file,
        "timestamp": timestamp,
        "total_requests": len(data)
    }
    summary_path = os.path.join(batch_run_dir, "batch_run_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nBatch preparation complete! Info saved to {summary_path}")
    print(f"Run the following command to execute the job:")
    print(f"python batch_api_code/main_batch_api.py --batch-run-dir {batch_run_dir}")

def execute_batches(args, summary, batch_files_to_process):
    """Main function to execute OpenAI batch jobs."""
    api_key = read_api_key("api_key/openai_key.txt")
    client = OpenAI(api_key=api_key)

    batch_jobs_info = []
    for batch_file_path in batch_files_to_process:
        print(f"Uploading {os.path.basename(batch_file_path)}...")
        with open(batch_file_path, "rb") as f:
            batch_input_file = client.files.create(file=f, purpose="batch")
        
        print(f"Creating batch job for file ID: {batch_input_file.id}")
        batch_job = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        batch_jobs_info.append({"id": batch_job.id, "status": batch_job.status, "output_file_id": None, "error_file_id": None})
        print(f"Batch job created with ID: {batch_job.id}, Status: {batch_job.status}")

    all_completed = False
    while not all_completed:
        all_completed = True
        print(f"\n--- Checking statuses at {datetime.datetime.now().strftime('%H:%M:%S')} ---")
        for job_info in batch_jobs_info:
            if job_info['status'] not in ['completed', 'failed', 'expired', 'cancelled']:
                all_completed = False
                retrieved_job = client.batches.retrieve(job_info['id'])
                job_info['status'] = retrieved_job.status
                job_info['output_file_id'] = retrieved_job.output_file_id
                job_info['error_file_id'] = retrieved_job.error_file_id
                print(f"Job {job_info['id']}: {job_info['status']}")
        
        if not all_completed:
            print(f"Not all jobs completed. Waiting {args.check_interval} seconds...")
            time.sleep(args.check_interval)

    print("\n--- All jobs finished! Downloading results... ---")
    all_results = []
    with open(summary['input_file'], 'r') as f:
        original_data_map = {item['ID']: item for item in json.load(f)}

    for job_info in batch_jobs_info:
        if job_info['status'] == 'completed' and job_info['output_file_id']:
            content = client.files.content(job_info['output_file_id']).read()
            for line in content.decode('utf-8').strip().split('\n'):
                try:
                    res = json.loads(line)
                    custom_id = res.get('custom_id')
                    if custom_id in original_data_map:
                        original_item = original_data_map[custom_id]
                        combined_item = original_item.copy()
                        combined_item['api_response'] = res.get('response', {})
                        all_results.append(combined_item)
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON line from result file for job {job_info['id']}.")
    
    input_file_name = os.path.splitext(os.path.basename(summary['input_file']))[0]
    final_output_filename = f"{input_file_name}_{summary['prompt_language']}_{summary['timestamp']}.json"
    final_output_path = os.path.join(args.output_dir, summary['model'], final_output_filename)
    
    ensure_dir_exists(os.path.dirname(final_output_path))
    with open(final_output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nExecution complete! Combined {len(all_results)} results and saved to: {final_output_path}")