import json
import re
import pandas as pd
import numpy as np
import os
from collections import defaultdict

def extract_answer_from_text(text):
    """Extract answer from curly braces in text"""
    if not text:
        return None
    matches = re.findall(r'\{([^}]*)\}', text)
    if matches:
        return matches[0].strip()
    return None

def normalize_answer(answer):
    """Normalize answer by removing spaces, commas, and other punctuation"""
    if not answer:
        return ""
    normalized = re.sub(r'[,\s\.\-\(\)\[\]]+', '', str(answer))
    return normalized.upper()

def load_and_process_json(file_path, model_name):
    """Load JSON file and extract answers based on model type"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error decoding JSON: {file_path}")
        return []
    
    results = []

    new_openrouter_models = [
        'qwen/qwen2.5-vl-32b-instruct', 
        'qwen/qwen2.5-vl-72b-instruct',
        'llama-4-maverick', 
        'llama-4-scout',
        'mistral-medium-3', 
        'mistral-small-3.2-24b-instruct'
    ]
    
    for item in data:
        try:
            question_id = item.get('ID', '')
            subject = item.get('subject', '')
            ground_truth = item.get('ground_truth', '').strip()
            extracted_answer = None
            api_response = item.get('api_response', {})

           
            if model_name in ['gpt-4.1-2025-04-14', 'o3-2025-04-16']:
                body = api_response.get('body', {})
                choices = body.get('choices', [])
                if choices:
                    message = choices[0].get('message', {})
                    content = message.get('content', '')
                    extracted_answer = extract_answer_from_text(content)
            
         
            elif model_name in new_openrouter_models:
                #  OpenRouter: api_response -> choices -> message -> content
                choices = api_response.get('choices', [])
                if choices:
                    message = choices[0].get('message', {})
                    content = message.get('content', '')
                    extracted_answer = extract_answer_from_text(content)

        =
            else:
                # Cấu trúc: api_response -> content -> text
                content_list = api_response.get('content', [])
                if content_list and isinstance(content_list, list) and len(content_list) > 0 and 'text' in content_list[0]:
                    text = content_list[0].get('text', '')
                    extracted_answer = extract_answer_from_text(text)
            
            results.append({
                'ID': question_id,
                'subject': subject,
                'ground_truth': ground_truth,
                'extracted_answer': extracted_answer,
                'has_extraction': extracted_answer is not None,
                'is_correct': extracted_answer is not None and normalize_answer(extracted_answer) == normalize_answer(ground_truth)
            })
            
        except Exception as e:
            print(f"Error processing item in {model_name} (ID: {item.get('ID')}): {e}")
            continue
    
    return results

def calculate_rates_by_subject(results):
    """Calculate extraction and accuracy rates by subject"""
    subject_stats = defaultdict(lambda: {'total': 0, 'extracted': 0, 'correct': 0})
    
    for result in results:
        subject = result['subject']
        subject_stats[subject]['total'] += 1
        if result['has_extraction']:
            subject_stats[subject]['extracted'] += 1
        if result['is_correct']:
            subject_stats[subject]['correct'] += 1
    
    rates = {}
    for subject, stats in subject_stats.items():
        extraction_rate = stats['extracted'] / stats['total'] if stats['total'] > 0 else 0
        accuracy_rate = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        rates[subject] = {
            'extraction_rate': extraction_rate,
            'accuracy_rate': accuracy_rate,
            'total_questions': stats['total']
        }
    
    return rates

def order_subjects_custom(all_subjects_set):
    """Order subjects according to specified order"""
    desired_order = ['math', 'physics', 'chemistry', 'bio', 'geography', 'driving test', 'IQ Test']
    subject_mapping = {}
    for subject in all_subjects_set:
        subject_lower = subject.lower()
        for desired in desired_order:
            if desired.lower() in subject_lower or subject_lower in desired.lower():
                subject_mapping[subject] = desired
                break
        if subject not in subject_mapping:
            subject_mapping[subject] = subject
    
    ordered_subjects = []
    for desired in desired_order:
        for original_subject in all_subjects_set:
            if subject_mapping.get(original_subject) == desired and original_subject not in ordered_subjects:
                ordered_subjects.append(original_subject)
    
    for subject in sorted(all_subjects_set):
        if subject not in ordered_subjects:
            ordered_subjects.append(subject)
    
    return ordered_subjects

def main():

    file_configs = [
       
        ('results/c4ai-aya-vision-8b/full_vqa_vn.json', 'c4ai-aya-vision-8b'),
        ('results/c4ai-aya-vision-32b/full_vqa_vn.json', 'c4ai-aya-vision-32b'),
        ('results/claude-sonnet-4-20250514/full_vqa_vn.json', 'claude-sonnet-4-20250514'),
        ('results/gemini-2.5-flash/full_vqa_vn.json', 'gemini-2.5-flash'),
        ('results/gemma-3-4b-it/full_vqa_vn.json', 'gemma-3-4b-it'),
        ('results/gemma-3-27b-it/full_vqa_vn.json', 'gemma-3-27b-it'),
        ('results/gpt-4.1-2025-04-14/full_vqa_vn.json', 'gpt-4.1-2025-04-14'),
        ('results/o3-2025-04-16/full_vqa_vn_20250703_172822.json', 'o3-2025-04-16'),
        
        ('results/qwen/qwen2.5-vl-32b-instruct/full_vqa_vn.json', 'qwen/qwen2.5-vl-32b-instruct'),
        ('results/qwen/qwen2.5-vl-72b-instruct/full_vqa_vn.json', 'qwen/qwen2.5-vl-72b-instruct'),
        ('results/meta-llama/llama-4-maverick/full_vqa_vn.json', 'llama-4-maverick'),
        ('results/meta-llama/llama-4-scout/full_vqa_vn.json', 'llama-4-scout'),
        ('results/mistralai/mistral-medium-3/full_vqa_vn.json', 'mistral-medium-3'),
        ('results/mistralai/mistral-small-3.2-24b-instruct/full_vqa_vn.json', 'mistral-small-3.2-24b-instruct'),
    ]
    
    all_model_results = {}
    all_subjects = set()
    
    print("Processing models...")
    for file_path, model_name in file_configs:
        print(f"Processing {model_name}...")
        results = load_and_process_json(file_path, model_name)
        if results:
            rates = calculate_rates_by_subject(results)
            all_model_results[model_name] = rates
            all_subjects.update(rates.keys())
            print(f"  Found {len(results)} questions across {len(rates)} subjects")
        else:
            print(f"  No results found for {model_name}")
    
    if not all_model_results:
        print("No data found!")
        return
    
    all_subjects = order_subjects_custom(all_subjects)
    
    
    commercial_models = ['gemini-2.5-flash', 'claude-sonnet-4-20250514', 'gpt-4.1-2025-04-14', 'o3-2025-04-16']
    
    open_source_models = [
        'c4ai-aya-vision-8b', 'c4ai-aya-vision-32b', 
        'gemma-3-4b-it', 'gemma-3-27b-it',
        'qwen/qwen2.5-vl-32b-instruct', 'qwen/qwen2.5-vl-72b-instruct',
        'llama-4-scout', 'llama-4-maverick',
        'mistral-small-3.2-24b-instruct', 'mistral-medium-3'
    ]
    
    commercial_models = [m for m in commercial_models if m in all_model_results]
    open_source_models = [m for m in open_source_models if m in all_model_results]

    def create_table_data(model_list, rate_type):
        """Create table data for given models and rate type"""
        table_data = []
        for model in model_list:
            row = {'Model': model}
            model_rates = all_model_results[model]
            rates = []
            for subject in all_subjects:
                rate = model_rates.get(subject, {}).get(rate_type, 0)
                row[subject] = round(rate * 100, 2)
                rates.append(rate)
            row['Mean'] = round(np.mean(rates) * 100, 2) if rates else 0
            table_data.append(row)
        
        if table_data:
            mean_row = {'Model': 'Mean'}
            for subject in all_subjects:
                subject_rates = [all_model_results[model].get(subject, {}).get(rate_type, 0) for model in model_list if model in all_model_results]
                mean_row[subject] = round(np.mean(subject_rates) * 100, 2) if subject_rates else 0
            
            all_rates = [rate for model in model_list if model in all_model_results for rate in [all_model_results[model].get(s, {}).get(rate_type, 0) for s in all_subjects]]
            mean_row['Mean'] = round(np.mean(all_rates) * 100, 2) if all_rates else 0
            table_data.append(mean_row)
        return table_data

    # Create tables
    commercial_extraction_data = create_table_data(commercial_models, 'extraction_rate')
    opensource_extraction_data = create_table_data(open_source_models, 'extraction_rate')
    commercial_accuracy_data = create_table_data(commercial_models, 'accuracy_rate')
    opensource_accuracy_data = create_table_data(open_source_models, 'accuracy_rate')
    
    # Create DataFrames
    commercial_extraction_df = pd.DataFrame(commercial_extraction_data)
    opensource_extraction_df = pd.DataFrame(opensource_extraction_data)
    commercial_accuracy_df = pd.DataFrame(commercial_accuracy_data)
    opensource_accuracy_df = pd.DataFrame(opensource_accuracy_data)
    
    # Get columns for headers
    columns = []
    if not commercial_extraction_df.empty:
        columns = commercial_extraction_df.columns.tolist()
    elif not opensource_extraction_df.empty:
        columns = opensource_extraction_df.columns.tolist()
    
    if not columns:
        print("No columns to create tables. Exiting.")
        return
        
    # Create headers and empty rows
    headers = {
        'ce_header': pd.DataFrame([['EXTRACTION RATE - COMMERCIAL MODELS'] + [''] * (len(columns) - 1)], columns=columns),
        'oe_header': pd.DataFrame([['EXTRACTION RATE - OPEN SOURCE MODELS'] + [''] * (len(columns) - 1)], columns=columns),
        'ca_header': pd.DataFrame([['ACCURACY RATE - COMMERCIAL MODELS'] + [''] * (len(columns) - 1)], columns=columns),
        'oa_header': pd.DataFrame([['ACCURACY RATE - OPEN SOURCE MODELS'] + [''] * (len(columns) - 1)], columns=columns),
        'empty_row': pd.DataFrame([[''] * len(columns)], columns=columns)
    }

    # Combine all parts
    combined_parts = []
    if not commercial_extraction_df.empty:
        combined_parts.extend([headers['ce_header'], commercial_extraction_df, headers['empty_row']])
    if not opensource_extraction_df.empty:
        combined_parts.extend([headers['oe_header'], opensource_extraction_df, headers['empty_row']])
    if not commercial_accuracy_df.empty:
        combined_parts.extend([headers['ca_header'], commercial_accuracy_df, headers['empty_row']])
    if not opensource_accuracy_df.empty:
        combined_parts.extend([headers['oa_header'], opensource_accuracy_df])
    
    if combined_parts:
        combined_df = pd.concat(combined_parts, ignore_index=True)
        combined_df.to_csv('results/analysis_results1.csv', index=False)
        print(f"\nFile saved: analysis_results1.csv")
    
    # Print tables
    print("\nCommercial Models - Extraction Rate:"); print(commercial_extraction_df) if not commercial_extraction_df.empty else print("No data found")
    print("\nOpen Source Models - Extraction Rate:"); print(opensource_extraction_df) if not opensource_extraction_df.empty else print("No data found")
    print("\nCommercial Models - Accuracy Rate:"); print(commercial_accuracy_df) if not commercial_accuracy_df.empty else print("No data found")
    print("\nOpen Source Models - Accuracy Rate:"); print(opensource_accuracy_df) if not opensource_accuracy_df.empty else print("No data found")
    
    print("\nSummary:")
    print(f"Subjects found: {', '.join(all_subjects)}")
    print(f"Commercial models: {', '.join(commercial_models)}")
    print(f"Open source models: {', '.join(open_source_models)}")
    
    for model in commercial_models + open_source_models:
        if model in all_model_results:
            total_questions = sum(rates.get('total_questions', 0) for rates in all_model_results[model].values())
            print(f"{model}: {total_questions} total questions")

if __name__ == "__main__":
    main()