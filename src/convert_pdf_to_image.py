

import os
import re
import time
import json
import shutil
import argparse
from pdf2image import convert_from_path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import uuid
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

@dataclass
class PDFConversionResult:
    success: bool
    pdf_name: str
    subject: str
    output_folder: str
    page_count: int = 0
    conversion_time: float = 0.0
    error_message: str = ""

class PDFToImagesConverter:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @staticmethod
    def sanitize_filename(name: str) -> str:
        name = re.sub(r'\.pdf$', '', name, flags=re.IGNORECASE)
        return re.sub(r'[\\/*?:"<>|]', '_', name)
    
    def convert_single_pdf_to_images(self, pdf_path: str, subject_output_folder: str) -> PDFConversionResult:
        try:
            start_time = time.time()
            pdf_name = os.path.basename(pdf_path)
            exam_name_sanitized = self.sanitize_filename(pdf_name)
            subject_name = os.path.basename(subject_output_folder)
            
            pdf_images_folder = os.path.join(subject_output_folder, exam_name_sanitized, "pages")
            os.makedirs(pdf_images_folder, exist_ok=True)
            
            print(f"  🔄 Converting: {pdf_name}")
            
            poppler_path = None
            if os.name == 'nt':  
                poppler_path = self.config.get('poppler_bin_path')
            
            pil_images = convert_from_path(
                pdf_path,
                dpi=self.config['pdf_convert_dpi'],
                thread_count=min(4, os.cpu_count() or 1),
                poppler_path=poppler_path,
                fmt='png'
            )
            
            # Lưu từng trang
            page_info_list = []
            for i, pil_img in enumerate(pil_images):
                page_num = i + 1
                page_filename = f"page_{page_num:03d}.png"
                page_path = os.path.join(pdf_images_folder, page_filename)
                
                pil_img.save(page_path, "PNG")
                
                page_info_list.append({
                    'page_num': page_num,
                    'filename': page_filename,
                    'path': page_path,
                    'width': pil_img.width,
                    'height': pil_img.height
                })
            
            # Lưu metadata
            metadata = {
                'pdf_name': pdf_name,
                'pdf_path': pdf_path,
                'exam_name_sanitized': exam_name_sanitized,
                'subject': subject_name,
                'page_count': len(pil_images),
                'conversion_time': time.time() - start_time,
                'pages': page_info_list,
                'config_used': self.config
            }
            
            metadata_path = os.path.join(subject_output_folder, exam_name_sanitized, "metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            conversion_time = time.time() - start_time
            
            print(f"  ✅ {pdf_name}: {len(pil_images)} pages in {conversion_time:.2f}s")
            
            return PDFConversionResult(
                success=True,
                pdf_name=pdf_name,
                subject=subject_name,
                output_folder=os.path.join(subject_output_folder, exam_name_sanitized),
                page_count=len(pil_images),
                conversion_time=conversion_time
            )
            
        except Exception as e:
            error_msg = str(e)
            if "poppler" in error_msg.lower():
                error_msg += " (Hint: Install poppler-utils: sudo apt install poppler-utils)"
            
            print(f"  ❌ {os.path.basename(pdf_path)}: {error_msg}")
            
            return PDFConversionResult(
                success=False,
                pdf_name=os.path.basename(pdf_path),
                subject=os.path.basename(subject_output_folder),
                output_folder="",
                error_message=error_msg
            )

def convert_pdf_batch(batch_args: Tuple[List[Tuple], str]) -> List[PDFConversionResult]:
    pdf_args_list, batch_id = batch_args
    batch_results = []
    
    print(f"  📦 Starting batch {batch_id} with {len(pdf_args_list)} PDFs...")
    
    # Using ThreadPoolExecutor for I/O bound tasks
    max_workers = min(len(pdf_args_list), max(1, cpu_count() // 2))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pdf = {}
        
        for pdf_path, subject_output_folder, config in pdf_args_list:
            converter = PDFToImagesConverter(config)
            future = executor.submit(converter.convert_single_pdf_to_images, pdf_path, subject_output_folder)
            future_to_pdf[future] = os.path.basename(pdf_path)
        
        completed = 0
        for future in as_completed(future_to_pdf):
            pdf_name = future_to_pdf[future]
            try:
                result = future.result()
                batch_results.append(result)
                completed += 1
                
                status = "✓" if result.success else "✗"
                print(f"    {status} Batch {batch_id} ({completed}/{len(pdf_args_list)}): {result.pdf_name}")
                
            except Exception as exc:
                batch_results.append(PDFConversionResult(
                    success=False,
                    pdf_name=pdf_name,
                    subject="Unknown",
                    output_folder="",
                    error_message=str(exc)
                ))
                completed += 1
                print(f"    ✗ Batch {batch_id} ({completed}/{len(pdf_args_list)}): {pdf_name} - Exception: {exc}")
    
    successful = sum(1 for r in batch_results if r.success)
    print(f"  📦 Completed batch {batch_id}: {successful}/{len(batch_results)} successful")
    
    return batch_results

def process_subject_pdfs(subject_folder_name: str, input_base: str, output_base: str, 
                        config: Dict[str, Any], batch_size: int = 50) -> List[PDFConversionResult]:
    
    subject_input_path = os.path.join(input_base, subject_folder_name)
    sanitized_subject_name = PDFToImagesConverter.sanitize_filename(subject_folder_name)
    subject_output_path = os.path.join(output_base, sanitized_subject_name)
    os.makedirs(subject_output_path, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"📚 PROCESSING SUBJECT: {subject_folder_name}")
    print(f"  Input: {subject_input_path}")
    print(f"  Output: {subject_output_path}")
    
    # Tìm tất cả PDF files
    pdf_files = [f for f in os.listdir(subject_input_path) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"  ⚠️  No PDF files found in {subject_folder_name}")
        return []
    
    print(f"  📄 Found {len(pdf_files)} PDF files")
    all_pdf_args = []
    for pdf_filename in pdf_files:
        pdf_path = os.path.join(subject_input_path, pdf_filename)
        all_pdf_args.append((pdf_path, subject_output_path, config))
    
    batches = []
    for i in range(0, len(all_pdf_args), batch_size):
        batch = all_pdf_args[i:i + batch_size]
        batch_id = f"B{i//batch_size + 1}"
        batches.append((batch, batch_id))
    
    print(f"  📦 Created {len(batches)} batches (batch_size={batch_size})")
    
    all_results = []
    max_batch_workers = min(len(batches), max(1, cpu_count() // 3))
    
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=max_batch_workers) as executor:
        future_to_batch = {
            executor.submit(convert_pdf_batch, batch_args): batch_args[1]
            for batch_args in batches
        }
        
        completed_batches = 0
        for future in as_completed(future_to_batch):
            batch_id = future_to_batch[future]
            try:
                batch_results = future.result()
                all_results.extend(batch_results)
                completed_batches += 1
                
                successful_in_batch = sum(1 for r in batch_results if r.success)
                print(f"  🎯 Progress: {completed_batches}/{len(batches)} batches - "
                      f"{successful_in_batch}/{len(batch_results)} PDFs successful in {batch_id}")
                
            except Exception as exc:
                print(f"  ❌ Exception in batch {batch_id}: {exc}")
                # Tạo failed results cho batch này
                for batch_args, bid in batches:
                    if bid == batch_id:
                        for pdf_path, subject_output_path, _ in batch_args:
                            all_results.append(PDFConversionResult(
                                success=False,
                                pdf_name=os.path.basename(pdf_path),
                                subject=sanitized_subject_name,
                                output_folder="",
                                error_message=f"Batch processing failed: {str(exc)}"
                            ))
                        break
    
    processing_time = time.time() - start_time
    successful_results = [r for r in all_results if r.success]
    total_pages = sum(r.page_count for r in successful_results)
    
    print(f"  🏁 Subject '{subject_folder_name}' completed:")
    print(f"    - Time: {processing_time:.2f}s")
    print(f"    - Success: {len(successful_results)}/{len(pdf_files)} PDFs")
    print(f"    - Total pages: {total_pages}")
    print(f"    - Speed: {len(pdf_files)/processing_time:.1f} PDFs/s")
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description="Convert PDFs to images in parallel")
    parser.add_argument('--input_base', type=str, default='./de_thi_chuan/DeThi12',
                        help="Base folder containing subject subfolders with PDF files")
    parser.add_argument('--output_base', type=str, default='./output_images',
                        help="Base folder to save converted images")
    parser.add_argument('--poppler_path', type=str, default=None,
                        help="Path to Poppler bin directory (Windows only)")
    parser.add_argument('--dpi', type=int, default=300,
                        help="DPI for PDF to image conversion")
    parser.add_argument('--batch_size', type=int, default=50,
                        help="Number of PDFs to process in parallel within each batch")
    parser.add_argument('--max_subject_workers', type=int, default=None,
                        help="Max subjects to process in parallel")
    
    args = parser.parse_args()
    
    if args.max_subject_workers is None:
        args.max_subject_workers = min(2, max(1, cpu_count() // 6))
    
    config = {
        'poppler_bin_path': args.poppler_path,
        'pdf_convert_dpi': args.dpi,
    }
    
    print(f"🚀 PDF TO IMAGES CONVERTER")
    print(f"📁 Input: {os.path.abspath(args.input_base)}")
    print(f"📂 Output: {os.path.abspath(args.output_base)}")
    print(f"📦 Batch Size: {args.batch_size}")
    print(f"👥 Subject Workers: {args.max_subject_workers}")
    print(f"🖥️  CPU Count: {cpu_count()}")
    
    if not os.path.isdir(args.input_base):
        print(f"❌ Input directory not found: {args.input_base}")
        return
    
    os.makedirs(args.output_base, exist_ok=True)
    
    subject_folders = [d for d in os.listdir(args.input_base) 
                      if os.path.isdir(os.path.join(args.input_base, d)) and not d.startswith('_')]
    
    if not subject_folders:
        print(f"❌ No subject folders found in {args.input_base}")
        return
    
    print(f"📚 Found {len(subject_folders)} subjects: {subject_folders}")
    
    # Đếm tổng số PDF
    total_pdfs = 0
    for subject in subject_folders:
        subject_path = os.path.join(args.input_base, subject)
        pdf_count = len([f for f in os.listdir(subject_path) if f.lower().endswith('.pdf')])
        total_pdfs += pdf_count
    print(f"📄 Total PDFs to convert: {total_pdfs}")
    
    start_time = time.time()
    all_results = []
    
    if args.max_subject_workers == 1:
        # Sequential processing of subjects
        for subject_folder in subject_folders:
            results = process_subject_pdfs(subject_folder, args.input_base, args.output_base, config, args.batch_size)
            all_results.extend(results)
    else:
        # Parallel processing of subjects
        with ProcessPoolExecutor(max_workers=args.max_subject_workers) as executor:
            future_to_subject = {
                executor.submit(process_subject_pdfs, subject_folder, args.input_base, args.output_base, config, args.batch_size): subject_folder
                for subject_folder in subject_folders
            }
            
            completed_subjects = 0
            for future in as_completed(future_to_subject):
                subject_name = future_to_subject[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    completed_subjects += 1
                    
                    successful = sum(1 for r in results if r.success)
                    print(f"🎉 Subject progress: {completed_subjects}/{len(subject_folders)} - "
                          f"{successful}/{len(results)} PDFs successful in {subject_name}")
                    
                except Exception as exc:
                    print(f"❌ Exception processing subject {subject_name}: {exc}")
    
    total_time = time.time() - start_time
    successful_results = [r for r in all_results if r.success]
    failed_results = [r for r in all_results if not r.success]
    total_pages = sum(r.page_count for r in successful_results)
    
    print(f"\n{'='*80}")
    print(f"🏆 PDF TO IMAGES CONVERSION COMPLETE!")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    print(f"⚡ Speed: {len(successful_results)/total_time:.1f} PDFs/second")
    print(f"✅ Successful: {len(successful_results)} PDFs")
    print(f"❌ Failed: {len(failed_results)} PDFs")
    print(f"📄 Total pages converted: {total_pages}")
    print(f"📊 Success rate: {len(successful_results)/len(all_results)*100:.1f}%")
    
    # Stats by subject
    print(f"\n📈 Results by Subject:")
    subject_stats = {}
    for result in all_results:
        if result.subject not in subject_stats:
            subject_stats[result.subject] = {'success': 0, 'failed': 0, 'pages': 0}
        
        if result.success:
            subject_stats[result.subject]['success'] += 1
            subject_stats[result.subject]['pages'] += result.page_count
        else:
            subject_stats[result.subject]['failed'] += 1
    
    for subject, stats in subject_stats.items():
        total = stats['success'] + stats['failed']
        success_rate = (stats['success'] / total * 100) if total > 0 else 0
        print(f"  📚 {subject}: {stats['success']}/{total} PDFs ({success_rate:.0f}%) → {stats['pages']} pages")
    
    if failed_results:
        print(f"\n❌ Failed conversions:")
        for result in failed_results[:10]:
            print(f"  - {result.subject}/{result.pdf_name}: {result.error_message}")
        if len(failed_results) > 10:
            print(f"  ... and {len(failed_results) - 10} more")
    
    print(f"\n✅ Images saved to: {os.path.abspath(args.output_base)}")
    print(f"📋 Ready for question extraction step!")

if __name__ == "__main__":
    main()