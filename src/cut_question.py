
import os
import re
import cv2
import json
import time
import sys
import argparse
import numpy as np
import pytesseract
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

@dataclass
class OCRMarker:
    page_idx: int
    y: int
    avg_height: float
    marker_page_num_1_based: int
    line_text: str

@dataclass
class QuestionMarker(OCRMarker):
    """Question marker"""
    q_num: str

@dataclass
class GroupDirective(OCRMarker):
    """Group directive (e.g., "Câu 1-5")"""
    start_num: int
    end_num: int

@dataclass
class CropItem:

    item_type: str  # 'single' or 'group'
    start_page_idx: int
    start_y: int
    end_page_idx: int
    end_y: int
    filename_base: str
    marker_page_num: int

@dataclass
class ExtractionResult:
    success: bool
    exam_name: str
    subject: str
    images_count: int = 0
    text_count: int = 0
    processing_time: float = 0.0
    error_message: str = ""

class ImageQuestionExtractor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._setup_tesseract()
        self._compile_regex_patterns()
        self._setup_subject_margins()
    
    def _setup_tesseract(self):
        try:
            if self.config.get('tesseract_path') and self.config['tesseract_path'] != 'tesseract':
                pytesseract.pytesseract.tesseract_cmd = self.config['tesseract_path']
            else:
                pytesseract.pytesseract.tesseract_cmd = 'tesseract'
            
            t_version = pytesseract.get_tesseract_version()
        except Exception as e:
            raise RuntimeError(f"Tesseract Error: {e}")
    
    def _compile_regex_patterns(self):
        self.question_num_regex = re.compile(self.config['regex_patterns']['question_num'], re.IGNORECASE)
        self.group_directive_regex = re.compile(self.config['regex_patterns']['group_directive'], re.IGNORECASE)
        self.simple_group_regex = re.compile(self.config['regex_patterns']['simple_group'], re.IGNORECASE)
        self.separator_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.config['separator_keywords']]
    
    def _setup_subject_margins(self):
        self.subject_top_margins = self.config.get('subject_top_margins', {})
        self.default_top_margin = self.config.get('default_top_margin', 10)
    
    def _get_top_margin_for_subject(self, subject_name: str) -> int:
        """Get margin cho subject"""
        subject_lower = subject_name.lower()
        
        for subject_pattern, margin in self.subject_top_margins.items():
            if subject_pattern.lower() in subject_lower:
                return margin
        
        return self.default_top_margin
    
    def _ocr_page_to_lines(self, page_path: str, page_num: int) -> Dict[Tuple, Dict]:
        """OCR một page thành lines"""
        try:
            with Image.open(page_path) as pil_image:
                ocr_data = pytesseract.image_to_data(
                    pil_image,
                    lang=self.config['tesseract_lang'],
                    config=self.config.get('tesseract_ocr_config', '--oem 3 --psm 3'),
                    output_type=pytesseract.Output.DICT
                )
        except Exception as ocr_err:
            return {}

        lines = {}
        for i in range(len(ocr_data['text'])):
            word_text = ocr_data['text'][i].strip()
            conf = int(float(ocr_data['conf'][i]))
            if not word_text or conf < self.config.get('ocr_word_min_confidence', 20):
                continue
            
            key = (ocr_data['block_num'][i], ocr_data['par_num'][i], ocr_data['line_num'][i])
            if key not in lines:
                lines[key] = {
                    'words': [], 'tops': [], 'confs': [],
                    'min_top': ocr_data['top'][i], 'heights': []
                }
            
            line_entry = lines[key]
            line_entry['words'].append(word_text)
            line_entry['tops'].append(ocr_data['top'][i])
            line_entry['confs'].append(conf)
            line_entry['min_top'] = min(line_entry['min_top'], ocr_data['top'][i])
            line_entry['heights'].append(ocr_data['height'][i])
        
        return lines
    
    def find_all_markers_and_directives(self, page_objects: List[Dict]) -> Tuple[List[QuestionMarker], List[GroupDirective]]:
        """Tìm tất cả markers và directives"""
        all_q_markers = []
        all_g_directives = []
        
        for page_obj in page_objects:
            lines = self._ocr_page_to_lines(page_obj['path'], page_obj['page_num'])
            
            for line_data in sorted(lines.values(), key=lambda item: item['min_top']):
                full_line_text = " ".join(line_data['words'])
                avg_conf = sum(line_data['confs']) / len(line_data['confs']) if line_data['confs'] else 0
                line_y = line_data['min_top']
                avg_height = sum(line_data['heights']) / len(line_data['heights']) if line_data['heights'] else 0

                # Check for Group Directives
                group_match = self.group_directive_regex.search(full_line_text)
                if not group_match:
                    group_match = self.simple_group_regex.search(full_line_text)
                
                if group_match and avg_conf > self.config.get('ocr_group_min_avg_confidence', 40):
                    try:
                        start_num, end_num = int(group_match.group(1)), int(group_match.group(2))
                        if start_num <= end_num:
                            all_g_directives.append(GroupDirective(
                                start_num=start_num, end_num=end_num,
                                page_idx=page_obj['page_num'] - 1, y=line_y, avg_height=avg_height,
                                marker_page_num_1_based=page_obj['page_num'], line_text=full_line_text.strip()
                            ))
                            continue
                    except ValueError:
                        pass

                # Check for Question Markers
                q_match = self.question_num_regex.match(full_line_text)
                if q_match and avg_conf > self.config.get('ocr_question_min_avg_confidence', 30):
                    all_q_markers.append(QuestionMarker(
                        q_num=q_match.group(1), page_idx=page_obj['page_num'] - 1,
                        y=line_y, avg_height=avg_height,
                        marker_page_num_1_based=page_obj['page_num'], line_text=full_line_text.strip()
                    ))

        all_q_markers.sort(key=lambda m: (m.page_idx, m.y))
        all_g_directives.sort(key=lambda d: (d.page_idx, d.y))
        
        return all_q_markers, all_g_directives
    
    def detect_image_in_question_region(self, question_image_np: np.ndarray) -> bool:
     
        if len(question_image_np.shape) == 3:
            gray = cv2.cvtColor(question_image_np, cv2.COLOR_BGR2GRAY)
        else:
            gray = question_image_np.copy()

        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = self.config['cv_min_image_area']
        text_h_range = self.config['cv_text_height_range']
        text_wh_ratio = self.config['cv_text_width_height_ratio']
        min_img_dim = self.config.get('cv_min_img_dim', 30)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            
            is_likely_text_shape = (
                (text_h_range[0] <= h <= text_h_range[1] and (w / (h + 1e-6)) < text_wh_ratio) or \
                (text_h_range[0] <= w <= text_h_range[1] and (h / (w + 1e-6)) < text_wh_ratio)
            )
            is_large_enough_for_image = (w > min_img_dim and h > min_img_dim)

            if area > min_area and not is_likely_text_shape and is_large_enough_for_image:
                return True
        return False
    
    def _find_early_break_separator(self, current_start_page_idx: int, current_start_y: int,
                                   default_end_page_idx: int, default_end_y: int,
                                   page_objects: List[Dict]) -> Optional[Tuple[int, int]]:
    
        min_offset = self.config['separator_min_offset_from_marker']
        max_search = self.config['separator_max_search_ahead_pixels']

        for page_idx in range(current_start_page_idx, default_end_page_idx + 1):
            page_obj = next((p for p in page_objects if p['page_num'] - 1 == page_idx), None)
            if not page_obj: 
                continue

            scan_y_start = min_offset
            if page_idx == current_start_page_idx:
                scan_y_start += current_start_y
            
            scan_y_end = page_obj['height']
            if page_idx == default_end_page_idx:
                scan_y_end = default_end_y
            if page_idx == current_start_page_idx:
                scan_y_end = min(scan_y_end, current_start_y + max_search)

            if scan_y_start >= scan_y_end:
                continue

            page_img = cv2.imread(page_obj['path'])
            if page_img is None:
                continue
                
            region = page_img[scan_y_start:scan_y_end, :]
            if region.shape[0] <= 10 or region.shape[1] <= 50:
                continue

            try:
                ocr_data = pytesseract.image_to_data(
                    Image.fromarray(region), 
                    lang=self.config['tesseract_lang'],
                    config=self.config.get('tesseract_separator_config', '--oem 3 --psm 6'),
                    output_type=pytesseract.Output.DICT
                )
                
                sep_lines = {}
                for k in range(len(ocr_data['text'])):
                    word = ocr_data['text'][k].strip().lower()
                    conf = int(float(ocr_data['conf'][k]))
                    if word and conf > self.config.get('ocr_separator_min_confidence', 30):
                        key = (ocr_data['block_num'][k], ocr_data['par_num'][k], ocr_data['line_num'][k])
                        if key not in sep_lines:
                            sep_lines[key] = {'words':[], 'top': ocr_data['top'][k]}
                        sep_lines[key]['words'].append(word)
                
                for line_data in sorted(sep_lines.values(), key=lambda x: x['top']):
                    full_line = " ".join(line_data['words'])
                    for pattern in self.separator_patterns:
                        if pattern.search(full_line):
                            separator_y = scan_y_start + line_data['top']
                            return page_idx, max(0, separator_y - self.config['bottom_margin'])
                            
            except Exception:
                pass
        
        return None
    
    def _determine_crop_boundaries(self, item_idx: int, all_markers: List[OCRMarker],
                                   page_objects: List[Dict]) -> Tuple[int, int]:
        
        current_marker = all_markers[item_idx]
        
        next_idx = item_idx + 1
        if next_idx < len(all_markers):
            next_marker = all_markers[next_idx]
            default_end_page_idx = next_marker.page_idx
            default_end_y = max(0, next_marker.y - self.config['bottom_margin'])
        else:
            default_end_page_idx = current_marker.page_idx
            page_obj = next(p for p in page_objects if p['page_num'] - 1 == default_end_page_idx)
            default_end_y = page_obj['height'] - self.config.get('bottom_margin_page_end', 5)

        final_end_page_idx, final_end_y = default_end_page_idx, default_end_y

        early_break = self._find_early_break_separator(
            current_marker.page_idx, current_marker.y,
            default_end_page_idx, default_end_y, page_objects
        )
        if early_break:
            final_end_page_idx, final_end_y = early_break

        return final_end_page_idx, final_end_y
    
    def _create_crop_items(self, all_q_markers: List[QuestionMarker], all_g_directives: List[GroupDirective],
                          page_objects: List[Dict], subject_name: str) -> List[CropItem]:
      
        items_to_crop = []
        processed_q_indices = set()
        
        subject_top_margin = self._get_top_margin_for_subject(subject_name)

        # Process Group Directives
        for directive in all_g_directives:
            group_q_indices = []
            for q_idx, q_marker in enumerate(all_q_markers):
                if q_idx in processed_q_indices:
                    continue
                try:
                    q_num_int = int(q_marker.q_num)
                    if directive.start_num <= q_num_int <= directive.end_num:
                        if (q_marker.page_idx > directive.page_idx or
                            (q_marker.page_idx == directive.page_idx and q_marker.y >= directive.y)):
                            group_q_indices.append(q_idx)
                except ValueError:
                    continue
            
            if not group_q_indices:
                continue

            first_q_marker = all_q_markers[group_q_indices[0]]
            if int(first_q_marker.q_num) != directive.start_num:
                continue

            for q_idx in group_q_indices:
                processed_q_indices.add(q_idx)
            
            last_q_idx = group_q_indices[-1]
            
            start_page_idx = directive.page_idx
            start_y = max(0, directive.y - subject_top_margin)
            
            end_page_idx, end_y = self._determine_crop_boundaries(
                last_q_idx, all_q_markers, page_objects
            )
            
            items_to_crop.append(CropItem(
                item_type='group',
                start_page_idx=start_page_idx,
                start_y=start_y,
                end_page_idx=end_page_idx,
                end_y=end_y,
                filename_base=f"question_{directive.start_num}-{directive.end_num}",
                marker_page_num=directive.marker_page_num_1_based
            ))

        # Process Single Questions
        for q_idx, q_marker in enumerate(all_q_markers):
            if q_idx in processed_q_indices:
                continue
            
            start_page_idx = q_marker.page_idx
            start_y = max(0, q_marker.y - subject_top_margin)
            
            end_page_idx, end_y = self._determine_crop_boundaries(
                q_idx, all_q_markers, page_objects
            )

            # Check for group directive interruption
            next_q_idx = q_idx + 1
            if next_q_idx < len(all_q_markers):
                next_q_marker = all_q_markers[next_q_idx]
                for directive in all_g_directives:
                    try:
                        if directive.start_num == int(next_q_marker.q_num):
                            if (directive.page_idx < end_page_idx or \
                               (directive.page_idx == end_page_idx and directive.y < end_y)):
                                if (directive.page_idx > q_marker.page_idx or \
                                   (directive.page_idx == q_marker.page_idx and directive.y > q_marker.y + q_marker.avg_height)):
                                    end_page_idx = directive.page_idx
                                    end_y = max(0, directive.y - self.config['bottom_margin'])
                                    break
                    except ValueError:
                        continue

            items_to_crop.append(CropItem(
                item_type='single',
                start_page_idx=start_page_idx,
                start_y=start_y,
                end_page_idx=end_page_idx,
                end_y=end_y,
                filename_base=f"question_{q_marker.q_num}",
                marker_page_num=q_marker.marker_page_num_1_based
            ))

        return sorted(items_to_crop, key=lambda x: (x.start_page_idx, x.start_y))
    
    def _stitch_and_save_crop_item(self, exam_name: str, item: CropItem, page_objects: List[Dict],
                                   image_output_folder: str, text_output_folder: str) -> Tuple[bool, bool]:
        
        if item.end_page_idx < item.start_page_idx or \
           (item.end_page_idx == item.start_page_idx and item.end_y <= item.start_y):
            return False, False
            
        image_parts = []
        for page_idx in range(item.start_page_idx, item.end_page_idx + 1):
            page_obj = next((p for p in page_objects if p['page_num'] - 1 == page_idx), None)
            if not page_obj:
                continue
            
            img = cv2.imread(page_obj['path'])
            if img is None:
                continue
            
            page_height = img.shape[0]
            slice_y0 = item.start_y if page_idx == item.start_page_idx else 0
            slice_y1 = item.end_y if page_idx == item.end_page_idx else page_height
            
            slice_y0 = max(0, min(slice_y0, page_height))
            slice_y1 = max(0, min(slice_y1, page_height))

            if slice_y0 < slice_y1:
                cropped_part = img[slice_y0:slice_y1, :]
                if cropped_part.shape[0] > 0 and cropped_part.shape[1] > 0:
                    image_parts.append(cropped_part)
        
        if not image_parts:
            return False, False

        try:
            if len(image_parts) > 1:
                final_image = cv2.vconcat(image_parts)
            else:
                final_image = image_parts[0]
        except cv2.error:
            # Try to fix width mismatch
            if not image_parts:
                return False, False
            target_width = image_parts[0].shape[1]
            fixed_parts = []
            for part in image_parts:
                if part.shape[1] != target_width:
                    try:
                        part = cv2.resize(part, (target_width, part.shape[0]))
                    except cv2.error:
                        return False, False
                fixed_parts.append(part)
            try:
                final_image = cv2.vconcat(fixed_parts) if len(fixed_parts) > 1 else fixed_parts[0]
            except cv2.error:
                return False, False

        # Detect if has image
        has_image = self.detect_image_in_question_region(final_image)
        target_folder = image_output_folder if has_image else text_output_folder
        
        suffix = "image" if has_image else "text_only"
        output_filename_base = f"{exam_name}_{item.filename_base}_page_{item.marker_page_num}_{suffix}"
        
        output_path = os.path.join(target_folder, f"{output_filename_base}.png")
        counter = 1
        while os.path.exists(output_path):
            output_path = os.path.join(target_folder, f"{output_filename_base}_v{counter}.png")
            counter += 1
        
        try:
            cv2.imwrite(output_path, final_image)
            return True, has_image
        except Exception:
            return False, False
    
    def extract_questions_from_images(self, exam_folder: str, output_base_folder: str) -> ExtractionResult:

        try:
            start_time = time.time()
            
            # Đọc metadata
            metadata_path = os.path.join(exam_folder, "metadata.json")
            if not os.path.exists(metadata_path):
                return ExtractionResult(
                    success=False,
                    exam_name=os.path.basename(exam_folder),
                    subject="Unknown",
                    error_message="Metadata file not found"
                )
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            exam_name = metadata['exam_name_sanitized']
            subject_name = metadata['subject']
            
            print(f"  🔍 Processing: {exam_name}")
            sys.stdout.flush()
            
            # Tạo page objects từ metadata
            pages_folder = os.path.join(exam_folder, "pages")
            page_objects = []
            for page_info in metadata['pages']:
                page_objects.append({
                    'page_num': page_info['page_num'],
                    'path': os.path.join(pages_folder, page_info['filename']),
                    'width': page_info['width'],
                    'height': page_info['height']
                })
            all_q_markers, all_g_directives = self.find_all_markers_and_directives(page_objects)
            
            if not all_q_markers:
                return ExtractionResult(
                    success=False,
                    exam_name=exam_name,
                    subject=subject_name,
                    error_message="No question markers found"
                )
            
        
            exam_output_folder = os.path.join(output_base_folder, subject_name, exam_name)
            image_folder = os.path.join(exam_output_folder, "question_with_images")
            text_folder = os.path.join(exam_output_folder, "question_text_only")
            os.makedirs(image_folder, exist_ok=True)
            os.makedirs(text_folder, exist_ok=True)
            
            # Tạo crop items
            crop_items = self._create_crop_items(all_q_markers, all_g_directives, page_objects, subject_name)
            
            if not crop_items:
                return ExtractionResult(
                    success=False,
                    exam_name=exam_name,
                    subject=subject_name,
                    error_message="No crop items generated"
                )
            
            # Extract và save với simple progress tracking
            images_count = 0
            text_count = 0
            total_items = len(crop_items)
            
            print(f"    📝 Found {total_items} questions to extract")
            sys.stdout.flush()
            
            for i, item in enumerate(crop_items, 1):
                was_saved, has_image = self._stitch_and_save_crop_item(
                    exam_name, item, page_objects, image_folder, text_folder
                )
                if was_saved:
                    if has_image:
                        images_count += 1
                    else:
                        text_count += 1
                
                # Print progress every 10 items or at the end
                if i % 10 == 0 or i == total_items:
                    print(f"    📊 Progress: {i}/{total_items} ({i/total_items*100:.0f}%) - {images_count}img, {text_count}txt")
                    sys.stdout.flush()
            
            processing_time = time.time() - start_time
            
            print(f"  ✅ {exam_name}: {images_count} images, {text_count} text ({processing_time:.2f}s)")
            sys.stdout.flush()
            
            return ExtractionResult(
                success=True,
                exam_name=exam_name,
                subject=subject_name,
                images_count=images_count,
                text_count=text_count,
                processing_time=processing_time
            )
            
        except Exception as e:
            return ExtractionResult(
                success=False,
                exam_name=os.path.basename(exam_folder),
                subject="Unknown",
                error_message=str(e)
            )


def extract_batch_of_exams(batch_args: Tuple[List[Tuple], str, int]) -> List[ExtractionResult]:
   
    exam_args_list, batch_id, batch_size = batch_args
    batch_results = []
    
    max_workers = min(len(exam_args_list), batch_size)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_exam = {}
        
        for exam_folder, output_base_folder, config in exam_args_list:
            extractor = ImageQuestionExtractor(config)
            future = executor.submit(extractor.extract_questions_from_images, exam_folder, output_base_folder)
            future_to_exam[future] = os.path.basename(exam_folder)
        
        completed = 0
        for future in as_completed(future_to_exam):
            exam_name = future_to_exam[future]
            try:
                result = future.result()
                batch_results.append(result)
                completed += 1
                
                status = "✓" if result.success else "✗"
                if result.success:
                    print(f"    {status} Batch {batch_id} ({completed}/{len(exam_args_list)}): "
                          f"{result.exam_name} ({result.images_count}img, {result.text_count}txt)")
                else:
                    print(f"    {status} Batch {batch_id} ({completed}/{len(exam_args_list)}): "
                          f"{result.exam_name} - {result.error_message}")
                
            except Exception as exc:
                batch_results.append(ExtractionResult(
                    success=False,
                    exam_name=exam_name,
                    subject="Unknown",
                    error_message=str(exc)
                ))
                completed += 1
                print(f"    ✗ Batch {batch_id} ({completed}/{len(exam_args_list)}): "
                      f"{exam_name} - Exception: {exc}")
    
    successful = sum(1 for r in batch_results if r.success)
    print(f"  📦 Completed extraction batch {batch_id}: {successful}/{len(batch_results)} successful")
    
    return batch_results

def process_subject_extractions(subject_folder_name: str, images_base: str, output_base: str,
                               config: Dict[str, Any], batch_size: int) -> List[ExtractionResult]:

    
    subject_images_path = os.path.join(images_base, subject_folder_name)
    subject_output_path = os.path.join(output_base, subject_folder_name)
    os.makedirs(subject_output_path, exist_ok=True)

    print(f"📚 EXTRACTING SUBJECT: {subject_folder_name}")
    print(f"  Images: {subject_images_path}")
    print(f"  Output: {subject_output_path}")
    sys.stdout.flush()
    
    if not os.path.exists(subject_images_path):
        print(f"  ⚠️  Subject images folder not found: {subject_folder_name}")
        return []
    
    exam_folders = [d for d in os.listdir(subject_images_path) 
                   if os.path.isdir(os.path.join(subject_images_path, d))]
    
    if not exam_folders:
        print(f"  ⚠️  No exam folders found in {subject_folder_name}")
        return []
    
    print(f"  📚 Found {len(exam_folders)} exam folders")
    sys.stdout.flush()
    
    all_exam_args = []
    for exam_folder_name in exam_folders:
        exam_folder_path = os.path.join(subject_images_path, exam_folder_name)
        all_exam_args.append((exam_folder_path, output_base, config))
    
    batches = []
    for i in range(0, len(all_exam_args), batch_size):
        batch = all_exam_args[i:i + batch_size]
        batch_id = f"B{i//batch_size + 1}"

        batches.append((batch, batch_id, batch_size))
    
    print(f"  📦 Created {len(batches)} extraction batches of size {batch_size}")
    sys.stdout.flush()
    all_results = []
    start_time = time.time()
    total_batches = len(batches)
    
    for i, batch_args in enumerate(batches, 1):
        batch_id = batch_args[1]
        print(f"\n--- Starting Batch {i}/{total_batches} ({batch_id}) ---")
        try:
            batch_results = extract_batch_of_exams(batch_args)
            all_results.extend(batch_results)
        except Exception as exc:
            print(f"  ❌ Unhandled exception in batch {batch_id}: {exc}")

    processing_time = time.time() - start_time
    successful_results = [r for r in all_results if r.success]
    total_images = sum(r.images_count for r in successful_results)
    total_text = sum(r.text_count for r in successful_results)
    
    print(f"\n  🏁 Subject '{subject_folder_name}' extraction completed:")
    print(f"    - Time: {processing_time:.2f}s")
    print(f"    - Success: {len(successful_results)}/{len(exam_folders)} exams")
    print(f"    - Total images: {total_images}")
    print(f"    - Total text: {total_text}")
    if processing_time > 0:
      print(f"    - Speed: {len(exam_folders)/processing_time:.1f} exams/s")
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description="Extract questions from converted images")
    parser.add_argument('--images_base', type=str, default='./TEST/test_top',
                        help="Base folder containing converted images")
    parser.add_argument('--output_base', type=str, default='./output_questions2',
                        help="Base folder to save extracted questions")
    parser.add_argument('--tesseract_path', type=str, default='tesseract',
                        help="Path to Tesseract executable")
    ### MODIFIED ###
    parser.add_argument('--batch_size', type=int, default=10,
                        help="Number of exams to process in parallel at one time.")

    parser.add_argument('--max_subject_workers', type=int, default=1,
                        help="Max subjects to process in parallel (set to 1 for controlled batching)")
    ### END MODIFIED ###
    
    args = parser.parse_args()


    if args.max_subject_workers != 1:
        print("⚠️ Warning: For precise control with batch_size, it's recommended to set --max_subject_workers to 1.")
        print(f"Setting max_subject_workers to 1.")
        args.max_subject_workers = 1

    config = {
        'tesseract_path': args.tesseract_path,
        'tesseract_lang': 'vie',
        'tesseract_ocr_config': '--oem 3 --psm 3',
        'tesseract_separator_config': '--oem 3 --psm 11',
        'bottom_margin': 5,
        'bottom_margin_page_end': 3,
        'ocr_word_min_confidence': 20,
        'ocr_question_min_avg_confidence': 30,
        'ocr_group_min_avg_confidence': 40,
        'ocr_separator_min_confidence': 30,
        'cv_min_image_area': 3000,
        'cv_text_height_range': (7, 40),
        'cv_text_width_height_ratio': 18,
        'cv_min_img_dim': 35,
        'regex_patterns': {
            'question_num': r"^\s*(?:Câu|Question)\s+(\d+)(?:\s*\([^)]+\))?\s*[.:]?",
            'group_directive': r"(?:sử dụng thông tin.+?|dùng thông tin.+?)?(?:cho các câu|trả lời(?: các)? câu)\s*(\d+)\s*(?:[-–—đếntới,\.]\s*|và\s+)(\d+)",
            'simple_group': r"(?:các\s*)?câu\s*(\d+)\s*(?:[-–—]\s*|và\s+)(\d+)"
        },
        'separator_keywords': [
            "phần ii", "phần iii", "phần iv", "phần v", "chương", "mục", "hết",
            "lưu ý:", "chú ý:", "hết phần", "hết bài", "hết đề",
            "đáp án", "lời giải chi tiết", "thí sinh không được sử dụng tài liệu",
            "giám thị không giải thích gì thêm", r"trang \d+/\d+"
        ],
        'separator_min_offset_from_marker': 30,
        'separator_max_search_ahead_pixels': 350,
        'subject_top_margins': {
            'math': 33,
            'biology': 10,
            'physics': 10,
            'chemistry': 10,
            'geography': 10,
        },
        'default_top_margin': 15,
    }
    
    print(f"🚀 IMAGE QUESTION EXTRACTOR WITH PROGRESS TRACKING")
    print(f"📁 Images: {os.path.abspath(args.images_base)}")
    print(f"📂 Output: {os.path.abspath(args.output_base)}")
    print(f"📦 Parallel Exams (Batch Size): {args.batch_size}")
    print(f"🖥️  CPU Count: {cpu_count()}")
    print(f"{'='*80}")
    
    if not os.path.isdir(args.images_base):
        print(f"❌ Images directory not found: {args.images_base}")
        return
    
    os.makedirs(args.output_base, exist_ok=True)
    
    subject_folders = [d for d in os.listdir(args.images_base) 
                      if os.path.isdir(os.path.join(args.images_base, d)) and not d.startswith('_')]
    
    if not subject_folders:
        print(f"❌ No subject folders found in {args.images_base}")
        return
    
    total_exams = 0
    subject_exam_counts = {}
    for subject in subject_folders:
        subject_path = os.path.join(args.images_base, subject)
        exam_count = len([d for d in os.listdir(subject_path) 
                         if os.path.isdir(os.path.join(subject_path, d))])
        subject_exam_counts[subject] = exam_count
        total_exams += exam_count
    
    print(f"📚 Found {len(subject_folders)} subjects with {total_exams} total exams:")
    for subject, count in subject_exam_counts.items():
        print(f"  📖 {subject}: {count} exams")
    print()
    
    start_time = time.time()
    all_results = []
    ### MODIFIED ###

    for subject_folder in subject_folders:
        results = process_subject_extractions(subject_folder, args.images_base, args.output_base, config, args.batch_size)
        all_results.extend(results)
    ### END MODIFIED ###

    total_time = time.time() - start_time
    successful_results = [r for r in all_results if r.success]
    failed_results = [r for r in all_results if not r.success]
    total_images = sum(r.images_count for r in successful_results)
    total_text = sum(r.text_count for r in successful_results)
    
    print(f"\n{'='*80}")
    print(f"🏆 QUESTION EXTRACTION COMPLETE!")
    if total_time > 0 and len(all_results) > 0:
        print(f"⏱️  Total time: {total_time:.2f} seconds")
        print(f"⚡ Speed: {len(all_results)/total_time:.1f} exams/second")
        print(f"✅ Successful: {len(successful_results)} exams")
        print(f"❌ Failed: {len(failed_results)} exams")
        print(f"🖼️  Total questions with images: {total_images}")
        print(f"📝 Total text-only questions: {total_text}")
        print(f"📊 Success rate: {len(successful_results)/len(all_results)*100:.1f}%")
    else:
        print("No exams were processed.")

    # Stats by subject
    if successful_results or failed_results:
        print(f"\n📈 Results by Subject:")
        subject_stats = {}
        for result in all_results:
            subject = result.subject if result.subject != "Unknown" else os.path.basename(os.path.dirname(os.path.dirname(result.exam_name)))
            if subject not in subject_stats:
                subject_stats[subject] = {'success': 0, 'failed': 0, 'images': 0, 'text': 0}
            
            if result.success:
                subject_stats[subject]['success'] += 1
                subject_stats[subject]['images'] += result.images_count
                subject_stats[subject]['text'] += result.text_count
            else:
                subject_stats[subject]['failed'] += 1
        
        for subject, stats in sorted(subject_stats.items()):
            total = stats['success'] + stats['failed']
            success_rate = (stats['success'] / total * 100) if total > 0 else 0
            print(f"  📚 {subject}: {stats['success']}/{total} exams ({success_rate:.0f}%) → "
                  f"{stats['images']} images, {stats['text']} text")
    
    if failed_results:
        print(f"\n❌ Failed extractions:")
        for result in failed_results[:10]:
            print(f"  - {result.subject}/{result.exam_name}: {result.error_message}")
        if len(failed_results) > 10:
            print(f"  ... and {len(failed_results) - 10} more")
    
    print(f"\n✅ Questions saved to: {os.path.abspath(args.output_base)}")

if __name__ == "__main__":
    main()