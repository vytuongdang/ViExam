# ViExam: Are Vision Language Models Better than Humans on Vietnamese Multimodal Exam Questions?

This repository contains the official codebase and dataset generation pipeline for **ViExam**, the first Vietnamese multimodal exam benchmark for evaluating Vision-Language Models (VLMs). The benchmark consists of **2,548 genuine multimodal exam questions** across 7 domains:

> Mathematics, Physics, Chemistry, Biology, Geography, Driving Test, and IQ Test.

---

## 🧪 Dataset Overview

| Subject      | #Questions |
| ------------ | ---------- |
| Mathematics  | 456        |
| Physics      | 361        |
| Chemistry    | 302        |
| Biology      | 341        |
| Geography    | 481        |
| Driving Test | 367        |
| IQ Test      | 240        |
| **Total**    | **2,548**  |

> Each question is an image containing both Vietnamese text and visuals. Most are 4-option multiple-choice questions. No screenshots of text-only questions are included — all questions are genuinely **multimodal**.

---

## 🧰 Repository Structure

```
.
🕜── api_code/
│   ├── api_handlers/          # API wrapper for VLMs (e.g., Claude, Gemini, OpenAI)
│   ├── main_api.py            # Main API call logic
│   ├── backup_api_code/       # Backup or legacy code
│   └── main_api_qwen.py
│
🕜── api_key/                   # API credentials (plaintext)
│   ├── claude_key.txt
│   ├── openai_key.txt
│   └── ...
│
🕜── batch_api_code/            # Batch processing for large-scale evaluation
│   ├── main_batch_api.py
│   ├── main_batch_prepare.py
│   └── handlers/
│
🕜── src/                       # Full pipeline for data extraction and preprocessing
    ├── test_cut_question.py   # Image cutting logic
    ├── convert_pdf_to_image.py
    ├── check_question.html   # Evaluation parsing
    ├── result.py       # Accuracy tables
    └── ...
```

---

## 🚀 Quickstart

### 1. Install requirements

```bash
pip install -r src/requirements.txt
```

### 2. Run evaluation on VLMs

```bash
python batch_api_code/main_batch_api.py
```

Or for individual models:

```bash
python api_code/main_api.py --model gpt-4
```

### 3. Analyze results

```bash
python src/result.py
```

---

## ✂️ Dataset Preparation Pipeline

Full multimodal exam questions are extracted from real Vietnamese exams using:

* PDF → PNG conversion
* OCR + heuristics to detect question/image boundaries
* Geometric filtering to exclude text-only items
* Manual verification by Vietnamese native speakers (3x agreement)

> See `src/test_cut_question.py` for more.

---

## ✏️ Human-in-the-loop Enhancement

We provide a web-based tool for:

* Editing OCR results
* Verifying model descriptions
* Exporting ground truth data

---

## 📊 Evaluation Protocol

* Evaluation across **7 subject domains**
* Models tested:

  * GPT-4.1, Claude-Sonnet-4, Gemini 2.5 Flash, o3
  * Aya Vision, Gemma, Mistral, LLaMA 4, Qwen
* OCR benchmark for Vietnamese
* Option distribution bias analysis
* Cross-lingual prompting (Vietnamese vs English)
* Text-only vs multimodal comparison