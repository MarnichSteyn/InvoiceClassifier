# Invoice Extractor

A document-understanding pipeline that extracts structured data from invoices using a fine-tuned [LayoutLMv3](https://huggingface.co/microsoft/layoutlmv3-base) model trained on the [DocILE](https://github.com/rossumai/docile) dataset. The system combines neural token classification with heuristic post-processing to produce a rich, structured JSON output from any invoice PDF, image, or pre-computed DocILE OCR file.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Extracted Fields](#extracted-fields)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Python API](#python-api)
- [Output Schema](#output-schema)
- [Training Your Own Model](#training-your-own-model)
  - [1. Prepare the Dataset](#1-prepare-the-dataset)
  - [2. Fine-tune LayoutLMv3](#2-fine-tune-layoutlmv3)
  - [3. Select Demo Candidates (Optional)](#3-select-demo-candidates-optional)
- [Project Structure](#project-structure)
- [Tests](#tests)
- [Label Mapping](#label-mapping)
- [Requirements](#requirements)

---

## Overview

Invoice Extractor is an end-to-end pipeline with three main stages:

1. **OCR** — Extract words and their bounding boxes from a PDF/image via Tesseract, or load pre-computed bounding boxes from a DocILE OCR JSON file.
2. **Inference** — Run each word through a fine-tuned LayoutLMv3 model that assigns a BIO token-classification label (e.g. `B-INVOICE_NUMBER`, `I-VENDOR_NAME`, `O`).
3. **Post-processing** — Aggregate consecutive BIO spans into field values, cluster line-item rows by vertical proximity, detect currency, resolve subtotal ambiguity, and format everything into the target JSON schema.

The model was trained on the DocILE benchmark corpus, which contains thousands of business documents with fine-grained field-level annotations. After fine-tuning, inference runs on any single- or multi-page invoice without needing DocILE data at runtime.

---

## Architecture

```
Invoice PDF / Image          DocILE OCR JSON
        │                          │
        ▼                          ▼
┌──────────────────────────────────────────┐
│            OCR Layer  (ocr.py)           │
│  Tesseract + pdf2image  │  JSON loader   │
│  → normalized word list (0–1000 scale)   │
└─────────────────────────┬────────────────┘
                          │  {text, bbox, page, confidence}
                          ▼
┌──────────────────────────────────────────┐
│        Model Inference  (predict.py)     │
│  LayoutLMv3  →  per-word BIO labels      │
│  + confidence scores                     │
└─────────────────────────┬────────────────┘
                          │  [{text, bbox, label, conf}]
                          ▼
┌──────────────────────────────────────────┐
│      Post-processing  (postprocess.py)   │
│  • BIO span aggregation (2-pass merge)   │
│  • Line-item clustering by y-center      │
└─────────────────────────┬────────────────┘
                          ▼
┌──────────────────────────────────────────┐
│         Rules Engine  (rules.py)         │
│  • Currency detection                    │
│  • Subtotal disambiguation               │
└─────────────────────────┬────────────────┘
                          ▼
┌──────────────────────────────────────────┐
│      Schema Formatter  (schema_output.py)│
│  → Structured invoice JSON               │
└──────────────────────────────────────────┘
```

### Key design decisions

| Decision | Rationale |
|---|---|
| Bboxes normalized to 0–1000 | Matches LayoutLMv3's expected input range |
| Dummy 224×224 white image for visual features | Tesseract provides no page image; the model still benefits from layout embeddings |
| Two-pass span merging | First pass groups consecutive B/I tags; second pass bridges small gaps (<3 O tokens) while guarding against row-collapse via a 20-unit y-center threshold |
| Subword → word label mapping | Only the first subword of each word inherits the predicted label; subsequent subwords are masked (-100) during both training and inference |

---

## Extracted Fields

The model classifies tokens into **12 target labels** (plus `O`):

| Label | Description |
|---|---|
| `INVOICE_NUMBER` | Document / invoice ID |
| `INVOICE_DATE` | Invoice issue date |
| `DUE_DATE` | Payment due date |
| `VENDOR_NAME` | Issuing party name |
| `VENDOR_VAT` | Issuing party VAT / tax number |
| `VENDOR_ADDRESS` | Issuing party address |
| `CUSTOMER_NAME` | Receiving party name |
| `CUSTOMER_VAT` | Receiving party VAT / tax number |
| `CUSTOMER_ADDRESS` | Receiving party address |
| `SUBTOTAL` | Pre-tax amount |
| `TOTAL` | Final payable amount |
| `LINE_DESCRIPTION` | Line-item description |
| `QUANTITY` | Line-item quantity |
| `UNIT_PRICE` | Line-item unit price |
| `AMOUNT` | Line-item total amount |

---

## Installation

### Prerequisites

- Python ≥ 3.11
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed and on your `PATH` (required for PDF/image input)
- [Poppler](https://poppler.freedesktop.org/) installed (required by `pdf2image` for PDF → image conversion)
- A trained checkpoint in `./checkpoints/run2/` (or specify a custom path)

### Install

```bash
# Clone the repository
git clone <repo-url>
cd invoice_extractor

# Install in editable mode
pip install -e .
```

---

## Quick Start

```bash
# Extract from a PDF
python cli.py extract --input invoice.pdf

# Extract from an image
python cli.py extract --input invoice.png

# Extract from a pre-computed DocILE OCR JSON
python cli.py extract --ocr-json path/to/docile_ocr.json

# Save output to a file
python cli.py extract --input invoice.pdf --output result.json

# Use a custom checkpoint and confidence threshold
python cli.py extract --input invoice.pdf --checkpoint ./checkpoints/run1 --confidence 0.4
```

You can also invoke via the package entry point:

```bash
python -m invoice_extractor extract --input invoice.pdf
```

---

## CLI Reference

```
usage: cli.py extract [-h] (--input FILE | --ocr-json FILE)
                      [--checkpoint DIR] [--confidence FLOAT]
                      [--output FILE]

options:
  --input FILE          Path to input PDF or image file
  --ocr-json FILE       Path to pre-computed DocILE OCR JSON (mutually
                        exclusive with --input)
  --checkpoint DIR      Path to model checkpoint directory
                        (default: ./checkpoints/run2)
  --confidence FLOAT    Minimum per-token confidence for span inclusion
                        (default: 0.5)
  --output FILE         Write JSON result to this file (default: stdout)
```

---

## Python API

```python
from invoice_extractor.inference.extract import extract

# From a PDF or image file
result = extract(input_path="invoice.pdf")

# From a DocILE OCR JSON
result = extract(ocr_json="path/to/ocr.json")

# With custom options
result = extract(
    input_path="invoice.pdf",
    checkpoint="./checkpoints/run2",   # default
    confidence_threshold=0.4,
)

print(result["invoice"]["number"])      # e.g. "INV-2024-0042"
print(result["amounts"]["total"])       # e.g. "1 234.56"
print(result["amounts"]["currency"])    # e.g. "ZAR"
for item in result["line_items"]:
    print(item["description"], item["amount"])
```

---

## Output Schema

The extractor returns a single JSON object with the following structure:

```json
{
  "document_type": "invoice",
  "source_file": "invoice.pdf",
  "processed_at": "2024-05-06T14:32:00",

  "sent_by": {
    "name": "Acme Corp",
    "vat_number": "GB123456789",
    "address": "123 High Street, London, EC1A 1BB",
    "contact": null
  },

  "sent_to": {
    "name": "Client Ltd",
    "vat_number": null,
    "address": "456 Park Lane, Manchester",
    "reference": null
  },

  "invoice": {
    "number": "INV-2024-0042",
    "date": "2024-04-01",
    "due_date": "2024-04-30",
    "reference": null
  },

  "amounts": {
    "subtotal": "1 034.00",
    "vat_rate": null,
    "vat_amount": null,
    "total": "1 234.56",
    "currency": "GBP"
  },

  "payment": {
    "bank": null,
    "account_number": null,
    "branch_code": null,
    "account_name": null
  },

  "line_items": [
    {
      "description": "Consulting services — April",
      "quantity": "5",
      "unit_price": "200.00",
      "amount": "1 000.00"
    }
  ]
}
```

Fields that the model could not extract are returned as `null`. Dates are normalized to ISO 8601 (`YYYY-MM-DD`) where parseable. Monetary amounts retain their original string representation to avoid rounding errors.

**Supported currencies** (detected via symbol or ISO code in OCR text):

| Code | Symbols |
|---|---|
| ZAR | R |
| USD | $ |
| EUR | € |
| GBP | £ |
| INR | ₹ |
| AUD | A$ |
| CAD | C$ |

---

## Training Your Own Model

### 1. Prepare the Dataset

Convert raw DocILE data (OCR JSONs + annotation JSONs) into a HuggingFace `Dataset` ready for training:

```bash
python -m invoice_extractor.data.prepare_dataset \
  --docile-dir /path/to/docile \
  --out-dir ./prepared_data
```

The script:
- Reads document IDs from `docile_dir/train.json` and `docile_dir/val.json`
- For each document, loads OCR words and field annotations
- Assigns BIO labels by checking whether each word's bounding box is ≥50% contained within an annotated field region
- Normalizes bboxes to the 0–1000 integer scale LayoutLMv3 expects
- Saves Arrow-format train/val splits to `out-dir/`

**Options:**

```
--docile-dir DIR      Root of the DocILE dataset
--out-dir DIR         Output directory for the prepared splits
--max-docs INT        Limit number of documents (useful for debugging)
```

### 2. Fine-tune LayoutLMv3

```bash
python invoice_extractor/training/train.py \
  --data-dir ./prepared_data \
  --output-dir ./checkpoints/run3 \
  --epochs 3 \
  --batch-size 1 \
  --grad-accum-steps 8 \
  --lr 5e-5
```

**Key training options:**

| Flag | Default | Description |
|---|---|---|
| `--data-dir` | — | Directory containing train/ and val/ splits |
| `--output-dir` | — | Where to save checkpoints |
| `--model-name` | `microsoft/layoutlmv3-base` | Base model to fine-tune |
| `--epochs` | 3 | Number of training epochs |
| `--batch-size` | 1 | Per-device batch size |
| `--grad-accum-steps` | 8 | Gradient accumulation steps |
| `--lr` | 5e-5 | Learning rate |
| `--debug` | false | Quick 20-sample run (~3 min) for sanity-checking |

Training uses BF16 mixed precision and saves the best checkpoint by macro-averaged F1 (evaluated via [seqeval](https://github.com/chakki-works/seqeval) in IOB2 strict mode).

### 3. Select Demo Candidates (Optional)

Score and rank validation documents to find the best demo examples:

```bash
python -m invoice_extractor.data.triage_demo_candidates \
  --docile-dir /path/to/docile \
  --top-n 10 \
  --out triage_results.json
```

Documents are scored out of 100 across four dimensions:

| Dimension | Max pts | Criteria |
|---|---|---|
| OCR quality | 30 | Mean Tesseract confidence |
| Field coverage | 40 | Presence of 6 core fields (ID, date, vendor, customer, total, line items) |
| Layout density | 15 | 30–200 words = 15 pts; 200–500 = 10 pts; 500+ = 5 pts |
| Page count | 15 | 1 page = 15 pts; 2 pages = 8 pts; 3+ = 0 pts |

---

## Project Structure

```
invoice_extractor/
├── invoice_extractor/
│   ├── __main__.py              # python -m invoice_extractor entry point
│   ├── data/
│   │   ├── docile_loader.py     # OCR/annotation loading; BIO label assignment
│   │   ├── label_mapping.py     # DocILE→target field mappings; BIO label list
│   │   ├── prepare_dataset.py   # Dataset preparation CLI
│   │   └── triage_demo_candidates.py
│   ├── training/
│   │   ├── train.py             # LayoutLMv3 fine-tuning script
│   │   ├── tokenize.py          # Word→subword tokenization & label alignment
│   │   └── eval_metrics.py      # seqeval-based token classification metrics
│   └── inference/
│       ├── __init__.py          # Exports extract()
│       ├── ocr.py               # Tesseract OCR and DocILE JSON loader
│       ├── predict.py           # InvoiceExtractor model class
│       ├── extract.py           # End-to-end pipeline orchestrator
│       ├── postprocess.py       # Span aggregation, merging, line-item clustering
│       ├── rules.py             # Currency detection, subtotal disambiguation
│       └── schema_output.py     # Target JSON schema formatter
├── tests/
│   ├── test_label_mapping.py    # Unit tests: BIO labels, IoU, containment, assignment
│   └── test_inference.py        # Unit tests: money/date parsing, span aggregation, rules
├── checkpoints/
│   ├── run2/                    # Default checkpoint (recommended)
│   └── run1/                    # Alternative full training run
├── prepared_data/               # Small train/val split
├── prepared_data_full/          # Full dataset split
├── cli.py                       # Top-level CLI
└── pyproject.toml
```

---

## Tests

```bash
pytest tests/
```

The test suite covers:

- **`test_label_mapping.py`** — BIO label structure and round-trip consistency; IoU and containment calculations; `assign_labels()` with single/multi-word annotations across pages
- **`test_inference.py`** — Monetary amount parsing (Anglo and European formats, symbols, ISO codes); date parsing across 9 formats → ISO 8601; span aggregation (single token, multi-token, bbox union, merging, orphan I-tags, confidence thresholding); line-item clustering; currency detection and tie-breaking; subtotal disambiguation; end-to-end inference shape check (when checkpoint is present)

---

## Label Mapping

The 22 DocILE field types are mapped to 12 target labels:

| DocILE field | Target label |
|---|---|
| `document_id` | `INVOICE_NUMBER` |
| `date_issue` | `INVOICE_DATE` |
| `date_due` | `DUE_DATE` |
| `vendor` / `seller_name` | `VENDOR_NAME` |
| `vendor_tax_id` | `VENDOR_VAT` |
| `vendor_address` | `VENDOR_ADDRESS` |
| `customer` / `customer_billing_name` | `CUSTOMER_NAME` |
| `customer_tax_id` | `CUSTOMER_VAT` |
| `customer_billing_address` | `CUSTOMER_ADDRESS` |
| `amount_total_net` / `amount_due` | `SUBTOTAL` |
| `amount_total_gross` | `TOTAL` |
| `line_item/description` | `LINE_DESCRIPTION` |
| `line_item/quantity` | `QUANTITY` |
| `line_item/unit_price` | `UNIT_PRICE` |
| `line_item/amount` | `AMOUNT` |

Any DocILE field not in this mapping is treated as `O` (outside).

---

## Requirements

| Package | Purpose |
|---|---|
| `transformers` | LayoutLMv3 model and processor |
| `datasets` | HuggingFace Dataset I/O |
| `torch` | Deep learning backend |
| `accelerate` | Mixed-precision and multi-GPU training |
| `seqeval` | Token classification metrics (IOB2) |
| `pytesseract` | Tesseract OCR Python bindings |
| `pdf2image` | PDF → PIL Image conversion |
| `pillow` | Image handling |
| `tqdm` | Progress bars |

External system dependencies:
- **Tesseract** ≥ 4.0 — install via `apt install tesseract-ocr` or from the [Tesseract releases page](https://github.com/tesseract-ocr/tesseract/releases)
- **Poppler** — install via `apt install poppler-utils` (Linux) or [the Windows builds](https://github.com/oschwartz10612/poppler-windows/releases)

---

## License

This project uses the [DocILE](https://github.com/rossumai/docile) dataset. Refer to the DocILE repository for dataset licensing terms.
