#!/usr/bin/env python
"""Top-level CLI entrypoint for invoice_extractor.

Usage:
    python cli.py extract --input invoice.pdf [--output result.json]
    python cli.py extract --ocr-json data/docile/ocr/DOC_ID.json [--output result.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="invoice_extractor",
        description="Extract structured fields from an invoice PDF, image, or DocILE OCR JSON.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ex = sub.add_parser("extract", help="Extract fields from an invoice")
    src = ex.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=Path, metavar="FILE", help="PDF or image file")
    src.add_argument(
        "--ocr-json", type=Path, metavar="FILE", help="Pre-computed DocILE OCR JSON"
    )
    ex.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("./checkpoints/run2"),
        metavar="DIR",
        help="Model checkpoint directory (default: ./checkpoints/run2)",
    )
    ex.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        metavar="FLOAT",
        help="Confidence threshold for span inclusion (default: 0.5)",
    )
    ex.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="FILE",
        help="Write JSON to this file instead of stdout",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "extract":
        from invoice_extractor.inference.extract import extract

        result = extract(
            input_path=args.input,
            ocr_json=args.ocr_json,
            checkpoint=args.checkpoint,
            confidence_threshold=args.confidence,
        )

        output = json.dumps(result, indent=2, ensure_ascii=False)

        if args.output:
            args.output.write_text(output, encoding="utf-8")
            print(f"Saved to {args.output}", file=sys.stderr)
        else:
            print(output)


if __name__ == "__main__":
    main()
