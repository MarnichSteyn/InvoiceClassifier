from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_ocr_json(path: Path) -> list[dict]:
    """Auto-detect OCR JSON format and dispatch to the right loader."""
    import json

    from invoice_extractor.inference.ocr import load_colleague_ocr, load_docile_ocr

    with open(path, encoding="utf-8") as f:
        top_keys = set(json.load(f).keys())

    if "metadata" in top_keys and "extracted_fields" in top_keys:
        logger.debug("Detected colleague OCR format for %s", path)
        return load_colleague_ocr(path)

    logger.debug("Detected DocILE OCR format for %s", path)
    return load_docile_ocr(path)


def extract(
    input_path: Path | None = None,
    ocr_json: Path | None = None,
    checkpoint: Path = Path("./checkpoints/run2"),
    confidence_threshold: float = 0.3,
) -> dict:
    """End-to-end pipeline. Returns the final JSON dict."""
    from invoice_extractor.inference.ocr import ocr_pdf_or_image
    from invoice_extractor.inference.postprocess import aggregate_spans, cluster_line_items
    from invoice_extractor.inference.predict import InvoiceExtractor
    from invoice_extractor.inference.rules import (
        detect_currency,
        disambiguate_subtotal,
        promote_names_from_addresses,
        recover_party_names,
    )
    from invoice_extractor.inference.schema_output import to_target_schema

    if input_path is None and ocr_json is None:
        raise ValueError("Provide either input_path or ocr_json")
    if input_path is not None and ocr_json is not None:
        raise ValueError("Provide either input_path or ocr_json, not both")

    if input_path is not None:
        words = ocr_pdf_or_image(Path(input_path))
        source_file = str(input_path)
    else:
        words = _load_ocr_json(Path(ocr_json))  # type: ignore[arg-type]
        source_file = str(ocr_json)

    if not words:
        logger.warning("No words detected — returning null result for %s", source_file)
        return {
            **to_target_schema({}, [], None, source_file),
            "document_type": "unknown",
        }

    extractor = InvoiceExtractor(Path(checkpoint))
    predictions = extractor.predict(words)

    spans_by_label = aggregate_spans(predictions, confidence_threshold=confidence_threshold)
    spans_by_label = recover_party_names(words, spans_by_label)
    spans_by_label = promote_names_from_addresses(spans_by_label)
    spans_by_label = disambiguate_subtotal(spans_by_label)
    line_items = cluster_line_items(spans_by_label)
    currency = detect_currency(words)

    return to_target_schema(spans_by_label, line_items, currency, source_file)
