from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract(
    input_path: Path | None = None,
    ocr_json: Path | None = None,
    checkpoint: Path = Path("./checkpoints/run2"),
    confidence_threshold: float = 0.3,
) -> dict:
    """End-to-end pipeline. Returns the final JSON dict."""
    from invoice_extractor.inference.ocr import load_docile_ocr, ocr_pdf_or_image
    from invoice_extractor.inference.postprocess import aggregate_spans, cluster_line_items
    from invoice_extractor.inference.predict import InvoiceExtractor
    from invoice_extractor.inference.rules import detect_currency, disambiguate_subtotal
    from invoice_extractor.inference.schema_output import to_target_schema

    if input_path is None and ocr_json is None:
        raise ValueError("Provide either input_path or ocr_json")
    if input_path is not None and ocr_json is not None:
        raise ValueError("Provide either input_path or ocr_json, not both")

    if input_path is not None:
        words = ocr_pdf_or_image(Path(input_path))
        source_file = str(input_path)
    else:
        words = load_docile_ocr(Path(ocr_json))  # type: ignore[arg-type]
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
    spans_by_label = disambiguate_subtotal(spans_by_label)
    line_items = cluster_line_items(spans_by_label)
    currency = detect_currency(words)

    return to_target_schema(spans_by_label, line_items, currency, source_file)
