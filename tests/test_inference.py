"""Tests for the inference module."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the package root is importable when running pytest directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from invoice_extractor.inference.postprocess import (
    _parse_money,
    aggregate_spans,
    cluster_line_items,
)
from invoice_extractor.inference.rules import detect_currency, disambiguate_subtotal
from invoice_extractor.inference.schema_output import _parse_date

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CHECKPOINT = Path(__file__).parent.parent / "checkpoints" / "run2"
_OCR_DIR = Path("d:/Projects/DocILE/docile/data/docile/ocr")

_checkpoint_present = _CHECKPOINT.exists()
_ocr_present = _OCR_DIR.exists() and any(_OCR_DIR.glob("*.json"))


# ---------------------------------------------------------------------------
# _parse_money
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("1,234.56", 1234.56),
        ("1.234,56", 1234.56),
        ("1234.56", 1234.56),
        ("1234,56", 1234.56),
        ("R 1,234.56", 1234.56),
        ("€ 1.234,00", 1234.00),
        ("USD 500", 500.0),
        ("ZAR1000", 1000.0),
        ("1,000", 1000.0),
        ("", None),
        ("N/A", None),
    ],
)
def test_parse_money(text: str, expected: float | None) -> None:
    result = _parse_money(text)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# _parse_date
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("2024-01-25", "2024-01-25"),
        ("25/01/2024", "2024-01-25"),
        ("25-01-2024", "2024-01-25"),
        ("25 January 2024", "2024-01-25"),
        ("25 Jan 2024", "2024-01-25"),
        ("January 25, 2024", "2024-01-25"),
        ("2024/01/25", "2024-01-25"),
        ("not a date", "not a date"),
        (None, None),
    ],
)
def test_parse_date(text: str | None, expected: str | None) -> None:
    assert _parse_date(text) == expected


# ---------------------------------------------------------------------------
# aggregate_spans — basic BIO grouping
# ---------------------------------------------------------------------------


def _pred(text: str, bbox: list[int], label: str, conf: float = 0.9, page: int = 0) -> dict:
    return {"text": text, "bbox": bbox, "page": page, "label": label, "confidence": conf}


def test_aggregate_spans_single_token() -> None:
    preds = [_pred("INV-001", [100, 100, 300, 120], "B-INVOICE_NUMBER")]
    spans = aggregate_spans(preds)
    assert "INVOICE_NUMBER" in spans
    assert spans["INVOICE_NUMBER"][0]["text"] == "INV-001"
    assert spans["INVOICE_NUMBER"][0]["confidence"] == pytest.approx(0.9)


def test_aggregate_spans_multi_token_span() -> None:
    preds = [
        _pred("INV", [100, 100, 200, 120], "B-INVOICE_NUMBER", conf=0.9),
        _pred("2024-001", [210, 100, 350, 120], "I-INVOICE_NUMBER", conf=0.85),
    ]
    spans = aggregate_spans(preds)
    assert spans["INVOICE_NUMBER"][0]["text"] == "INV 2024-001"
    # confidence is the min over tokens
    assert spans["INVOICE_NUMBER"][0]["confidence"] == pytest.approx(0.85)


def test_aggregate_spans_bbox_union() -> None:
    preds = [
        _pred("ACME", [10, 300, 100, 320], "B-VENDOR_NAME"),
        _pred("Corp", [110, 295, 180, 325], "I-VENDOR_NAME"),
    ]
    spans = aggregate_spans(preds)
    bbox = spans["VENDOR_NAME"][0]["bbox"]
    assert bbox == [10, 295, 180, 325]


def test_aggregate_spans_o_tag_small_gap_merges() -> None:
    # 1 O token between same-label spans → gap < 3 → merged
    preds = [
        _pred("ACME", [10, 10, 100, 30], "B-VENDOR_NAME"),
        _pred("---", [110, 10, 200, 30], "O"),
        _pred("Corp", [210, 10, 300, 30], "I-VENDOR_NAME"),
    ]
    spans = aggregate_spans(preds)
    assert len(spans["VENDOR_NAME"]) == 1
    assert spans["VENDOR_NAME"][0]["text"] == "ACME Corp"


def test_aggregate_spans_gap_too_large_no_merge() -> None:
    # 3 O tokens between same-label spans → not fewer than 3 → no merge
    preds = [
        _pred("ACME", [10, 10, 100, 30], "B-VENDOR_NAME"),
        _pred("-", [110, 10, 140, 30], "O"),
        _pred("-", [150, 10, 180, 30], "O"),
        _pred("-", [190, 10, 220, 30], "O"),
        _pred("Corp", [230, 10, 330, 30], "B-VENDOR_NAME"),
    ]
    spans = aggregate_spans(preds)
    assert len(spans["VENDOR_NAME"]) == 2
    assert spans["VENDOR_NAME"][0]["text"] == "ACME"
    assert spans["VENDOR_NAME"][1]["text"] == "Corp"


def test_aggregate_spans_orphan_i_tag() -> None:
    """An I-tag without a preceding B-tag should start a new span."""
    preds = [_pred("100.00", [400, 500, 500, 520], "I-TOTAL")]
    spans = aggregate_spans(preds)
    assert "TOTAL" in spans
    assert spans["TOTAL"][0]["text"] == "100.00"


def test_aggregate_spans_low_confidence_treated_as_o() -> None:
    # "noise" is treated as O (conf < threshold), "Corp" is an orphan span.
    # Gap = 1 O token → merge pass combines them.
    preds = [
        _pred("ACME", [10, 10, 100, 30], "B-VENDOR_NAME", conf=0.9),
        _pred("noise", [110, 10, 200, 30], "I-VENDOR_NAME", conf=0.1),  # below threshold
        _pred("Corp", [210, 10, 300, 30], "I-VENDOR_NAME", conf=0.88),
    ]
    spans = aggregate_spans(preds, confidence_threshold=0.5)
    assert len(spans["VENDOR_NAME"]) == 1
    assert spans["VENDOR_NAME"][0]["text"] == "ACME Corp"


def test_aggregate_spans_multiple_fields() -> None:
    preds = [
        _pred("ACME", [10, 10, 100, 30], "B-VENDOR_NAME"),
        _pred("Corp", [110, 10, 200, 30], "I-VENDOR_NAME"),
        _pred("---", [0, 100, 10, 120], "O"),
        _pred("INV-007", [10, 200, 200, 220], "B-INVOICE_NUMBER"),
        _pred("---", [0, 300, 10, 320], "O"),
        _pred("1000.00", [300, 400, 450, 420], "B-TOTAL"),
    ]
    spans = aggregate_spans(preds)
    assert spans["VENDOR_NAME"][0]["text"] == "ACME Corp"
    assert spans["INVOICE_NUMBER"][0]["text"] == "INV-007"
    assert spans["TOTAL"][0]["text"] == "1000.00"


def test_aggregate_spans_empty_input() -> None:
    assert aggregate_spans([]) == {}


# ---------------------------------------------------------------------------
# cluster_line_items
# ---------------------------------------------------------------------------


def _span(text: str, x1: int, y1: int, x2: int, y2: int, page: int = 0) -> dict:
    return {
        "text": text,
        "bbox": [x1, y1, x2, y2],
        "page": page,
        "confidence": 0.9,
        "tokens": [],
    }


def test_cluster_line_items_single_row() -> None:
    spans = {
        "LINE_DESCRIPTION": [_span("Widget A", 10, 100, 200, 120)],
        "LINE_QUANTITY": [_span("2", 210, 102, 250, 118)],
        "LINE_UNIT_PRICE": [_span("50.00", 260, 101, 330, 119)],
        "LINE_AMOUNT": [_span("100.00", 340, 100, 430, 120)],
    }
    rows = cluster_line_items(spans)
    assert len(rows) == 1
    assert rows[0]["description"] == "Widget A"
    assert rows[0]["quantity"] == pytest.approx(2.0)
    assert rows[0]["unit_price"] == pytest.approx(50.0)
    assert rows[0]["amount"] == pytest.approx(100.0)


def test_cluster_line_items_two_rows() -> None:
    spans = {
        "LINE_DESCRIPTION": [
            _span("Widget A", 10, 100, 200, 120),
            _span("Widget B", 10, 200, 200, 220),
        ],
        "LINE_AMOUNT": [
            _span("100.00", 340, 100, 430, 120),
            _span("200.00", 340, 200, 430, 220),
        ],
    }
    rows = cluster_line_items(spans)
    assert len(rows) == 2
    assert rows[0]["description"] == "Widget A"
    assert rows[0]["amount"] == pytest.approx(100.0)
    assert rows[1]["description"] == "Widget B"
    assert rows[1]["amount"] == pytest.approx(200.0)


def test_cluster_line_items_empty() -> None:
    assert cluster_line_items({}) == []


# ---------------------------------------------------------------------------
# detect_currency
# ---------------------------------------------------------------------------


def _words_from_text(text: str) -> list[dict]:
    return [{"text": t, "bbox": [0, 0, 10, 10], "page": 0} for t in text.split()]


def test_detect_currency_zar_symbol() -> None:
    assert detect_currency(_words_from_text("R 1,234.56")) == "ZAR"


def test_detect_currency_zar_code() -> None:
    assert detect_currency(_words_from_text("ZAR 1234.56 ZAR")) == "ZAR"


def test_detect_currency_usd() -> None:
    assert detect_currency(_words_from_text("Total $ 500.00")) == "USD"


def test_detect_currency_eur_code_wins_over_symbol_tie() -> None:
    # EUR code present — should win over ambiguous results
    words = _words_from_text("EUR 100 EUR 200")
    assert detect_currency(words) == "EUR"


def test_detect_currency_none() -> None:
    assert detect_currency(_words_from_text("Total 500.00")) is None


# ---------------------------------------------------------------------------
# disambiguate_subtotal
# ---------------------------------------------------------------------------


def _money_span(text: str, conf: float = 0.9) -> dict:
    return {"text": text, "bbox": [0, 0, 100, 20], "page": 0, "confidence": conf, "tokens": []}


def test_disambiguate_subtotal_two_totals() -> None:
    spans = {"TOTAL": [_money_span("1000.00"), _money_span("800.00")]}
    result = disambiguate_subtotal(spans)
    assert result["SUBTOTAL"][0]["text"] == "800.00"
    assert result["TOTAL"][0]["text"] == "1000.00"


def test_disambiguate_subtotal_existing_subtotal_untouched() -> None:
    spans = {
        "SUBTOTAL": [_money_span("800.00")],
        "TOTAL": [_money_span("1000.00"), _money_span("900.00")],
    }
    result = disambiguate_subtotal(spans)
    # Existing subtotal preserved; TOTAL list unchanged
    assert result["SUBTOTAL"][0]["text"] == "800.00"
    assert len(result["TOTAL"]) == 2


def test_disambiguate_subtotal_single_total_untouched() -> None:
    spans = {"TOTAL": [_money_span("1000.00")]}
    result = disambiguate_subtotal(spans)
    assert "SUBTOTAL" not in result or not result.get("SUBTOTAL")
    assert len(result["TOTAL"]) == 1


# ---------------------------------------------------------------------------
# Full inference — shape check (requires checkpoint + OCR data)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _checkpoint_present,
    reason=f"Checkpoint not found at {_CHECKPOINT}",
)
@pytest.mark.skipif(
    not _ocr_present,
    reason=f"No OCR JSON files found in {_OCR_DIR}",
)
def test_full_inference_output_shape() -> None:
    """Run end-to-end extraction on one DocILE val doc and verify schema shape."""
    from invoice_extractor.inference.extract import extract

    ocr_file = sorted(_OCR_DIR.glob("*.json"))[0]
    result = extract(ocr_json=ocr_file, checkpoint=_CHECKPOINT)

    # Top-level keys
    for key in ("document_type", "source_file", "processed_at", "sent_by", "sent_to",
                "invoice", "amounts", "payment", "line_items"):
        assert key in result, f"Missing key: {key}"

    assert result["document_type"] in ("tax_invoice", "unknown")
    assert isinstance(result["line_items"], list)

    amounts = result["amounts"]
    for field in ("subtotal", "vat_amount", "total", "currency"):
        assert field in amounts
        val = amounts[field]
        assert val is None or isinstance(val, (int, float, str)), (
            f"amounts.{field} has unexpected type: {type(val)}"
        )

    sent_by = result["sent_by"]
    assert "name" in sent_by and "vat_number" in sent_by
    assert "contact" in sent_by
    assert "phone" in sent_by["contact"] and "email" in sent_by["contact"]

    invoice = result["invoice"]
    for field in ("number", "date", "due_date"):
        assert field in invoice

    payment = result["payment"]
    for field in ("bank", "account_number", "branch_code", "account_name"):
        assert field in payment

    for item in result["line_items"]:
        for field in ("description", "quantity", "unit_price", "amount"):
            assert field in item
