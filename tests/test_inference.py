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
from invoice_extractor.inference.rules import (
    detect_currency,
    disambiguate_subtotal,
    promote_names_from_addresses,
    recover_party_names,
)
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
# recover_party_names
# ---------------------------------------------------------------------------


def _w(text: str, x1: int, y1: int, x2: int, y2: int, page: int = 0) -> dict:
    return {"text": text, "bbox": [x1, y1, x2, y2], "page": page, "confidence": 0.9}


def test_recover_customer_name_via_client_keyword() -> None:
    words = [
        _w("Client:", 10, 100, 80, 120),
        _w("Lorillard", 90, 100, 200, 120),
        _w("Tobacco", 210, 100, 300, 120),
        _w("Co", 310, 100, 350, 120),
    ]
    spans = recover_party_names(words, {})
    assert "CUSTOMER_NAME" in spans
    assert spans["CUSTOMER_NAME"][0]["text"] == "Lorillard Tobacco Co"
    assert spans["CUSTOMER_NAME"][0]["source"] == "rule"


def test_recover_customer_name_two_token_keyword() -> None:
    words = [
        _w("Bill", 10, 200, 60, 220),
        _w("To:", 65, 200, 110, 220),
        _w("Acme", 120, 200, 200, 220),
        _w("Ltd", 205, 200, 260, 220),
    ]
    spans = recover_party_names(words, {})
    assert "CUSTOMER_NAME" in spans
    assert spans["CUSTOMER_NAME"][0]["text"] == "Acme Ltd"
    assert spans["CUSTOMER_NAME"][0]["source"] == "rule"


def test_recover_vendor_name_via_from_keyword() -> None:
    words = [
        _w("From:", 10, 50, 70, 70),
        _w("BSMG", 80, 50, 150, 70),
        _w("Worldwide,", 155, 50, 270, 70),
        _w("Inc.", 275, 50, 330, 70),
    ]
    spans = recover_party_names(words, {})
    assert "VENDOR_NAME" in spans
    assert spans["VENDOR_NAME"][0]["text"] == "BSMG Worldwide, Inc."
    assert spans["VENDOR_NAME"][0]["source"] == "rule"


def test_recover_skips_if_model_already_found_name() -> None:
    words = [
        _w("Client:", 10, 100, 80, 120),
        _w("ShouldBeIgnored", 90, 100, 250, 120),
    ]
    existing = {"CUSTOMER_NAME": [{"text": "Model Result", "bbox": [0, 0, 100, 20],
                                   "page": 0, "confidence": 0.85, "tokens": []}]}
    spans = recover_party_names(words, existing)
    assert spans["CUSTOMER_NAME"][0]["text"] == "Model Result"


def test_recover_stops_at_address_digit() -> None:
    words = [
        _w("Client:", 10, 100, 80, 120),
        _w("Acme", 90, 100, 160, 120),
        _w("123", 165, 100, 200, 120),   # starts with digit — stop before this
        _w("Main", 205, 100, 260, 120),
    ]
    spans = recover_party_names(words, {})
    assert spans["CUSTOMER_NAME"][0]["text"] == "Acme"


def test_recover_stops_at_line_break() -> None:
    words = [
        _w("Client:", 10, 100, 80, 120),
        _w("Acme", 90, 100, 160, 120),
        _w("Corp", 10, 200, 80, 220),   # y-jump of ~90 — not on same line
    ]
    spans = recover_party_names(words, {})
    assert spans["CUSTOMER_NAME"][0]["text"] == "Acme"


def test_recover_returns_unchanged_when_no_keyword() -> None:
    words = [
        _w("Invoice", 10, 10, 100, 30),
        _w("Number:", 110, 10, 220, 30),
        _w("INV-001", 230, 10, 350, 30),
    ]
    spans = recover_party_names(words, {})
    assert "CUSTOMER_NAME" not in spans
    assert "VENDOR_NAME" not in spans


def test_recover_candidate_too_short_skipped() -> None:
    words = [
        _w("Client:", 10, 100, 80, 120),
        _w("AB", 90, 100, 120, 120),   # < 3 chars
    ]
    spans = recover_party_names(words, {})
    assert "CUSTOMER_NAME" not in spans


# ---------------------------------------------------------------------------
# promote_names_from_addresses
# ---------------------------------------------------------------------------


def _addr_span(tokens: list[dict], conf: float = 0.82) -> dict:
    """Build a span dict whose 'tokens' list is the raw word dicts."""
    return {
        "text": " ".join(t["text"] for t in tokens),
        "bbox": [
            min(t["bbox"][0] for t in tokens),
            min(t["bbox"][1] for t in tokens),
            max(t["bbox"][2] for t in tokens),
            max(t["bbox"][3] for t in tokens),
        ],
        "page": 0,
        "confidence": conf,
        "tokens": list(tokens),
    }


def test_promote_vendor_name_from_address() -> None:
    tokens = [
        _w("BSMG", 10, 238, 80, 252),
        _w("Worldwide,", 85, 238, 200, 252),
        _w("Inc.", 205, 238, 260, 252),
        _w("P.O.", 10, 253, 60, 267),    # address stop
        _w("Box", 65, 253, 110, 267),
        _w("100583", 115, 253, 190, 267),
    ]
    spans = {"VENDOR_ADDRESS": [_addr_span(tokens)]}
    result = promote_names_from_addresses(spans)
    assert "VENDOR_NAME" in result
    assert result["VENDOR_NAME"][0]["text"] == "BSMG Worldwide, Inc."
    assert result["VENDOR_NAME"][0]["source"] == "rule"


def test_promote_customer_name_stops_at_attn() -> None:
    tokens = [
        _w("LORILLARD", 10, 210, 140, 224),
        _w("Attn:", 10, 226, 70, 237),    # attention line — stop before this
        _w("Sheldon", 75, 226, 160, 237),
    ]
    spans = {"CUSTOMER_ADDRESS": [_addr_span(tokens)]}
    result = promote_names_from_addresses(spans)
    assert result["CUSTOMER_NAME"][0]["text"] == "LORILLARD"
    assert result["CUSTOMER_NAME"][0]["source"] == "rule"


def test_promote_skips_when_name_already_present() -> None:
    tokens = [_w("BSMG", 10, 238, 80, 252), _w("P.O.", 10, 253, 60, 267)]
    spans = {
        "VENDOR_NAME": [{"text": "Model Result", "bbox": [0, 0, 100, 20],
                         "page": 0, "confidence": 0.9, "tokens": []}],
        "VENDOR_ADDRESS": [_addr_span(tokens)],
    }
    result = promote_names_from_addresses(spans)
    assert result["VENDOR_NAME"][0]["text"] == "Model Result"


def test_promote_skips_when_no_address_span() -> None:
    result = promote_names_from_addresses({})
    assert "VENDOR_NAME" not in result
    assert "CUSTOMER_NAME" not in result


def test_promote_stops_at_digit_start() -> None:
    tokens = [
        _w("Acme", 10, 100, 80, 120),
        _w("123", 10, 135, 80, 150),     # starts with digit — stop
        _w("Main", 85, 135, 160, 150),
    ]
    spans = {"VENDOR_ADDRESS": [_addr_span(tokens)]}
    result = promote_names_from_addresses(spans)
    assert result["VENDOR_NAME"][0]["text"] == "Acme"


def test_promote_skips_all_digit_candidate() -> None:
    tokens = [
        _w("12345", 10, 100, 80, 120),   # digit-only — too short/invalid after stripping
        _w("Main", 85, 100, 160, 120),
    ]
    spans = {"VENDOR_ADDRESS": [_addr_span(tokens)]}
    result = promote_names_from_addresses(spans)
    # "12345" would be stopped by _ADDRESS_STOP_RE (starts with digit) so nothing collected
    assert "VENDOR_NAME" not in result


def test_promote_uses_highest_confidence_address_span() -> None:
    tok_low = [_w("LowConf", 10, 100, 100, 120), _w("Corp", 105, 100, 170, 120),
               _w("P.O.", 10, 135, 60, 150)]
    tok_high = [_w("HighConf", 10, 200, 120, 220), _w("Inc.", 125, 200, 175, 220),
                _w("P.O.", 10, 235, 60, 250)]
    spans = {"VENDOR_ADDRESS": [_addr_span(tok_low, conf=0.4), _addr_span(tok_high, conf=0.9)]}
    result = promote_names_from_addresses(spans)
    assert result["VENDOR_NAME"][0]["text"] == "HighConf Inc."


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


# ---------------------------------------------------------------------------
# load_colleague_ocr
# ---------------------------------------------------------------------------


_COLLEAGUE_OCR_SAMPLE = {
    "metadata": {"source": "paddleocr"},
    "pages": [
        {
            "page_number": 1,
            "width": 595.32,
            "height": 841.92,
            "blocks": [
                {
                    "block_id": "p1_b0_w0",
                    "block_type": "WORD",
                    "text": "TAX",
                    "confidence": 99.9,
                    "bbox": {"left": 0.06511, "top": 0.06335, "width": 0.02794, "height": 0.01271},
                    "word_ids": None,
                },
                {
                    "block_id": "p1_b0_l0",
                    "block_type": "LINE",
                    "text": "TAX INVOICE",
                    "confidence": 98.0,
                    "bbox": {"left": 0.06511, "top": 0.06335, "width": 0.10000, "height": 0.01271},
                    "word_ids": ["p1_b0_w0"],
                },
                {
                    "block_id": "p1_b1_w0",
                    "block_type": "WORD",
                    "text": "",  # empty — should be skipped
                    "confidence": 50.0,
                    "bbox": {"left": 0.2, "top": 0.2, "width": 0.05, "height": 0.02},
                    "word_ids": None,
                },
                {
                    "block_id": "p1_b2_w0",
                    "block_type": "WORD",
                    "text": "INVOICE",
                    "confidence": 95.5,
                    "bbox": {"left": 0.1, "top": 0.06335, "width": 0.0, "height": 0.01271},  # zero width — skip
                    "word_ids": None,
                },
            ],
        },
        {
            "page_number": 2,
            "width": 595.32,
            "height": 841.92,
            "blocks": [
                {
                    "block_id": "p2_b0_w0",
                    "block_type": "WORD",
                    "text": "Total",
                    "confidence": 88.0,
                    "bbox": {"left": 0.5, "top": 0.9, "width": 0.1, "height": 0.02},
                    "word_ids": None,
                },
            ],
        },
    ],
    "extracted_fields": {"document_type": "tax_invoice"},
}


def test_load_colleague_ocr_basic(tmp_path: Path) -> None:
    import json

    from invoice_extractor.inference.ocr import load_colleague_ocr

    ocr_file = tmp_path / "colleague.json"
    ocr_file.write_text(json.dumps(_COLLEAGUE_OCR_SAMPLE), encoding="utf-8")

    words = load_colleague_ocr(ocr_file)

    # Only WORD blocks with non-empty text and non-zero area should appear
    assert len(words) == 2, f"Expected 2 words, got {len(words)}: {words}"


def test_load_colleague_ocr_format(tmp_path: Path) -> None:
    import json

    from invoice_extractor.inference.ocr import load_colleague_ocr

    ocr_file = tmp_path / "colleague.json"
    ocr_file.write_text(json.dumps(_COLLEAGUE_OCR_SAMPLE), encoding="utf-8")

    words = load_colleague_ocr(ocr_file)
    tax_word = words[0]

    # text
    assert tax_word["text"] == "TAX"

    # page is 0-indexed (page_number 1 → 0)
    assert tax_word["page"] == 0

    # confidence normalised to 0-1
    assert tax_word["confidence"] == pytest.approx(0.999, rel=1e-4)

    # bbox in 0-1000 integer scale
    bbox = tax_word["bbox"]
    assert len(bbox) == 4
    assert all(isinstance(v, int) for v in bbox)
    assert bbox[0] == int(round(0.06511 * 1000))   # x1
    assert bbox[1] == int(round(0.06335 * 1000))   # y1
    assert bbox[2] == int(round((0.06511 + 0.02794) * 1000))  # x2
    assert bbox[3] == int(round((0.06335 + 0.01271) * 1000))  # y2
    assert 0 <= bbox[0] < bbox[2] <= 1000
    assert 0 <= bbox[1] < bbox[3] <= 1000


def test_load_colleague_ocr_second_page(tmp_path: Path) -> None:
    import json

    from invoice_extractor.inference.ocr import load_colleague_ocr

    ocr_file = tmp_path / "colleague.json"
    ocr_file.write_text(json.dumps(_COLLEAGUE_OCR_SAMPLE), encoding="utf-8")

    words = load_colleague_ocr(ocr_file)
    total_word = words[1]

    assert total_word["text"] == "Total"
    assert total_word["page"] == 1  # page_number 2 → index 1


def test_load_colleague_ocr_line_blocks_filtered(tmp_path: Path) -> None:
    import json

    from invoice_extractor.inference.ocr import load_colleague_ocr

    ocr_file = tmp_path / "colleague.json"
    ocr_file.write_text(json.dumps(_COLLEAGUE_OCR_SAMPLE), encoding="utf-8")

    words = load_colleague_ocr(ocr_file)
    texts = [w["text"] for w in words]

    # "TAX INVOICE" is a LINE block and must not appear
    assert "TAX INVOICE" not in texts


def test_extract_auto_detects_colleague_format(tmp_path: Path) -> None:
    """_load_ocr_json should route colleague-format files to load_colleague_ocr."""
    import json

    from invoice_extractor.inference.extract import _load_ocr_json

    ocr_file = tmp_path / "colleague.json"
    ocr_file.write_text(json.dumps(_COLLEAGUE_OCR_SAMPLE), encoding="utf-8")

    words = _load_ocr_json(ocr_file)
    assert len(words) == 2
    assert words[0]["text"] == "TAX"
