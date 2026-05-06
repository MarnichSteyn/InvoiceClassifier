from __future__ import annotations

import re
from datetime import datetime, timezone


def _best_span(spans: list[dict]) -> dict | None:
    if not spans:
        return None
    return max(spans, key=lambda s: s["confidence"])


def _best_text(spans_by_label: dict, label: str) -> str | None:
    span = _best_span(spans_by_label.get(label, []))
    return span["text"] if span else None


_DATE_FORMATS = [
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%d %B %Y",
    "%d %b %Y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%Y/%m/%d",
    "%d.%m.%Y",
]


def _parse_date(text: str | None) -> str | None:
    if not text:
        return None
    text = text.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return text  # return raw string if nothing parses


def _parse_money(text: str | None) -> float | None:
    if text is None:
        return None
    from invoice_extractor.inference.postprocess import _parse_money as _pm

    return _pm(text)


def to_target_schema(
    spans_by_label: dict,
    line_items: list[dict],
    currency: str | None,
    source_file: str | None,
) -> dict:
    """Build the final output dict in the target schema."""
    return {
        "document_type": "tax_invoice",
        "source_file": source_file,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "sent_by": {
            "name": _best_text(spans_by_label, "VENDOR_NAME"),
            "vat_number": _best_text(spans_by_label, "VENDOR_VAT"),
            "address": _best_text(spans_by_label, "VENDOR_ADDRESS"),
            "contact": {"phone": None, "email": None},
        },
        "sent_to": {
            "name": _best_text(spans_by_label, "CUSTOMER_NAME"),
            "vat_number": None,
            "address": _best_text(spans_by_label, "CUSTOMER_ADDRESS"),
            "reference": None,
        },
        "invoice": {
            "number": _best_text(spans_by_label, "INVOICE_NUMBER"),
            "date": _parse_date(_best_text(spans_by_label, "INVOICE_DATE")),
            "due_date": _parse_date(_best_text(spans_by_label, "DUE_DATE")),
            "reference": None,
        },
        "amounts": {
            "subtotal": _parse_money(_best_text(spans_by_label, "SUBTOTAL")),
            "vat_rate": None,
            "vat_amount": _parse_money(_best_text(spans_by_label, "VAT_AMOUNT")),
            "total": _parse_money(_best_text(spans_by_label, "TOTAL")),
            "currency": currency,
        },
        "payment": {
            "bank": None,
            "account_number": None,
            "branch_code": None,
            "account_name": None,
        },
        "line_items": line_items,
    }
