from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Each value is a list of regex patterns for that currency.
# Symbol patterns are kept separate from code patterns for tie-breaking.
_SYMBOL_PATTERNS: dict[str, list[str]] = {
    "ZAR": [r"\bR(?=[\s\d])"],
    "USD": [r"\$"],
    "EUR": [r"€"],
    "GBP": [r"£"],
    "INR": [r"₹"],
}

_CODE_PATTERNS: dict[str, str] = {
    "ZAR": r"\bZAR\b",
    "USD": r"\bUSD\b",
    "EUR": r"\bEUR\b",
    "GBP": r"\bGBP\b",
    "INR": r"\bINR\b",
    "AUD": r"\bAUD\b",
    "CAD": r"\bCAD\b",
}

_ALL_CURRENCIES = list(_CODE_PATTERNS.keys())


def detect_currency(words: list[dict]) -> str | None:
    """
    Search OCR text for currency codes and symbols.
    Returns ISO code (ZAR, USD, EUR, GBP, INR, AUD, CAD) or None.
    Tie-break: prefer the currency whose ISO code (not symbol) appears.
    """
    full_text = " ".join(w["text"] for w in words)

    total: dict[str, int] = {c: 0 for c in _ALL_CURRENCIES}
    code_hits: dict[str, int] = {c: 0 for c in _ALL_CURRENCIES}

    for currency, pattern in _CODE_PATTERNS.items():
        hits = len(re.findall(pattern, full_text))
        total[currency] += hits
        code_hits[currency] = hits

    for currency, patterns in _SYMBOL_PATTERNS.items():
        for pattern in patterns:
            total[currency] += len(re.findall(pattern, full_text))

    max_count = max(total.values())
    if max_count == 0:
        return None

    candidates = [c for c, cnt in total.items() if cnt == max_count]
    if len(candidates) == 1:
        return candidates[0]

    # Tie-break: highest ISO-code hit count
    return max(candidates, key=lambda c: code_hits[c])


def disambiguate_subtotal(spans_by_label: dict) -> dict:
    """
    If SUBTOTAL is absent and TOTAL has 2+ spans, treat the smallest value as
    SUBTOTAL and the largest as TOTAL.
    Returns a shallow-copied spans_by_label with possibly updated SUBTOTAL/TOTAL.
    """
    from invoice_extractor.inference.postprocess import _parse_money

    result = {k: list(v) for k, v in spans_by_label.items()}

    if result.get("SUBTOTAL"):
        return result

    total_spans = result.get("TOTAL", [])
    if len(total_spans) < 2:
        return result

    parsed: list[tuple[float, dict]] = []
    for span in total_spans:
        value = _parse_money(span["text"])
        if value is not None:
            parsed.append((value, span))

    if len(parsed) < 2:
        return result

    parsed.sort(key=lambda x: x[0])
    result["SUBTOTAL"] = [parsed[0][1]]
    result["TOTAL"] = [parsed[-1][1]]
    return result


# ---------------------------------------------------------------------------
# Party-name keyword recovery
# ---------------------------------------------------------------------------

_CUSTOMER_KEYWORDS_1: frozenset[str] = frozenset({
    "client:", "client", "customer:", "customer", "account:",
})
_CUSTOMER_KEYWORDS_2: frozenset[str] = frozenset({
    "bill to:", "bill to", "billed to:", "billed to",
    "invoiced to:", "invoiced to", "sold to:", "sold to",
    "ship to:", "ship to", "account holder:",
})

_VENDOR_KEYWORDS_1: frozenset[str] = frozenset({
    "from:", "vendor:", "supplier:", "biller:", "company:",
})
_VENDOR_KEYWORDS_2: frozenset[str] = frozenset({
    "issued by:",
})

_ALL_KEYWORDS_1 = _CUSTOMER_KEYWORDS_1 | _VENDOR_KEYWORDS_1
_ALL_KEYWORDS_2 = _CUSTOMER_KEYWORDS_2 | _VENDOR_KEYWORDS_2

# Stop collecting tokens when the text looks like the start of an address.
_ADDRESS_STOP_RE = re.compile(r"^\d+|P\.O\.|PO\s*Box", re.IGNORECASE)
_STREET_SUFFIXES: frozenset[str] = frozenset({
    "street", "st", "ave", "avenue", "road", "rd",
    "drive", "dr", "lane", "ln", "blvd", "boulevard",
})
# Additional single-token stops used when walking address spans for a name prefix.
_ADDRESS_SPAN_STOPS: frozenset[str] = frozenset({
    "attn", "attn.", "attention", "c/o", "po", "box",
})


def recover_party_names(words: list[dict], spans_by_label: dict) -> dict:
    """Fallback for VENDOR_NAME / CUSTOMER_NAME when the model didn't extract them.

    Scans OCR text for known keyword markers and grabs the following tokens
    if the corresponding name field has no model prediction.

    Returns spans_by_label with potentially new entries added for
    VENDOR_NAME and/or CUSTOMER_NAME under a special 'source': 'rule' flag.
    """
    result = {k: list(v) for k, v in spans_by_label.items()}

    needs_vendor = not result.get("VENDOR_NAME")
    needs_customer = not result.get("CUSTOMER_NAME")

    if not needs_vendor and not needs_customer:
        return result

    n = len(words)

    for i, word in enumerate(words):
        text_i = word["text"].lower().strip()
        text_i2 = (text_i + " " + words[i + 1]["text"].lower().strip()) if i + 1 < n else ""

        # Determine which label this keyword matches and how many tokens the keyword spans.
        matched_label: str | None = None
        kw_span = 0
        matched_kw = ""

        if needs_customer:
            if text_i in _CUSTOMER_KEYWORDS_1:
                matched_label, kw_span, matched_kw = "CUSTOMER_NAME", 1, text_i
            elif text_i2 in _CUSTOMER_KEYWORDS_2:
                matched_label, kw_span, matched_kw = "CUSTOMER_NAME", 2, text_i2

        if needs_vendor and matched_label is None:
            if text_i in _VENDOR_KEYWORDS_1:
                matched_label, kw_span, matched_kw = "VENDOR_NAME", 1, text_i
            elif text_i2 in _VENDOR_KEYWORDS_2:
                matched_label, kw_span, matched_kw = "VENDOR_NAME", 2, text_i2

        if matched_label is None:
            continue

        # Skip if this label was already filled by a prior iteration.
        if result.get(matched_label):
            continue

        kw_y = (word["bbox"][1] + word["bbox"][3]) / 2
        collected: list[dict] = []
        prev_y = kw_y

        for j in range(i + kw_span, min(i + kw_span + 4, n)):
            tok = words[j]
            tok_y = (tok["bbox"][1] + tok["bbox"][3]) / 2
            tok_lower = tok["text"].lower().strip()
            tok_lower_next = (tok_lower + " " + words[j + 1]["text"].lower().strip()) if j + 1 < n else ""

            # Terminators
            if abs(tok_y - kw_y) > 10:
                break
            if abs(tok_y - prev_y) > 30:
                break
            if _ADDRESS_STOP_RE.match(tok["text"]):
                break
            if tok_lower in _STREET_SUFFIXES:
                break
            if tok_lower in _ALL_KEYWORDS_1 or tok_lower_next in _ALL_KEYWORDS_2:
                break

            collected.append(tok)
            prev_y = tok_y

        if not collected:
            continue

        candidate = " ".join(t["text"] for t in collected)
        if len(candidate) < 3 or candidate.replace(" ", "").isdigit():
            continue

        span = {
            "text": candidate,
            "bbox": [
                min(t["bbox"][0] for t in collected),
                min(t["bbox"][1] for t in collected),
                max(t["bbox"][2] for t in collected),
                max(t["bbox"][3] for t in collected),
            ],
            "page": collected[0]["page"],
            "confidence": 0.5,
            "source": "rule",
            "tokens": list(collected),
        }
        result[matched_label] = [span]
        logger.info("[rule] Recovered %s=%r via keyword %r", matched_label, candidate, matched_kw)

        if matched_label == "VENDOR_NAME":
            needs_vendor = False
        else:
            needs_customer = False

        if not needs_vendor and not needs_customer:
            break

    return result


# ---------------------------------------------------------------------------
# Address-first-line name promotion
# ---------------------------------------------------------------------------


def promote_names_from_addresses(spans_by_label: dict) -> dict:
    """Promote the leading non-address tokens of a party's address span to their
    name field when the name field is absent.

    Handles the common pattern where the model labels a block like
    'ACME Corp\\n123 Main St…' as VENDOR_ADDRESS: the first line is the
    company name, not the address proper.

    Applies to both VENDOR_NAME/VENDOR_ADDRESS and CUSTOMER_NAME/CUSTOMER_ADDRESS.
    """
    result = {k: list(v) for k, v in spans_by_label.items()}

    for name_label, addr_label in (
        ("VENDOR_NAME", "VENDOR_ADDRESS"),
        ("CUSTOMER_NAME", "CUSTOMER_ADDRESS"),
    ):
        if result.get(name_label):
            continue
        addr_spans = result.get(addr_label)
        if not addr_spans:
            continue

        addr_span = max(addr_spans, key=lambda s: s["confidence"])
        tokens = addr_span.get("tokens", [])
        if not tokens:
            continue

        name_tokens = []
        for tok in tokens:
            tok_text = tok["text"]
            tok_lower = tok_text.lower().strip().rstrip(":")
            if _ADDRESS_STOP_RE.match(tok_text):
                break
            if tok_lower in _STREET_SUFFIXES or tok_lower in _ADDRESS_SPAN_STOPS:
                break
            name_tokens.append(tok)

        if not name_tokens:
            continue

        candidate = " ".join(t["text"] for t in name_tokens)
        if len(candidate) < 3 or candidate.replace(" ", "").isdigit():
            continue

        span = {
            "text": candidate,
            "bbox": [
                min(t["bbox"][0] for t in name_tokens),
                min(t["bbox"][1] for t in name_tokens),
                max(t["bbox"][2] for t in name_tokens),
                max(t["bbox"][3] for t in name_tokens),
            ],
            "page": name_tokens[0]["page"],
            "confidence": min(t["confidence"] for t in name_tokens),
            "source": "rule",
            "tokens": list(name_tokens),
        }
        result[name_label] = [span]
        logger.info(
            "[rule] Promoted %s=%r from first line of %s",
            name_label, candidate, addr_label,
        )

    return result
