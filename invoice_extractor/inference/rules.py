from __future__ import annotations

import re

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
