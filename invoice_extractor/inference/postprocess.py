from __future__ import annotations

import re


def _parse_money(text: str) -> float | None:
    """Parse a monetary string in anglo (1,234.56) or european (1.234,56) format."""
    text = text.strip()
    text = re.sub(r"[€$£₹]", "", text)
    text = re.sub(r"\b(ZAR|USD|EUR|GBP|INR|AUD|CAD)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bR\s+", "", text)
    text = text.strip()

    has_dot = "." in text
    has_comma = "," in text

    if has_dot and has_comma:
        last_dot = text.rfind(".")
        last_comma = text.rfind(",")
        if last_dot > last_comma:
            text = text.replace(",", "")
        else:
            text = text.replace(".", "").replace(",", ".")
    elif has_comma and not has_dot:
        after_comma = text[text.rfind(",") + 1 :]
        if len(after_comma) in (1, 2):
            text = text.replace(",", ".")
        else:
            text = text.replace(",", "")

    text = re.sub(r"[^\d.\-]", "", text)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Internal span builder — includes prediction-index tracking fields
# (_start, _end) used by the merge pass and stripped before returning.
# ---------------------------------------------------------------------------


def _make_span(label: str, tokens: list[dict], start_idx: int, end_idx: int) -> dict:
    x1 = min(t["bbox"][0] for t in tokens)
    y1 = min(t["bbox"][1] for t in tokens)
    x2 = max(t["bbox"][2] for t in tokens)
    y2 = max(t["bbox"][3] for t in tokens)
    return {
        "text": " ".join(t["text"] for t in tokens),
        "bbox": [x1, y1, x2, y2],
        "page": tokens[0]["page"],
        "confidence": min(t["confidence"] for t in tokens),
        "tokens": list(tokens),
        "_start": start_idx,
        "_end": end_idx,
    }


def _merge_two(a: dict, b: dict) -> dict:
    """Union-bbox merge of two spans; confidence is the minimum of the two."""
    return {
        "text": a["text"] + " " + b["text"],
        "bbox": [
            min(a["bbox"][0], b["bbox"][0]),
            min(a["bbox"][1], b["bbox"][1]),
            max(a["bbox"][2], b["bbox"][2]),
            max(a["bbox"][3], b["bbox"][3]),
        ],
        "page": a["page"],
        "confidence": min(a["confidence"], b["confidence"]),
        "tokens": a["tokens"] + b["tokens"],
        "_start": a["_start"],
        "_end": b["_end"],
    }


def _count_o_tokens(
    predictions: list[dict], after: int, before: int, threshold: float
) -> int:
    """Count tokens in predictions[after+1 : before] that are effectively O."""
    return sum(
        1
        for p in predictions[after + 1 : before]
        if p["label"] == "O" or p["confidence"] < threshold
    )


def _merge_nearby_spans(
    spans_by_label: dict,
    predictions: list[dict],
    confidence_threshold: float,
    gap_threshold: int = 3,
    y_proximity: int = 20,
) -> dict:
    """
    Second pass: merge consecutive same-label spans that have fewer than
    gap_threshold O-tagged tokens between them AND whose y-centres are within
    y_proximity units (0-1000 scale, so 20 ≈ 2% of page height).
    The y-proximity guard prevents table rows from collapsing across different
    rows even when they are adjacent in reading order with few O-tokens.
    """
    result: dict[str, list[dict]] = {}
    for label, spans in spans_by_label.items():
        if len(spans) < 2:
            result[label] = spans
            continue
        merged = [dict(spans[0])]
        for span in spans[1:]:
            prev = merged[-1]
            o_count = _count_o_tokens(
                predictions, prev["_end"], span["_start"], confidence_threshold
            )
            y_a = (prev["bbox"][1] + prev["bbox"][3]) / 2
            y_b = (span["bbox"][1] + span["bbox"][3]) / 2
            if o_count < gap_threshold and abs(y_a - y_b) <= y_proximity:
                merged[-1] = _merge_two(prev, span)
            else:
                merged.append(dict(span))
        result[label] = merged
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def aggregate_spans(
    predictions: list[dict], confidence_threshold: float = 0.5
) -> dict:
    """
    Group consecutive B-X / I-X predictions into field spans, then do a
    second pass merging same-label spans separated by fewer than 3 O-tagged
    tokens.

    Returns dict mapping entity label to a list of extraction dicts:
    {text, bbox, page, confidence, tokens}.
    Orphan I-tags (no preceding B-tag for the same label) start a new span.
    """
    spans_by_label: dict[str, list[dict]] = {}
    current_label: str | None = None
    current_tokens: list[dict] = []
    current_start_idx: int = 0
    current_end_idx: int = 0

    def _flush(up_to_idx: int) -> None:
        nonlocal current_label, current_tokens
        if current_label and current_tokens:
            span = _make_span(current_label, current_tokens, current_start_idx, up_to_idx)
            spans_by_label.setdefault(current_label, []).append(span)
        current_label = None
        current_tokens = []

    for idx, pred in enumerate(predictions):
        raw_label = pred["label"]
        conf = pred["confidence"]
        effective = raw_label if conf >= confidence_threshold else "O"

        if effective == "O":
            _flush(current_end_idx)

        elif effective.startswith("B-"):
            _flush(current_end_idx)
            current_label = effective[2:]
            current_tokens = [pred]
            current_start_idx = idx
            current_end_idx = idx

        elif effective.startswith("I-"):
            entity = effective[2:]
            if current_label == entity:
                current_tokens.append(pred)
                current_end_idx = idx
            else:
                # Orphan I-tag — promote to new span
                _flush(current_end_idx)
                current_label = entity
                current_tokens = [pred]
                current_start_idx = idx
                current_end_idx = idx

    _flush(current_end_idx)

    # Merge pass
    spans_by_label = _merge_nearby_spans(
        spans_by_label, predictions, confidence_threshold, gap_threshold=3
    )

    # Strip internal tracking fields
    for spans in spans_by_label.values():
        for span in spans:
            span.pop("_start", None)
            span.pop("_end", None)

    return spans_by_label


def cluster_line_items(
    spans_by_label: dict, y_tolerance: int = 15
) -> list[dict]:
    """
    Group line item spans into rows by y-centre proximity.

    Returns list of {description, quantity, unit_price, amount}.
    """
    LINE_FIELD_MAP = {
        "LINE_DESCRIPTION": "description",
        "LINE_QUANTITY": "quantity",
        "LINE_UNIT_PRICE": "unit_price",
        "LINE_AMOUNT": "amount",
    }

    tagged: list[dict] = []
    for label, field in LINE_FIELD_MAP.items():
        for span in spans_by_label.get(label, []):
            y_centre = (span["bbox"][1] + span["bbox"][3]) / 2
            tagged.append({"field": field, "y_centre": y_centre, "span": span})

    if not tagged:
        return []

    tagged.sort(key=lambda s: s["y_centre"])

    rows: list[list[dict]] = [[tagged[0]]]
    for item in tagged[1:]:
        if item["y_centre"] - rows[-1][-1]["y_centre"] <= y_tolerance:
            rows[-1].append(item)
        else:
            rows.append([item])

    result: list[dict] = []
    for row in rows:
        best: dict[str, dict] = {}
        for item in row:
            field = item["field"]
            span = item["span"]
            if field not in best or span["bbox"][0] < best[field]["bbox"][0]:
                best[field] = span

        row_dict: dict[str, object] = {
            "description": None,
            "quantity": None,
            "unit_price": None,
            "amount": None,
        }
        for field, span in best.items():
            text = span["text"]
            if field == "description":
                row_dict["description"] = text
            else:
                row_dict[field] = _parse_money(text)

        result.append(row_dict)

    return result
