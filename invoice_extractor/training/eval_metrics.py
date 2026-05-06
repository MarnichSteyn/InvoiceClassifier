from __future__ import annotations

import numpy as np
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2


def compute_metrics(eval_pred, id_to_label: dict[int, str]) -> dict[str, float]:
    """
    eval_pred is (logits, labels) where logits are (B, T, C) and labels are
    (B, T) with -100 marking positions to ignore.

    Returns a flat dict with overall and per-class precision, recall, f1,
    and per-class support. Keys like eval_f1, eval_f1_INVOICE_NUMBER, etc.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)  # (B, T)

    true_seqs: list[list[str]] = []
    pred_seqs: list[list[str]] = []

    for pred_row, label_row in zip(predictions, labels):
        true_seq: list[str] = []
        pred_seq: list[str] = []
        for pred_id, label_id in zip(pred_row, label_row):
            if label_id == -100:
                continue
            true_seq.append(id_to_label[int(label_id)])
            pred_seq.append(id_to_label[int(pred_id)])
        true_seqs.append(true_seq)
        pred_seqs.append(pred_seq)

    report: dict = classification_report(
        true_seqs, pred_seqs, mode="strict", scheme=IOB2, output_dict=True, zero_division=0
    )

    _skip = {"macro avg", "micro avg", "weighted avg"}

    result: dict[str, float] = {
        "eval_precision": report["macro avg"]["precision"],
        "eval_recall": report["macro avg"]["recall"],
        "eval_f1": report["macro avg"]["f1-score"],
    }

    for entity_name, metrics in report.items():
        if entity_name in _skip or not isinstance(metrics, dict):
            continue
        safe = entity_name.replace("-", "_").replace(" ", "_")
        result[f"eval_f1_{safe}"] = metrics["f1-score"]
        result[f"eval_precision_{safe}"] = metrics["precision"]
        result[f"eval_recall_{safe}"] = metrics["recall"]
        result[f"eval_support_{safe}"] = metrics["support"]

    return result
