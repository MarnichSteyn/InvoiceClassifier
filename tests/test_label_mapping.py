import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from invoice_extractor.data.label_mapping import (
    DOCILE_TO_TARGET,
    TARGET_LABELS,
    bio_labels,
    id_to_label,
    label_to_id,
)
from invoice_extractor.data.docile_loader import assign_labels, compute_containment, compute_iou


def test_bio_labels_starts_with_O():
    labels = bio_labels()
    assert labels[0] == "O"


def test_bio_labels_length():
    labels = bio_labels()
    assert len(labels) == 1 + 2 * len(TARGET_LABELS)


def test_bio_labels_structure():
    labels = bio_labels()
    # Every target label should have a B- and I- variant
    for target in TARGET_LABELS:
        assert f"B-{target}" in labels
        assert f"I-{target}" in labels


def test_label_roundtrip():
    for label, idx in label_to_id.items():
        assert id_to_label[idx] == label


def test_id_roundtrip():
    for idx, label in id_to_label.items():
        assert label_to_id[label] == idx


def test_all_target_values_in_target_labels():
    for v in DOCILE_TO_TARGET.values():
        assert v in TARGET_LABELS


def test_compute_iou_identical():
    box = [0.1, 0.1, 0.5, 0.5]
    assert compute_iou(box, box) == 1.0


def test_compute_iou_no_overlap():
    a = [0.0, 0.0, 0.2, 0.2]
    b = [0.5, 0.5, 0.9, 0.9]
    assert compute_iou(a, b) == 0.0


def test_compute_iou_partial_overlap():
    # a = [0, 0, 1, 1], b = [0.5, 0, 1.5, 1]
    # intersection: [0.5, 0, 1, 1] = 0.5 * 1 = 0.5
    # area_a = 1.0, area_b = 1.0, union = 1.5
    # iou = 0.5 / 1.5 = 1/3
    a = [0.0, 0.0, 1.0, 1.0]
    b = [0.5, 0.0, 1.5, 1.0]
    iou = compute_iou(a, b)
    assert abs(iou - 1 / 3) < 1e-9


def test_compute_containment_identical():
    box = [0.1, 0.1, 0.5, 0.5]
    assert compute_containment(box, box) == 1.0


def test_compute_containment_fully_inside():
    inner = [0.2, 0.2, 0.4, 0.4]
    outer = [0.0, 0.0, 1.0, 1.0]
    assert compute_containment(inner, outer) == 1.0


def test_compute_containment_half_outside():
    # inner [0,0,1,1], outer [0.5,0,1.5,1]: intersection [0.5,0,1,1] = 0.5 of inner area
    inner = [0.0, 0.0, 1.0, 1.0]
    outer = [0.5, 0.0, 1.5, 1.0]
    assert abs(compute_containment(inner, outer) - 0.5) < 1e-9


def test_compute_containment_no_overlap():
    inner = [0.0, 0.0, 0.2, 0.2]
    outer = [0.5, 0.5, 0.9, 0.9]
    assert compute_containment(inner, outer) == 0.0


def test_compute_containment_zero_area_inner():
    inner = [0.3, 0.3, 0.3, 0.5]  # zero width
    outer = [0.0, 0.0, 1.0, 1.0]
    assert compute_containment(inner, outer) == 0.0


def test_assign_labels_single_annotation():
    # 3 words arranged horizontally; annotation covers the middle word exactly
    words = [
        {"text": "hello", "bbox_norm": [0.0, 0.0, 0.1, 0.1], "page": 0},
        {"text": "world", "bbox_norm": [0.2, 0.0, 0.3, 0.1], "page": 0},
        {"text": "foo",   "bbox_norm": [0.5, 0.0, 0.6, 0.1], "page": 0},
    ]
    annotations = [
        {
            "target_label": "X",
            "page": 0,
            "bbox_norm": [0.2, 0.0, 0.3, 0.1],
        }
    ]
    result = assign_labels(words, annotations, containment_threshold=0.5)
    assert result == ["O", "B-X", "O"]


def test_assign_labels_two_words_same_annotation():
    words = [
        {"text": "a", "bbox_norm": [0.0, 0.0, 0.5, 0.1], "page": 0},
        {"text": "b", "bbox_norm": [0.5, 0.0, 1.0, 0.1], "page": 0},
    ]
    # annotation spans both words; each word is fully contained
    annotations = [
        {
            "target_label": "VENDOR_NAME",
            "page": 0,
            "bbox_norm": [0.0, 0.0, 1.0, 0.1],
        }
    ]
    result = assign_labels(words, annotations, containment_threshold=0.4)
    assert result[0] == "B-VENDOR_NAME"
    assert result[1] == "I-VENDOR_NAME"


def test_assign_labels_different_pages():
    words = [
        {"text": "w1", "bbox_norm": [0.1, 0.1, 0.4, 0.2], "page": 0},
        {"text": "w2", "bbox_norm": [0.1, 0.1, 0.4, 0.2], "page": 1},
    ]
    annotations = [
        {"target_label": "TOTAL", "page": 0, "bbox_norm": [0.1, 0.1, 0.4, 0.2]}
    ]
    result = assign_labels(words, annotations, containment_threshold=0.5)
    assert result == ["B-TOTAL", "O"]
