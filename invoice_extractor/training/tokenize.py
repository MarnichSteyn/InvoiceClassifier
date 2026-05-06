from __future__ import annotations

from PIL import Image
from transformers import LayoutLMv3Processor


def _dummy_image() -> Image.Image:
    return Image.new("RGB", (224, 224), color=(255, 255, 255))


def tokenize_and_align_labels(
    examples: dict,
    processor: LayoutLMv3Processor,
    max_length: int = 512,
) -> dict:
    """
    Tokenizes word-level tokens into subwords and aligns bboxes and labels.

    Input columns: tokens (list[str]), bboxes (list[list[int]] in 0-1000 scale),
    labels (list[int]).

    For each subword:
      - bbox inherits the source word's bbox
      - label is the source word's label for the FIRST subword, -100 for the rest
      - special tokens (CLS, SEP, PAD) get -100

    Returns processor output dict with input_ids, attention_mask, bbox,
    pixel_values, and aligned labels.
    """
    images = [_dummy_image() for _ in examples["tokens"]]

    encoding = processor(
        images=images,
        text=examples["tokens"],
        boxes=examples["bboxes"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors=None,
    )

    aligned_labels: list[list[int]] = []
    for i, word_labels in enumerate(examples["labels"]):
        word_ids = encoding.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids: list[int] = []
        for word_idx in word_ids:
            if word_idx is None:                  # special tokens (CLS, SEP, PAD)
                label_ids.append(-100)
            elif word_idx != previous_word_idx:   # first subword of a word
                label_ids.append(word_labels[word_idx])
            else:                                  # subsequent subwords
                label_ids.append(-100)
            previous_word_idx = word_idx
        aligned_labels.append(label_ids)

    result = dict(encoding)
    result["labels"] = aligned_labels
    return result
