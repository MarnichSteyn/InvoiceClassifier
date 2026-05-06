from __future__ import annotations

import logging
from pathlib import Path

import torch
from PIL import Image

logger = logging.getLogger(__name__)

_DUMMY_IMAGE = Image.new("RGB", (224, 224), color=(255, 255, 255))


class InvoiceExtractor:
    def __init__(self, checkpoint_dir: Path) -> None:
        """Load model + processor from disk. Move to GPU if available."""
        from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

        checkpoint_dir = Path(checkpoint_dir)
        self.processor = LayoutLMv3Processor.from_pretrained(
            str(checkpoint_dir), apply_ocr=False
        )
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(str(checkpoint_dir))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, words: list[dict]) -> list[dict]:
        """
        Run model inference. Returns one entry per input word:
            {text, bbox, page, label, confidence}
        """
        if not words:
            return []

        texts = [w["text"] for w in words]
        boxes = [w["bbox"] for w in words]

        encoding = self.processor(
            images=_DUMMY_IMAGE,
            text=texts,
            boxes=boxes,
            truncation=True,
            padding="max_length",
            max_length=384,
            return_tensors="pt",
        )

        word_ids = encoding.word_ids(batch_index=0)
        max_word_in_encoding = max(
            (wid for wid in word_ids if wid is not None), default=-1
        )
        if max_word_in_encoding < len(words) - 1:
            logger.warning(
                "Input truncated: %d words provided, only %d fit in max_length=384",
                len(words),
                max_word_in_encoding + 1,
            )

        inputs = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits  # (1, seq_len, num_labels)
        probs = torch.softmax(logits, dim=-1)[0]  # (seq_len, num_labels)
        pred_ids = probs.argmax(dim=-1)  # (seq_len,)
        pred_confs = probs.max(dim=-1).values  # (seq_len,)

        # First subword per word wins
        word_preds: dict[int, tuple[str, float]] = {}
        id2label = self.model.config.id2label
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx in word_preds:
                continue
            label = id2label[pred_ids[token_idx].item()]
            conf = pred_confs[token_idx].item()
            word_preds[word_idx] = (label, conf)

        results: list[dict] = []
        for i, word in enumerate(words):
            label, conf = word_preds.get(i, ("O", 0.0))
            results.append(
                {
                    "text": word["text"],
                    "bbox": word["bbox"],
                    "page": word["page"],
                    "label": label,
                    "confidence": conf,
                }
            )

        return results
