from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def ocr_pdf_or_image(path: Path) -> list[dict]:
    """Run Tesseract on a PDF or image. Returns word list in internal format."""
    try:
        import pytesseract
        from pytesseract import Output
    except ImportError:
        raise ImportError("pytesseract is required: pip install pytesseract")

    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow is required: pip install pillow")

    path = Path(path)

    if path.suffix.lower() == ".pdf":
        try:
            from pdf2image import convert_from_path
        except ImportError:
            raise ImportError("pdf2image is required: pip install pdf2image")
        try:
            pages = convert_from_path(str(path), dpi=200)
        except Exception as exc:
            msg = str(exc).lower()
            if "poppler" in msg or "pdfinfo" in msg or "pdftoppm" in msg:
                raise RuntimeError(
                    "poppler not found. Install from "
                    "https://github.com/oschwartz10612/poppler-windows/releases/ "
                    "(Windows) or via your system package manager, "
                    "then add it to PATH."
                ) from exc
            raise
    else:
        from PIL import Image as _Image
        pages = [_Image.open(path).convert("RGB")]

    words: list[dict] = []
    for page_idx, image in enumerate(pages):
        W, H = image.size
        try:
            data = pytesseract.image_to_data(image, output_type=Output.DICT)
        except pytesseract.TesseractNotFoundError:
            raise RuntimeError(
                "Tesseract not found. Install from "
                "https://github.com/UB-Mannheim/tesseract/wiki (Windows) "
                "or via your system package manager, then add it to PATH."
            )

        n = len(data["text"])
        for i in range(n):
            text = data["text"][i]
            if not text or not text.strip():
                continue
            conf = data["conf"][i]
            if conf == -1:
                continue
            conf_float = float(conf) / 100.0
            if conf_float < 0.30:
                continue
            left = data["left"][i]
            top = data["top"][i]
            width = data["width"][i]
            height = data["height"][i]
            x1 = min(1000, max(0, int(round(left / W * 1000))))
            y1 = min(1000, max(0, int(round(top / H * 1000))))
            x2 = min(1000, max(0, int(round((left + width) / W * 1000))))
            y2 = min(1000, max(0, int(round((top + height) / H * 1000))))
            if x1 >= x2 or y1 >= y2:
                continue
            words.append(
                {
                    "text": text.strip(),
                    "bbox": [x1, y1, x2, y2],
                    "page": page_idx,
                    "confidence": conf_float,
                }
            )

    return words


def load_docile_ocr(path: Path) -> list[dict]:
    """Load pre-computed DocILE OCR JSON. Returns word list in internal format."""
    from invoice_extractor.data.docile_loader import load_ocr_words

    raw = load_ocr_words(Path(path))
    words: list[dict] = []
    for w in raw:
        x1, y1, x2, y2 = w["bbox_norm"]
        bbox_int = [
            min(1000, max(0, int(round(x1 * 1000)))),
            min(1000, max(0, int(round(y1 * 1000)))),
            min(1000, max(0, int(round(x2 * 1000)))),
            min(1000, max(0, int(round(y2 * 1000)))),
        ]
        words.append(
            {
                "text": w["text"],
                "bbox": bbox_int,
                "page": w["page"],
                "confidence": float(w.get("confidence", 1.0)),
            }
        )
    return words
