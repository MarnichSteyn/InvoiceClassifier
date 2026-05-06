"""Enables: python -m invoice_extractor extract ..."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path so cli.py is importable
_root = str(Path(__file__).parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from cli import main  # noqa: E402

main()
