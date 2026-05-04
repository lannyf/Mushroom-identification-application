"""Shared trait-extraction cache for benchmark runners.

Visual trait extraction (YOLO + OpenCV) is expensive and is invoked
by multiple runners (tree, trait_db, LLM, multimodal). This module
caches results keyed by image SHA-256 so each image is analysed only
once per benchmark run.
"""

import hashlib

from models.visual_trait_extractor import extract as _original_extract

# In-memory cache: sha256_hex -> extraction_result dict.
_extract_cache: dict = {}


def extract(image_bytes: bytes) -> dict:
    """Return cached visual traits or extract them if not yet cached.

    Args:
        image_bytes: Raw JPEG/PNG bytes.

    Returns:
        Dictionary with keys such as ``visible_traits`` and ``bounding_boxes``.
    """
    h = hashlib.sha256(image_bytes).hexdigest()
    if h not in _extract_cache:
        _extract_cache[h] = _original_extract(image_bytes)
    return _extract_cache[h]
