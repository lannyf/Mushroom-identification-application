"""Shared LLM result cache for benchmark runners.

The standalone LLM runner and the three hybrid strategies
(weighted, geometric, voting) all need the same LLM prediction.
This module caches Ollama responses keyed by image SHA-256 so the
slow LLM call (~30 s on CPU) is made at most once per image.
"""

import hashlib
from typing import Any, Dict, Optional

# In-memory cache: sha256_hex -> dict with species_id, confidence, etc.
_llm_cache: Dict[str, Any] = {}


def get_cached(image_bytes: bytes) -> Optional[Any]:
    """Retrieve a previously cached LLM result, if any."""
    h = hashlib.sha256(image_bytes).hexdigest()
    return _llm_cache.get(h)


def set_cached(image_bytes: bytes, result: Any) -> None:
    """Store an LLM result so other runners can reuse it."""
    h = hashlib.sha256(image_bytes).hexdigest()
    _llm_cache[h] = result
