"""Benchmark runner for the LLM (Ollama) classifier.

Converts the 5 visually extracted traits into a natural-language
observation, sends it to the LLM, and resolves the returned free-text
species name to a canonical ``species_id`` via a comprehensive
English/Swedish/scientific name lookup.

Uses the shared LLM cache so multimodal runners do not trigger
duplicate Ollama calls.
"""

import csv
import time
from typing import Optional

from benchmarks.config import CNN_NAME_TO_SPECIES_ID, SPECIES_CSV
from benchmarks.runners.base import BenchmarkRunner, RunnerResult
from benchmarks.runners._extract_cache import extract


def _load_name_mappings():
    """Build comprehensive English/Swedish/Scientific name → species_id mappings from species.csv."""
    mappings = {}
    with open(SPECIES_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            sid = row["species_id"]
            # Map all name variants
            for key in ("english_name", "swedish_name", "scientific_name"):
                name = row.get(key, "").strip()
                if name:
                    mappings[name.lower()] = sid
    return mappings


_NAME_MAPPINGS = _load_name_mappings()


def _resolve_species_name(name: str) -> str:
    """Resolve an LLM-output species name to a species_id.

    Tries exact match, then stripping parentheticals, then token matching.
    Falls back to CNN_NAME_TO_SPECIES_ID for legacy names.
    """
    if not name or name in ("Unknown", "Error", "Unable to parse"):
        return "unknown"

    # 1. Direct lookup in CNN mapping (legacy)
    if name in CNN_NAME_TO_SPECIES_ID:
        return CNN_NAME_TO_SPECIES_ID[name]

    normalized = name.lower().strip()

    # 2. Direct lookup in full name mappings
    if normalized in _NAME_MAPPINGS:
        return _NAME_MAPPINGS[normalized]

    # 3. Strip parentheticals: "Hedgehog Mushroom (Blek taggsvamp)" → "hedgehog mushroom"
    if "(" in normalized:
        without_paren = normalized.split("(")[0].strip()
        if without_paren in _NAME_MAPPINGS:
            return _NAME_MAPPINGS[without_paren]

    # 4. Check if any known name is a substring of the output
    for known_name, sid in _NAME_MAPPINGS.items():
        if known_name in normalized or normalized in known_name:
            return sid

    # 5. Fallback: return the raw name so it's visible in logs
    return name


def _traits_to_observation(visible_traits: dict) -> str:
    """Format the 5 visual traits as a natural-language observation for the LLM prompt."""
    return (
        f"Mushroom observation:\n"
        f"- Cap colour: {visible_traits['dominant_color']}"
        f" (secondary: {visible_traits['secondary_color']})\n"
        f"- Cap shape: {visible_traits['cap_shape']}\n"
        f"- Surface texture: {visible_traits['surface_texture']}\n"
        f"- Has ridges: {visible_traits['has_ridges']}\n"
        f"- Brightness: {visible_traits['brightness']}\n"
    )


class LLMRunner(BenchmarkRunner):
    """Wrapper around ``models.llm_classifier.LLMClassifier``.

    The classifier is instantiated with the Ollama backend if available.
    If Ollama is not running, all predictions return ``coverage=False``.
    """

    name = "llm"

    def __init__(self):
        from models.llm_classifier import LLMClassifier, OllamaBackend

        if OllamaBackend.is_available():
            try:
                self.classifier = LLMClassifier(backend_type="ollama")
            except Exception:
                self.classifier = None
        else:
            self.classifier = None

    def predict(self, sample) -> RunnerResult:
        """Classify a sample via LLM with caching.

        Args:
            sample: ``BenchmarkSample`` with ``image_bytes``.

        Returns:
            ``RunnerResult`` with the LLM's top prediction, or an error
            if the backend is unavailable.
        """
        if self.classifier is None:
            return RunnerResult(
                method_name="llm",
                predictions=[],
                coverage=False,
                error="LLM backend (Ollama) not available",
            )

        from benchmarks.runners._llm_cache import get_cached, set_cached
        import logging

        logger = logging.getLogger(__name__)

        # Return cached result if the same image was already processed.
        cached = get_cached(sample.image_bytes)
        if cached is not None:
            return RunnerResult(
                method_name="llm",
                predictions=[(cached["species_id"], cached["confidence"])],
                coverage=cached["coverage"],
                inference_time_ms=0.0,
                error=cached.get("error"),
                metadata={"reasoning": cached.get("reasoning", ""), "cached": True},
            )

        logger.info("LLM processing %s (%s) ...", sample.species_id, sample.image_path.name)
        t0 = time.perf_counter()
        visible_traits = extract(sample.image_bytes)["visible_traits"]
        observation = _traits_to_observation(visible_traits)

        try:
            result = self.classifier.classify(observation=observation)
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info("LLM result for %s: %s (%.1fms)", sample.species_id, result.top_species, elapsed)

            species_name = result.top_species
            species_id = _resolve_species_name(species_name)

            # Cache successful result for reuse by multimodal runners.
            set_cached(
                sample.image_bytes,
                {
                    "species_id": species_id,
                    "confidence": result.top_confidence,
                    "coverage": True,
                    "reasoning": result.reasoning,
                },
            )

            return RunnerResult(
                method_name="llm",
                predictions=[(species_id, result.top_confidence)],
                coverage=True,
                inference_time_ms=elapsed,
                metadata={"reasoning": result.reasoning},
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            # Cache failure so we don't retry the same slow call.
            set_cached(
                sample.image_bytes,
                {
                    "species_id": "unknown",
                    "confidence": 0.0,
                    "coverage": False,
                    "error": str(exc),
                },
            )
            return RunnerResult(
                method_name="llm",
                predictions=[],
                coverage=False,
                error=str(exc),
                inference_time_ms=elapsed,
            )
