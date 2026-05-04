"""Benchmark runner for the trait-database comparator.

Scores every species in the database against the visually extracted
traits and returns the full ranked list.
"""

import time

from benchmarks.config import DATA_RAW_DIR
from benchmarks.runners.base import BenchmarkRunner, RunnerResult
from benchmarks.runners._extract_cache import extract


class TraitDBRunner(BenchmarkRunner):
    """Wrapper around ``models.trait_database_comparator.TraitDatabaseComparator``.

    Uses the shared extraction cache so YOLO/CV work is not repeated
    when this runner is invoked alongside tree or multimodal runners.
    """

    name = "trait_db"

    def __init__(self):
        from models.trait_database_comparator import TraitDatabaseComparator

        self.comparator = TraitDatabaseComparator(str(DATA_RAW_DIR))

    def predict(self, sample) -> RunnerResult:
        """Rank all 50 species by trait-match score.

        Args:
            sample: ``BenchmarkSample`` with ``image_bytes``.

        Returns:
            ``RunnerResult`` where ``predictions`` contains all 50 species
            sorted by descending match score.
        """
        t0 = time.perf_counter()
        visible_traits = extract(sample.image_bytes)["visible_traits"]
        ranked = self.comparator.rank_all_species(visible_traits)
        elapsed = (time.perf_counter() - t0) * 1000

        predictions = [(r["species_id"], r["score"]) for r in ranked]
        return RunnerResult(
            method_name="trait_db",
            predictions=predictions,
            coverage=True,
            inference_time_ms=elapsed,
            metadata={"visible_traits": visible_traits},
        )
