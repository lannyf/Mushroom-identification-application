"""Benchmark runner for multimodal fusion strategies.

Uses **FinalAggregator** (``strategy="final_aggregator"``) — the original
hierarchical three-step aggregator (CNN → tree → trait DB validation).

Sub-runners are instantiated internally so the benchmark engine sees a
single uniform interface.
"""

import time
from typing import Dict, Any

from benchmarks.config import DATA_RAW_DIR, SPECIES_CSV, SPECIES_ID_TO_CNN_NAME
from benchmarks.runners.base import BenchmarkRunner, RunnerResult
from benchmarks.runners.cnn_runner import CNNRunner
from benchmarks.runners.trait_db_runner import TraitDBRunner
from benchmarks.runners.tree_runner import TreeRunner


class MultimodalRunner(BenchmarkRunner):
    """Orchestrates sub-runners and feeds their outputs into FinalAggregator.

    Because the sub-runners use the shared extraction cache,
    duplicate YOLO/CV work is eliminated automatically.
    """

    name = "multimodal"

    def __init__(self, strategy: str = "final_aggregator"):
        self.strategy = strategy
        self.cnn_runner = CNNRunner()
        self.tree_runner = TreeRunner(mode="oracle")
        self.trait_db_runner = TraitDBRunner()

        if strategy == "final_aggregator":
            from models.final_aggregator import FinalAggregator

            self.aggregator = FinalAggregator(str(SPECIES_CSV))
        else:
            raise ValueError(
                f"Unsupported strategy: {strategy!r}. "
                'Only "final_aggregator" is supported.'
            )

        # Used for Step 3 validation in the FinalAggregator pipeline.
        from models.trait_database_comparator import TraitDatabaseComparator

        self._trait_comparator = TraitDatabaseComparator(str(DATA_RAW_DIR))

    def predict(self, sample) -> RunnerResult:
        """Run the FinalAggregator fusion strategy on a single sample.

        Args:
            sample: ``BenchmarkSample`` with ``image_bytes``.

        Returns:
            ``RunnerResult`` with the fused top prediction and timing
            that includes all sub-runner work.
        """
        t0 = time.perf_counter()
        cnn_res = self.cnn_runner.predict(sample)
        tree_res = self.tree_runner.predict(sample)
        trait_res = self.trait_db_runner.predict(sample)

        # Three-step hierarchical aggregation.
        step1 = self._build_step1(cnn_res, sample)
        step2 = self._build_step2(tree_res)
        step3 = self._build_step3(tree_res, trait_res)
        final = self.aggregator.aggregate(step1, step2, step3)

        rec = final["final_recommendation"]
        top_id = rec["species_id"]
        conf = rec["overall_confidence"]
        agreement = final.get("method_agreement", "unknown")

        elapsed = (time.perf_counter() - t0) * 1000
        return RunnerResult(
            method_name=f"multimodal_{self.strategy}",
            predictions=[(top_id, conf)],
            coverage=True,
            inference_time_ms=elapsed,
            metadata={"method_agreement": agreement},
        )

    def _build_step1(self, cnn_res, sample):
        """Construct the Step 1 input for FinalAggregator.

        Merges the cached visual trait extraction with the CNN's top-k
        prediction, translating species_ids back to English names where
        needed.
        """
        from benchmarks.runners._extract_cache import extract

        step1 = extract(sample.image_bytes)
        top_species_english = SPECIES_ID_TO_CNN_NAME.get(
            cnn_res.top_species, cnn_res.top_species
        )
        step1["ml_prediction"] = {
            "top_species": top_species_english,
            "confidence": cnn_res.top_confidence,
            "top_k": [
                {"species": SPECIES_ID_TO_CNN_NAME.get(sp, sp), "confidence": conf}
                for sp, conf in cnn_res.predictions[:5]
            ],
            "reasoning": "CNN benchmark prediction",
            "method": "cnn",
        }
        return step1

    def _build_step2(self, tree_res):
        """Construct the Step 2 input for FinalAggregator from the tree result."""
        if not tree_res.coverage:
            return {"status": "question", "species": ""}
        return {
            "status": "conclusion",
            "species": tree_res.metadata.get("swedish_name", tree_res.top_species),
            "path": tree_res.metadata.get("path", []),
            "auto_answered": tree_res.metadata.get("auto_answered", []),
        }

    def _build_step3(self, tree_res, trait_res):
        """Construct the Step 3 input for FinalAggregator.

        Validates the tree's candidate species against the visually
        extracted traits using the trait-database comparator.
        """
        if not tree_res.coverage or not trait_res.coverage:
            return {"status": "species_not_found"}

        visible_traits = trait_res.metadata["visible_traits"]
        swedish_name = tree_res.metadata.get("swedish_name", tree_res.top_species)
        return self._trait_comparator.compare(swedish_name, visible_traits)
