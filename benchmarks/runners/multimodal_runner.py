"""Benchmark runner for multimodal fusion strategies.

Supports two very different fusion pipelines:

1. **FinalAggregator** (``strategy="final_aggregator"``) — the original
   hierarchical three-step aggregator (CNN → tree → trait DB validation).
2. **HybridClassifier** (``strategy="weighted"|"geometric"|"voting"``) —
   late fusion of CNN, trait DB, and LLM predictions via weighted average,
   geometric mean, or majority voting.

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
from benchmarks.runners.llm_runner import LLMRunner


class MultimodalRunner(BenchmarkRunner):
    """Orchestrates sub-runners and feeds their outputs into a fusion strategy.

    Because the sub-runners use the shared extraction and LLM caches,
    duplicate YOLO/CV work and duplicate Ollama calls are eliminated
    automatically.
    """

    name = "multimodal"

    def __init__(self, strategy: str = "final_aggregator"):
        self.strategy = strategy
        self.cnn_runner = CNNRunner()
        self.tree_runner = TreeRunner(mode="oracle")
        self.trait_db_runner = TraitDBRunner()
        self.llm_runner = LLMRunner()

        if strategy == "final_aggregator":
            from models.final_aggregator import FinalAggregator

            self.aggregator = FinalAggregator(str(SPECIES_CSV))
        else:
            from models.hybrid_classifier import HybridClassifier, AggregationMethod

            method_map = {
                "weighted": AggregationMethod.WEIGHTED_AVERAGE,
                "geometric": AggregationMethod.GEOMETRIC_MEAN,
                "voting": AggregationMethod.VOTING,
            }
            method = method_map.get(strategy, AggregationMethod.WEIGHTED_AVERAGE)
            self.aggregator = HybridClassifier(aggregation_method=method)

        # Used for Step 3 validation in the FinalAggregator pipeline.
        from models.trait_database_comparator import TraitDatabaseComparator

        self._trait_comparator = TraitDatabaseComparator(str(DATA_RAW_DIR))

    def predict(self, sample) -> RunnerResult:
        """Run the selected fusion strategy on a single sample.

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

        if self.strategy == "final_aggregator":
            # Three-step hierarchical aggregation.
            step1 = self._build_step1(cnn_res, sample)
            step2 = self._build_step2(tree_res)
            step3 = self._build_step3(tree_res, trait_res)
            final = self.aggregator.aggregate(step1, step2, step3)

            rec = final["final_recommendation"]
            top_id = rec["species_id"]
            conf = rec["overall_confidence"]
            agreement = final.get("method_agreement", "unknown")
        else:
            # Late fusion of CNN + trait DB + LLM via HybridClassifier.
            from models.hybrid_classifier import MethodPrediction

            llm_res = self.llm_runner.predict(sample)

            image_prediction = None
            trait_prediction = None
            llm_prediction = None

            if cnn_res.coverage and cnn_res.predictions:
                image_prediction = MethodPrediction(
                    method="image",
                    species=cnn_res.top_species,
                    confidence=cnn_res.top_confidence,
                    top_k=cnn_res.predictions[:5],
                )
            if trait_res.coverage and trait_res.predictions:
                trait_prediction = MethodPrediction(
                    method="trait",
                    species=trait_res.top_species,
                    confidence=trait_res.top_confidence,
                    top_k=trait_res.predictions[:5],
                )
            if llm_res.coverage and llm_res.predictions:
                llm_prediction = MethodPrediction(
                    method="llm",
                    species=llm_res.top_species,
                    confidence=llm_res.top_confidence,
                    top_k=llm_res.predictions[:5],
                )

            result = self.aggregator.classify(
                image_prediction=image_prediction,
                trait_prediction=trait_prediction,
                llm_prediction=llm_prediction,
            )
            top_id = result.top_species
            conf = result.top_confidence
            agreement = "full" if result.consensus_strength == 1.0 else "partial"

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
