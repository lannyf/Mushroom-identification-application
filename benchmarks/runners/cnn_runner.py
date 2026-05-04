"""Benchmark runner for the CNN (EfficientNet-B3) classifier.

Loads the trained ``MushroomCNN`` model, runs inference on raw image
bytes, and maps the model's English output names to canonical
``species_id`` values.
"""

import time
from typing import Optional

from benchmarks.config import CNN_NAME_TO_SPECIES_ID
from benchmarks.runners.base import BenchmarkRunner, RunnerResult


class CNNRunner(BenchmarkRunner):
    """Wrapper around ``models.cnn_classifier.MushroomCNN``.

    Returns the top-5 predictions as ranked ``(species_id, confidence)``
    tuples. If the model weights are missing, ``coverage`` is set to
    False and an error message is returned.
    """

    name = "cnn"

    def __init__(self):
        from models.cnn_classifier import MushroomCNN

        self.cnn = MushroomCNN()

    def predict(self, sample) -> RunnerResult:
        """Run CNN inference and map output names to species_ids.

        Args:
            sample: ``BenchmarkSample`` with ``image_bytes``.

        Returns:
            ``RunnerResult`` with top-5 predictions or an error if the
            model is unavailable.
        """
        if not self.cnn.is_trained:
            return RunnerResult(
                method_name="cnn",
                predictions=[],
                coverage=False,
                error="CNN weights not found at artifacts/cnn_weights.pt",
            )

        t0 = time.perf_counter()
        scores = self.cnn.predict(sample.image_bytes)
        elapsed = (time.perf_counter() - t0) * 1000

        if scores is None:
            return RunnerResult(
                method_name="cnn",
                predictions=[],
                coverage=False,
                error="CNN inference failed",
                inference_time_ms=elapsed,
            )

        # Map English model output names → canonical species_ids and sort.
        mapped = [
            (CNN_NAME_TO_SPECIES_ID[name], conf)
            for name, conf in scores.items()
            if name in CNN_NAME_TO_SPECIES_ID
        ]
        mapped.sort(key=lambda x: x[1], reverse=True)

        return RunnerResult(
            method_name="cnn",
            predictions=mapped,
            coverage=True,
            inference_time_ms=elapsed,
        )
