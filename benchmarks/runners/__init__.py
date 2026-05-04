"""Benchmark runners — one wrapper per identification method.

Each runner adapts a model from ``models/`` to the common
``BenchmarkRunner`` interface so the benchmark engine can treat
them uniformly.
"""

from benchmarks.runners.base import BenchmarkRunner, RunnerResult
from benchmarks.runners.cnn_runner import CNNRunner
from benchmarks.runners.trait_db_runner import TraitDBRunner
from benchmarks.runners.tree_runner import TreeRunner
from benchmarks.runners.llm_runner import LLMRunner
from benchmarks.runners.multimodal_runner import MultimodalRunner

__all__ = [
    "BenchmarkRunner",
    "RunnerResult",
    "CNNRunner",
    "TraitDBRunner",
    "TreeRunner",
    "LLMRunner",
    "MultimodalRunner",
]
