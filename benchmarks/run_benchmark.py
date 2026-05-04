#!/usr/bin/env python3
"""Benchmark runner for the mushroom identification system.

Orchestrates the entire evaluation pipeline:

1. Load the evaluation dataset (``GroundTruthDataset``).
2. Run each selected method (CNN, tree, trait DB, LLM, multimodal).
3. Compute metrics: top-k accuracy, coverage, F1, ECE, bootstrap CIs.
4. Generate reports: JSON, CSV, Markdown.
5. Generate visualisations: bar charts, heat-maps, confidence histograms.

Usage::

    python -m benchmarks.run_benchmark [--methods all] [--subset all]

"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

from benchmarks.dataset import GroundTruthDataset
from benchmarks.metrics import (
    bootstrap_ci,
    compute_coverage,
    compute_ece,
    compute_macro_f1,
    compute_mean_inference_time,
    compute_ood_metrics,
    compute_pairwise_agreement,
    compute_top_k_accuracy,
)
from benchmarks.reports import generate_csv_report, generate_json_report, generate_markdown_report
from benchmarks.runners.base import RunnerResult
from benchmarks.runners.cnn_runner import CNNRunner
from benchmarks.runners.llm_runner import LLMRunner
from benchmarks.runners.multimodal_runner import MultimodalRunner
from benchmarks.runners.trait_db_runner import TraitDBRunner
from benchmarks.runners.tree_runner import TreeRunner
from benchmarks.visualize import (
    plot_accuracy_comparison,
    plot_agreement_heatmap,
    plot_confidence_distribution,
    plot_ood_analysis,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Registry of all benchmarkable methods. Factories are used when a
# runner needs constructor arguments (e.g. tree mode or hybrid strategy).
METHODS = {
    "cnn": CNNRunner,
    "tree_auto": lambda: TreeRunner(mode="auto"),
    "tree_oracle": lambda: TreeRunner(mode="oracle"),
    "trait_db": TraitDBRunner,
    "llm": LLMRunner,
    "multimodal_final": lambda: MultimodalRunner(strategy="final_aggregator"),
}


def run():
    parser = argparse.ArgumentParser(description="Run benchmark suite")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results"),
        help="Directory to write results",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(METHODS.keys()) + ["all"],
        default=["all"],
        help="Methods to benchmark",
    )
    parser.add_argument(
        "--subset",
        choices=["all", "in_distribution", "out_of_distribution"],
        default="all",
        help="Which image subset to evaluate",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------------------
    dataset = GroundTruthDataset()
    if args.subset == "in_distribution":
        samples = list(dataset.in_distribution())
    elif args.subset == "out_of_distribution":
        samples = list(dataset.out_of_distribution())
    else:
        samples = list(dataset)

    logger.info("Loaded %d samples (%s)", len(samples), args.subset)

    if "all" in args.methods:
        selected_methods = list(METHODS.keys())
    else:
        selected_methods = args.methods

    # -------------------------------------------------------------------------
    # Run every selected method on every sample
    # -------------------------------------------------------------------------
    all_results: Dict[str, List[RunnerResult]] = {m: [] for m in selected_methods}
    ground_truth = [s.species_id for s in samples]

    for method_name in selected_methods:
        logger.info("Running %s ...", method_name)
        runner = METHODS[method_name]()
        for sample in samples:
            result = runner.predict(sample)
            all_results[method_name].append(result)

    # -------------------------------------------------------------------------
    # Compute metrics
    # -------------------------------------------------------------------------
    metrics: Dict[str, Dict] = {}
    all_labels = sorted(set(ground_truth))

    for method_name in selected_methods:
        results = all_results[method_name]
        method_metrics = {
            "coverage": compute_coverage(results),
            "top1_accuracy": compute_top_k_accuracy(results, ground_truth, k=1),
            "top3_accuracy": compute_top_k_accuracy(results, ground_truth, k=3),
            "macro_f1": compute_macro_f1(results, ground_truth, all_labels),
            "mean_time_ms": compute_mean_inference_time(results),
            "ece": compute_ece(results, ground_truth),
        }

        # Bootstrap confidence intervals for key accuracy metrics
        top1_lo, top1_hi = bootstrap_ci(
            results, ground_truth, lambda r, g: compute_top_k_accuracy(r, g, k=1), n_bootstrap=2000
        )
        top3_lo, top3_hi = bootstrap_ci(
            results, ground_truth, lambda r, g: compute_top_k_accuracy(r, g, k=3), n_bootstrap=2000
        )
        method_metrics["top1_ci"] = [round(top1_lo, 3), round(top1_hi, 3)]
        method_metrics["top3_ci"] = [round(top3_lo, 3), round(top3_hi, 3)]

        # OOD analysis is only meaningful for the CNN
        if method_name == "cnn":
            id_results = [r for r, s in zip(results, samples) if s.in_distribution]
            ood_results = [r for r, s in zip(results, samples) if not s.in_distribution]
            method_metrics["ood"] = compute_ood_metrics(id_results, ood_results)

        metrics[method_name] = method_metrics

    metrics["agreement"] = compute_pairwise_agreement(all_results)

    # -------------------------------------------------------------------------
    # Write reports
    # -------------------------------------------------------------------------
    generate_json_report(metrics, all_results, ground_truth, output_dir / "report.json")
    generate_csv_report(all_results, ground_truth, samples, output_dir / "report.csv")
    generate_markdown_report(metrics, output_dir / "report.md")

    # -------------------------------------------------------------------------
    # Generate plots
    # -------------------------------------------------------------------------
    plot_accuracy_comparison(metrics, output_dir / "accuracy_comparison.png")
    plot_agreement_heatmap(metrics["agreement"], output_dir / "agreement_heatmap.png")
    plot_ood_analysis(
        [r for r, s in zip(all_results["cnn"], samples) if s.in_distribution],
        [r for r, s in zip(all_results["cnn"], samples) if not s.in_distribution],
        output_dir / "ood_confidence.png",
    )
    plot_confidence_distribution(
        all_results["cnn"], ground_truth, output_dir / "confidence_distribution.png"
    )

    logger.info("Benchmark complete. Results written to %s", output_dir)


if __name__ == "__main__":
    run()
