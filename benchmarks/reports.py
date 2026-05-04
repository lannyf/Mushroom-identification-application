"""Report generators — JSON, CSV, and Markdown output.

Each function consumes the unified ``metrics`` dict and ``all_results``
dict produced by ``run_benchmark.py`` and writes a human- or machine-readable
file to disk.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from benchmarks.runners.base import RunnerResult


def generate_json_report(
    metrics: Dict,
    all_results: Dict[str, List[RunnerResult]],
    ground_truth: List[str],
    output_path: Path,
):
    """Write a complete structured JSON report with per-image breakdown.

    The report contains three top-level sections:
    * ``metadata`` — timestamp, image count, method list.
    * ``metrics`` — aggregated scores for every method.
    * ``per_image`` — ground truth and every method's prediction for each image.
    """
    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_images": len(ground_truth),
            "methods": list(all_results.keys()),
        },
        "metrics": metrics,
        "per_image": [],
    }
    for i, gt in enumerate(ground_truth):
        entry = {"ground_truth": gt}
        for method, results in all_results.items():
            r = results[i]
            entry[method] = {
                "top1": r.top_species if r.coverage else None,
                "confidence": r.top_confidence if r.coverage else None,
                "correct": r.top_species == gt if r.coverage else False,
                "coverage": r.coverage,
                "time_ms": r.inference_time_ms,
            }
        report["per_image"].append(entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def generate_csv_report(
    all_results: Dict[str, List[RunnerResult]],
    ground_truth: List[str],
    samples,
    output_path: Path,
):
    """Write a spreadsheet with one row per image and one column triplet per method.

    Each method contributes three columns: ``{method}_top1``,
    ``{method}_correct`` (1/0), and ``{method}_coverage`` (1/0).
    """
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["image_path", "ground_truth"]
        for method in all_results:
            header.extend([f"{method}_top1", f"{method}_correct", f"{method}_coverage"])
        writer.writerow(header)

        for i, (sample, gt) in enumerate(zip(samples, ground_truth)):
            row = [str(sample.image_path), gt]
            for method, results in all_results.items():
                r = results[i]
                row.extend(
                    [
                        r.top_species if r.coverage else "N/A",
                        "1" if (r.coverage and r.top_species == gt) else "0",
                        "1" if r.coverage else "0",
                    ]
                )
            writer.writerow(row)


def generate_markdown_report(metrics: Dict, output_path: Path):
    """Write a thesis-ready Markdown table with accuracy and confidence intervals.

    The table includes Top-1 / Top-3 accuracy, 95 % bootstrap confidence
    intervals, coverage, and mean inference time for every method.
    If CNN OOD metrics are present, a short OOD summary is appended.
    """
    lines = ["# Benchmark Results\n"]

    lines.append("## Accuracy Summary\n")
    lines.append("| Method | Top-1 | 95% CI | Top-3 | 95% CI | Coverage | Mean Time (ms) |")
    lines.append("|--------|-------|--------|-------|--------|----------|----------------|")
    for method, m in metrics.items():
        if method == "agreement":
            continue
        top1_ci = m.get("top1_ci", [None, None])
        top3_ci = m.get("top3_ci", [None, None])
        top1_ci_str = f"[{top1_ci[0]:.3f}, {top1_ci[1]:.3f}]" if top1_ci[0] is not None else "—"
        top3_ci_str = f"[{top3_ci[0]:.3f}, {top3_ci[1]:.3f}]" if top3_ci[0] is not None else "—"
        lines.append(
            f"| {method} | {m.get('top1_accuracy', 0):.3f} | {top1_ci_str} | "
            f"{m.get('top3_accuracy', 0):.3f} | {top3_ci_str} | "
            f"{m.get('coverage', 0):.3f} | "
            f"{m.get('mean_time_ms', 0):.1f} |"
        )

    if "cnn" in metrics and "ood" in metrics["cnn"]:
        lines.append("\n## OOD Analysis (CNN)\n")
        ood = metrics["cnn"]["ood"]
        lines.append(f"- ID avg confidence: {ood['id_avg_confidence']:.3f}")
        lines.append(f"- OOD avg confidence: {ood['ood_avg_confidence']:.3f}")
        lines.append(f"- Confidence gap: {ood['confidence_gap']:.3f}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
