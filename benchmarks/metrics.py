"""Metric computations for the benchmark suite.

All functions operate on lists of ``RunnerResult`` objects and
produce scalar summaries (accuracy, coverage, calibration, etc.).
The bootstrap routine provides confidence intervals for stochastic
estimates such as top-k accuracy.
"""

import numpy as np
from typing import Callable, Dict, List, Tuple

from benchmarks.runners.base import RunnerResult


def compute_top_k_accuracy(
    results: List[RunnerResult],
    ground_truth: List[str],
    k: int = 1,
) -> float:
    """Fraction of covered predictions where the ground truth is in the top-k.

    Samples with ``coverage=False`` are skipped so that accuracy is
    computed only over attempted predictions.
    """
    correct = 0
    total = 0
    for res, gt in zip(results, ground_truth):
        if not res.coverage:
            continue
        total += 1
        top_k = [sp for sp, _ in res.predictions[:k]]
        if gt in top_k:
            correct += 1
    return correct / total if total > 0 else 0.0


def compute_coverage(results: List[RunnerResult]) -> float:
    """Proportion of samples for which the method produced a prediction."""
    return sum(1 for r in results if r.coverage) / len(results) if results else 0.0


def compute_macro_f1(
    results: List[RunnerResult],
    ground_truth: List[str],
    all_labels: List[str],
) -> float:
    """Macro-averaged F1 across all species present in the evaluation set.

    Uncovered predictions are treated as the literal string ``"UNKNOWN"``.
    """
    from sklearn.metrics import f1_score

    y_pred = [r.top_species if r.coverage else "UNKNOWN" for r in results]
    return f1_score(
        ground_truth, y_pred, labels=all_labels, average="macro", zero_division=0
    )


def compute_mean_inference_time(results: List[RunnerResult]) -> float:
    """Average wall-clock time in milliseconds across all predictions."""
    times = [r.inference_time_ms for r in results]
    return sum(times) / len(times) if times else 0.0


def compute_ece(
    results: List[RunnerResult],
    ground_truth: List[str],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE) with uniform confidence bins.

    Measures how well the reported confidences match the empirical
    accuracy within each bin. Lower is better.
    """
    confidences = []
    accuracies = []
    for res, gt in zip(results, ground_truth):
        if res.coverage and res.predictions:
            confidences.append(res.top_confidence)
            accuracies.append(1.0 if res.top_species == gt else 0.0)

    if not confidences:
        return 0.0

    ece = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        bin_indices = [j for j, c in enumerate(confidences) if lo <= c < hi]
        if bin_indices:
            avg_conf = sum(confidences[j] for j in bin_indices) / len(bin_indices)
            avg_acc = sum(accuracies[j] for j in bin_indices) / len(bin_indices)
            ece += abs(avg_acc - avg_conf) * (len(bin_indices) / len(confidences))
    return ece


def compute_ood_metrics(
    id_results: List[RunnerResult],
    ood_results: List[RunnerResult],
) -> Dict[str, float]:
    """Compare average confidence on in-distribution vs out-of-distribution samples.

    A large positive ``confidence_gap`` suggests the model is somewhat
    aware that OOD inputs are unfamiliar.
    """
    id_conf = [r.top_confidence for r in id_results if r.coverage]
    ood_conf = [r.top_confidence for r in ood_results if r.coverage]
    return {
        "id_avg_confidence": sum(id_conf) / len(id_conf) if id_conf else 0.0,
        "ood_avg_confidence": sum(ood_conf) / len(ood_conf) if ood_conf else 0.0,
        "confidence_gap": (sum(id_conf) / len(id_conf) if id_conf else 0.0)
        - (sum(ood_conf) / len(ood_conf) if ood_conf else 0.0),
    }


def bootstrap_ci(
    results: List[RunnerResult],
    ground_truth: List[str],
    metric_fn: Callable[[List[RunnerResult], List[str]], float],
    n_bootstrap: int = 2000,
    ci: float = 0.95,
) -> Tuple[float, float]:
    """Compute a bootstrap confidence interval for an arbitrary metric.

    Resamples the evaluation set with replacement ``n_bootstrap`` times
    and returns the percentile-based interval at the requested confidence
    level.

    Args:
        results: List of ``RunnerResult`` objects.
        ground_truth: Parallel list of ground-truth species IDs.
        metric_fn: Metric callable with signature ``(results, ground_truth) -> float``.
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence level (e.g. 0.95 for 95 %).

    Returns:
        ``(lower_bound, upper_bound)`` for the specified confidence level.
    """
    rng = np.random.default_rng(42)
    n = len(results)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        resampled_results = [results[i] for i in idx]
        resampled_gt = [ground_truth[i] for i in idx]
        scores.append(metric_fn(resampled_results, resampled_gt))
    alpha = 1 - ci
    lower = float(np.percentile(scores, 100 * alpha / 2))
    upper = float(np.percentile(scores, 100 * (1 - alpha / 2)))
    return lower, upper


def compute_pairwise_agreement(
    all_results: Dict[str, List[RunnerResult]],
) -> Dict[str, Dict[str, float]]:
    """Fraction of samples on which each pair of methods agrees on the top-1 prediction.

    Only samples where *both* methods have ``coverage=True`` are counted.
    The diagonal is always 1.0 (a method agrees with itself).
    """
    methods = list(all_results.keys())
    agreement = {m1: {m2: 0.0 for m2 in methods} for m1 in methods}
    n = len(next(iter(all_results.values())))

    for i in range(n):
        for m1 in methods:
            for m2 in methods:
                r1 = all_results[m1][i]
                r2 = all_results[m2][i]
                if r1.coverage and r2.coverage and r1.top_species == r2.top_species:
                    agreement[m1][m2] += 1 / n
    return agreement
