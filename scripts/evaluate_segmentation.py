#!/usr/bin/env python3
"""
Evaluate segmentation model against ground-truth masks.

Usage:
    python scripts/evaluate_segmentation.py \
        --model artifacts/yolov8_seg_ft.pt \
        --images-dir data/raw/evaluation_images \
        --masks-dir data/SegMaskSAM2_eval \
        --output artifacts/segmentation_evaluation.json

Computes per-image and aggregate metrics:
    - Mask IoU (Intersection over Union)
    - Precision & Recall at IoU thresholds 0.50 and 0.75
    - Fallback rate (images where model produces no valid mask)

If --compare-generic is passed, also runs the generic yolov8n-seg model
and reports relative improvement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


def load_model(model_path: str) -> Any:
    from ultralytics import YOLO
    return YOLO(model_path)


def load_mask(mask_path: str, target_size: Tuple[int, int]) -> Optional[np.ndarray]:
    """Load binary mask and resize to target (W, H)."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    target_w, target_h = target_size
    if mask.shape[1] != target_w or mask.shape[0] != target_h:
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    intersection = np.count_nonzero(pred & gt)
    union = np.count_nonzero(pred | gt)
    return intersection / union if union > 0 else 0.0


def compute_precision_recall(
    pred: np.ndarray,
    gt: np.ndarray,
    iou_threshold: float = 0.50,
) -> Tuple[float, float]:
    """Compute precision and recall for a single mask pair at given IoU threshold."""
    iou = compute_iou(pred, gt)
    tp = 1 if iou >= iou_threshold else 0
    # Single instance per image for our dataset
    precision = float(tp)  # 1 TP / 1 pred
    recall = float(tp)     # 1 TP / 1 gt
    return precision, recall


def run_inference(model: Any, image_path: str) -> Optional[np.ndarray]:
    """Run YOLO segmentation and return the best mask, or None if no detection."""
    results = model(image_path, verbose=False)
    for r in results:
        masks = getattr(r, "masks", None)
        boxes = getattr(r, "boxes", None)
        if masks is None or boxes is None:
            continue
        try:
            mask_arr = masks.data.cpu().numpy()
            box_arr = boxes.data.cpu().numpy()
        except Exception:
            continue
        if mask_arr.shape[0] == 0:
            continue
        # Select highest confidence detection
        best_idx = int(np.argmax(box_arr[:, 4]))
        best_mask = mask_arr[best_idx]
        if best_mask.dtype == bool:
            return best_mask.astype(np.uint8)
        return (best_mask > 0).astype(np.uint8)
    return None


def evaluate_model(
    model_path: str,
    images_dir: str,
    masks_dir: str,
    iou_thresholds: List[float] = None,
) -> Dict[str, Any]:
    """Evaluate a single model against ground-truth masks."""
    if iou_thresholds is None:
        iou_thresholds = [0.50, 0.75]

    model = load_model(model_path)
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)

    mask_files = sorted(masks_dir.glob("*.png"))
    if not mask_files:
        print(f"No masks found in {masks_dir}", file=sys.stderr)
        return {"error": "no_masks"}

    per_image: List[Dict[str, Any]] = []
    iou_list: List[float] = []
    fallback_count = 0

    for mask_file in mask_files:
        stem = mask_file.stem
        # Try to find matching image (mask may be named <image_stem>_sam2.png)
        img_stem = stem
        if stem.endswith("_sam2"):
            img_stem = stem[:-5]

        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
            candidate = images_dir / f"{img_stem}{ext}"
            if candidate.exists():
                img_path = str(candidate)
                break
            # Also search recursively for evaluation images
            for found in images_dir.rglob(f"{img_stem}{ext}"):
                img_path = str(found)
                break
            if img_path:
                break

        if img_path is None:
            print(f"  Warning: no image found for mask {mask_file.name}, skipping", file=sys.stderr)
            continue

        img_pil = Image.open(img_path).convert("RGB")
        W, H = img_pil.size

        gt_mask = load_mask(str(mask_file), (W, H))
        pred_mask = run_inference(model, img_path)

        if pred_mask is None:
            fallback_count += 1
            entry = {
                "image": str(Path(img_path).name),
                "mask": str(mask_file.name),
                "status": "no_detection",
                "iou": 0.0,
            }
            for th in iou_thresholds:
                entry[f"precision@{th}"] = 0.0
                entry[f"recall@{th}"] = 0.0
            per_image.append(entry)
            continue

        # Resize pred to match GT if needed
        if pred_mask.shape[1] != W or pred_mask.shape[0] != H:
            pred_mask = cv2.resize(pred_mask, (W, H), interpolation=cv2.INTER_NEAREST)
            pred_mask = (pred_mask > 0).astype(np.uint8)

        iou = compute_iou(pred_mask, gt_mask)
        iou_list.append(iou)

        entry: Dict[str, Any] = {
            "image": str(Path(img_path).name),
            "mask": str(mask_file.name),
            "status": "ok",
            "iou": round(iou, 4),
        }
        for th in iou_thresholds:
            p, r = compute_precision_recall(pred_mask, gt_mask, th)
            entry[f"precision@{th}"] = round(p, 4)
            entry[f"recall@{th}"] = round(r, 4)
        per_image.append(entry)

    n = len(per_image)
    summary: Dict[str, Any] = {
        "model": model_path,
        "images_evaluated": n,
        "fallback_count": fallback_count,
        "fallback_rate": round(fallback_count / n, 4) if n > 0 else 0.0,
        "mean_iou": round(float(np.mean(iou_list)), 4) if iou_list else 0.0,
        "median_iou": round(float(np.median(iou_list)), 4) if iou_list else 0.0,
    }
    for th in iou_thresholds:
        precisions = [e[f"precision@{th}"] for e in per_image if e["status"] == "ok"]
        recalls = [e[f"recall@{th}"] for e in per_image if e["status"] == "ok"]
        summary[f"mean_precision@{th}"] = round(float(np.mean(precisions)), 4) if precisions else 0.0
        summary[f"mean_recall@{th}"] = round(float(np.mean(recalls)), 4) if recalls else 0.0

    return {
        "summary": summary,
        "per_image": per_image,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate segmentation model")
    parser.add_argument("--model", required=True, help="Path to YOLO model .pt")
    parser.add_argument("--images-dir", required=True, help="Directory of evaluation images")
    parser.add_argument("--masks-dir", required=True, help="Directory of ground-truth mask PNGs")
    parser.add_argument("--output", default="artifacts/segmentation_evaluation.json", help="Output JSON path")
    parser.add_argument("--compare-generic", action="store_true", help="Also evaluate generic yolov8n-seg for comparison")
    parser.add_argument("--iou-thresholds", nargs="+", type=float, default=[0.50, 0.75])
    args = parser.parse_args()

    print(f"Evaluating model: {args.model}")
    print(f"Images: {args.images_dir}")
    print(f"Masks:  {args.masks_dir}")
    print()

    results = evaluate_model(args.model, args.images_dir, args.masks_dir, args.iou_thresholds)
    if "error" in results:
        return 1

    output_data: Dict[str, Any] = {"fine_tuned": results}

    if args.compare_generic:
        generic_model = "yolov8n-seg.pt"
        print(f"\nEvaluating generic model: {generic_model}")
        generic_results = evaluate_model(generic_model, args.images_dir, args.masks_dir, args.iou_thresholds)
        output_data["generic"] = generic_results

        # Compute deltas
        ft = results["summary"]
        gen = generic_results["summary"]
        deltas: Dict[str, Any] = {}
        for key in ["mean_iou", "fallback_rate", "mean_precision@0.5", "mean_recall@0.5"]:
            if key in ft and key in gen:
                deltas[key] = round(ft[key] - gen[key], 4)
        output_data["deltas"] = deltas

        # Promotion gate check
        improved_metrics = sum(1 for v in deltas.values() if v > 0)
        print(f"\nPromotion check: improved on {improved_metrics} metrics")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
    print(f"\nResults saved to: {out_path}")

    summary = results["summary"]
    print(f"\nSummary ({args.model}):")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
