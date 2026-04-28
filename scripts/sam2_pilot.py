#!/usr/bin/env python3
"""
SAM 2 pilot script: 20-image quality gate for pseudo-label generation.

Usage:
    # Must be run from OUTSIDE the sam2/ repo directory to avoid import shadowing
    cd /tmp && python /path/to/project/scripts/sam2_pilot.py

Prompt strategy (per plan):
  1. Primary: center positive point + 4 corner negative points.
     This tells SAM 2 "segment the object near the center, not the background
     at the edges."
  2. Fallback: generic YOLOv8n-seg bbox prompt (only if primary fails).

Mask ranking cascade:
   hard filter: border_touch AND prompt_overlap < 0.5
   primary sort: prompt_overlap (desc)
   secondary sort: compactness (desc)
   tertiary sort: iou_prediction (desc)

Usability criteria:
   mask area ≥ 1% of image, prompt_overlap ≥ 0.30, border_touch < 0.80,
   compactness ≥ 0.05

Pass gate: >80% of 20 images produce usable masks.

Output:
    artifacts/sam2_pilot_report.json
    artifacts/sam2_pilot_masks/          binary PNG masks
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


def _check_cwd_safe(project_root: Path) -> None:
    cwd = Path.cwd()
    if (cwd / "sam2").is_dir():
        print(
            "ERROR: You are running from a directory that contains a 'sam2' sub-dir.\n"
            "This shadows the sam2 Python package. Please cd to /tmp (or any dir\n"
            "without a 'sam2' sub-dir) and re-run.\n"
            f"  cd /tmp && python {project_root}/scripts/sam2_pilot.py",
            file=sys.stderr,
        )
        raise SystemExit(1)


def load_yolo_for_prompts(weights_path: str) -> Any:
    from ultralytics import YOLO
    return YOLO(weights_path)


def load_sam2_predictor(config_file: str, checkpoint: str, device: str = "cpu") -> Any:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    model = build_sam2(config_file, checkpoint, device=device)
    return SAM2ImagePredictor(model)


def yolo_bbox_prompt(
    yolo_model: Any,
    image_path: str,
    min_confidence: float = 0.25,
    min_area_ratio: float = 0.01,
    max_area_ratio: float = 0.95,
    min_aspect: float = 0.2,
    max_aspect: float = 5.0,
    max_center_distance_ratio: float = 0.40,
) -> Optional[Tuple[np.ndarray, float]]:
    """
    Run generic YOLO on image and return the best bbox prompt for SAM 2.
    Returns (box_xyxy, confidence) or None if no plausible bbox found.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None
    H, W = img_bgr.shape[:2]
    results = yolo_model(img_bgr, verbose=False)
    candidates: List[Tuple[np.ndarray, float]] = []

    for r in results:
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        try:
            arr = boxes.data.cpu().numpy()
        except Exception:
            continue
        for i in range(arr.shape[0]):
            x1, y1, x2, y2, conf, _ = arr[i]
            conf = float(conf)
            if conf < min_confidence:
                continue
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            area_ratio = (bw * bh) / (W * H)
            if not (min_area_ratio <= area_ratio <= max_area_ratio):
                continue
            aspect = float(bw) / float(bh)
            if not (min_aspect <= aspect <= max_aspect):
                continue
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            img_cx, img_cy = W / 2.0, H / 2.0
            dist_ratio = max(abs(cx - img_cx) / W, abs(cy - img_cy) / H)
            if dist_ratio > max_center_distance_ratio:
                continue
            box = np.array([float(x1), float(y1), float(x2), float(y2)], dtype=np.float32)
            candidates.append((box, conf))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0]


def sam2_predict_masks(
    predictor: Any,
    img_np: np.ndarray,
    point_coords: Optional[np.ndarray] = None,
    point_labels: Optional[np.ndarray] = None,
    box: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run SAM 2 prediction and return (masks, iou_predictions)."""
    predictor.set_image(img_np)
    masks, iou_preds, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box,
        multimask_output=True,
    )
    return masks, iou_preds


def compute_mask_metrics(
    mask: np.ndarray,
    reference_box: Optional[np.ndarray],
    iou_prediction: float,
) -> Dict[str, Any]:
    """Compute ranking metrics for a single SAM 2 mask."""
    H, W = mask.shape[:2]
    binary = (mask > 0).astype(np.uint8)
    area = int(np.count_nonzero(binary))
    area_ratio = area / (H * W)

    # Border touch: fraction of mask pixels touching image border
    border_mask = np.zeros_like(binary)
    border_mask[0, :] = 1
    border_mask[-1, :] = 1
    border_mask[:, 0] = 1
    border_mask[:, -1] = 1
    border_pixels = int(np.count_nonzero(binary & border_mask))
    border_touch = border_pixels / area if area > 0 else 0.0

    # Prompt overlap: IoU between mask bbox and reference box
    prompt_overlap = 0.0
    compactness = 0.0
    ys, xs = np.where(binary > 0)
    if len(xs) > 0 and reference_box is not None:
        mx1, mx2 = int(xs.min()), int(xs.max())
        my1, my2 = int(ys.min()), int(ys.max())
        mask_box = np.array([mx1, my1, mx2, my2], dtype=np.float32)

        ix1 = max(reference_box[0], mask_box[0])
        iy1 = max(reference_box[1], mask_box[1])
        ix2 = min(reference_box[2], mask_box[2])
        iy2 = min(reference_box[3], mask_box[2])
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih
        ref_area = max(1, (reference_box[2] - reference_box[0]) * (reference_box[3] - reference_box[1]))
        mask_area_box = max(1, (mask_box[2] - mask_box[0]) * (mask_box[3] - mask_box[1]))
        union = ref_area + mask_area_box - inter
        prompt_overlap = inter / union if union > 0 else 0.0

    if area > 0:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            a = cv2.contourArea(largest)
            p = cv2.arcLength(largest, True)
            compactness = (4.0 * np.pi * a) / (p * p) if p > 0 else 0.0

    return {
        "area": area,
        "area_ratio": round(area_ratio, 4),
        "border_touch": round(border_touch, 4),
        "prompt_overlap": round(prompt_overlap, 4),
        "compactness": round(compactness, 4),
        "sam_score": round(float(iou_prediction), 4),
    }


def rank_masks(
    masks: np.ndarray,
    iou_preds: np.ndarray,
    reference_box: Optional[np.ndarray],
) -> List[Dict[str, Any]]:
    """Rank SAM 2 candidate masks using the cascade from the plan."""
    ranked: List[Dict[str, Any]] = []
    for i in range(masks.shape[0]):
        mask = masks[i]
        mask_uint8 = mask.astype(np.uint8) if mask.dtype == bool else (mask > 0).astype(np.uint8)
        metrics = compute_mask_metrics(mask_uint8, reference_box, float(iou_preds[i]))
        metrics["mask_index"] = i
        ranked.append(metrics)

    # Hard filter: reject if border_touch AND prompt_overlap both < 0.5
    filtered = [
        m for m in ranked
        if not (m["border_touch"] < 0.5 and m["prompt_overlap"] < 0.5)
    ]
    if not filtered:
        filtered = ranked

    filtered.sort(key=lambda m: (m["prompt_overlap"], m["compactness"], m["sam_score"]), reverse=True)
    return filtered


def is_usable(metrics: Dict[str, Any]) -> bool:
    """Return True if mask meets usability criteria."""
    return (
        metrics["area_ratio"] >= 0.01
        and metrics["prompt_overlap"] >= 0.30
        and metrics["border_touch"] < 0.80
        and metrics["compactness"] >= 0.05
    )


def attempt_center_point(
    predictor: Any,
    img_np: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Primary strategy: center positive point + 4 corner negative points.
    Returns (masks, iou_preds, reference_box) or None on failure.
    """
    H, W = img_np.shape[:2]
    # Center positive point
    center = np.array([[W / 2.0, H / 2.0]], dtype=np.float32)
    labels = np.array([1], dtype=np.int32)
    # Corner negative points (inset by 5 px to stay inside image)
    corners = np.array([
        [5.0, 5.0],
        [W - 5.0, 5.0],
        [5.0, H - 5.0],
        [W - 5.0, H - 5.0],
    ], dtype=np.float32)
    corner_labels = np.array([0, 0, 0, 0], dtype=np.int32)

    points = np.vstack([center, corners])
    plabels = np.concatenate([labels, corner_labels])

    masks, iou_preds = sam2_predict_masks(predictor, img_np, point_coords=points, point_labels=plabels)
    # Reference box for ranking: center ±40% of image dims
    ref_box = np.array([
        W * 0.1, H * 0.1,
        W * 0.9, H * 0.9,
    ], dtype=np.float32)
    return masks, iou_preds, ref_box


def attempt_bbox_fallback(
    predictor: Any,
    img_np: np.ndarray,
    yolo_model: Any,
    image_path: str,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Fallback strategy: use YOLO bbox as SAM 2 box prompt.
    Returns (masks, iou_preds, reference_box) or None if no YOLO detection.
    """
    bbox_result = yolo_bbox_prompt(yolo_model, image_path)
    if bbox_result is None:
        return None
    box, _ = bbox_result
    masks, iou_preds = sam2_predict_masks(predictor, img_np, box=box)
    return masks, iou_preds, box


def select_pilot_images(
    images_dir: str,
    n: int = 20,
    seed: int = 42,
) -> List[str]:
    """Select N stratified training images (2 per species where possible)."""
    root = Path(images_dir)
    species_dirs = [d for d in root.iterdir() if d.is_dir()]
    per_species: Dict[str, List[str]] = {}
    all_images: List[str] = []
    for sp in sorted(species_dirs):
        imgs = sorted([
            str(p) for p in sp.glob("*")
            if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")
        ])
        if imgs:
            per_species[sp.name] = imgs
            all_images.extend(imgs)

    if len(all_images) <= n:
        return all_images

    random.seed(seed)
    selected: List[str] = []
    for sp in sorted(per_species.keys()):
        imgs = per_species[sp]
        k = min(2, len(imgs))
        selected.extend(random.sample(imgs, k))

    if len(selected) < n:
        remaining = [p for p in all_images if p not in selected]
        need = n - len(selected)
        selected.extend(random.sample(remaining, min(need, len(remaining))))

    if len(selected) > n:
        selected = random.sample(selected, n)

    return sorted(selected)


def run_pilot(
    project_root: Path,
    images_dir: str,
    yolo_weights: str,
    sam2_config: str,
    sam2_checkpoint: str,
    n_images: int,
    seed: int,
    device: str,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run the SAM 2 pilot on N images and return a report."""
    _check_cwd_safe(project_root)

    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(exist_ok=True)

    print("Loading YOLOv8n-seg for fallback bbox prompts...")
    yolo_model = load_yolo_for_prompts(str(project_root / yolo_weights))

    print("Loading SAM 2 model...")
    predictor = load_sam2_predictor(sam2_config, str(project_root / sam2_checkpoint), device=device)

    image_paths = select_pilot_images(images_dir, n=n_images, seed=seed)
    print(f"Selected {len(image_paths)} pilot images.")

    results: List[Dict[str, Any]] = []
    usable_count = 0
    total_time = 0.0
    strategy_counts = {"center_point": 0, "bbox_fallback": 0, "failed": 0}

    for idx, img_path in enumerate(image_paths, 1):
        rel_path = str(Path(img_path).relative_to(project_root))
        print(f"[{idx}/{len(image_paths)}] {rel_path} ... ", end="", flush=True)
        t0 = time.time()

        img_pil = Image.open(img_path).convert("RGB")
        img_np = np.array(img_pil)

        # ---- Attempt 1: center point + negative corners ----
        attempt1 = attempt_center_point(predictor, img_np)
        best = None
        strategy = "center_point"

        if attempt1 is not None:
            masks, iou_preds, ref_box = attempt1
            ranked = rank_masks(masks, iou_preds, ref_box)
            best = ranked[0]
            if not is_usable(best):
                best = None  # trigger fallback

        # ---- Attempt 2: YOLO bbox fallback ----
        if best is None:
            strategy = "bbox_fallback"
            attempt2 = attempt_bbox_fallback(predictor, img_np, yolo_model, img_path)
            if attempt2 is not None:
                masks, iou_preds, ref_box = attempt2
                ranked = rank_masks(masks, iou_preds, ref_box)
                best = ranked[0]
                if not is_usable(best):
                    best = None

        if best is None:
            print("FAILED")
            strategy_counts["failed"] += 1
            results.append({
                "image": rel_path,
                "status": "failed",
                "usable": False,
            })
            continue

        strategy_counts[strategy] += 1
        best_mask = masks[best["mask_index"]]
        mask_uint8 = (best_mask.astype(np.uint8) if best_mask.dtype == bool else (best_mask > 0).astype(np.uint8)) * 255

        mask_name = Path(img_path).stem + "_sam2.png"
        mask_path = masks_dir / mask_name
        cv2.imwrite(str(mask_path), mask_uint8)

        elapsed = time.time() - t0
        total_time += elapsed

        usable = is_usable(best)
        if usable:
            usable_count += 1

        entry: Dict[str, Any] = {
            "image": rel_path,
            "status": "success",
            "usable": usable,
            "strategy": strategy,
            "elapsed_seconds": round(elapsed, 2),
            "selected_mask": {
                "mask_index": best["mask_index"],
                "mask_path": str(mask_path.relative_to(project_root)),
                **{k: best[k] for k in ("area", "area_ratio", "border_touch", "prompt_overlap", "compactness", "sam_score")},
            },
            "all_masks": ranked,
        }
        results.append(entry)
        label = "USABLE" if usable else "REVIEW"
        print(f"{label} ({strategy}, overlap={best['prompt_overlap']:.2f}, compact={best['compactness']:.2f}, score={best['sam_score']:.2f})")

    pass_rate = usable_count / len(image_paths) if image_paths else 0.0
    report: Dict[str, Any] = {
        "pilot_config": {
            "n_images": n_images,
            "seed": seed,
            "device": device,
            "yolo_weights": yolo_weights,
            "sam2_config": sam2_config,
            "sam2_checkpoint": sam2_checkpoint,
        },
        "summary": {
            "total_images": len(image_paths),
            "usable_masks": usable_count,
            "review_needed": len(image_paths) - usable_count,
            "pass_rate": round(pass_rate, 4),
            "pass_gate": pass_rate > 0.80,
            "mean_time_seconds": round(total_time / len(image_paths), 2) if image_paths else 0.0,
            "strategy_counts": strategy_counts,
        },
        "results": results,
    }

    def _convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError

    report_path = output_dir / "sam2_pilot_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=_convert), encoding="utf-8")
    print(f"\nPilot report saved to: {report_path}")
    print(f"Pass rate: {pass_rate:.1%} (gate: >80%)")
    print(f"Gate result: {'PASS' if pass_rate > 0.80 else 'FAIL'}")
    print(f"Strategies: center_point={strategy_counts['center_point']}, bbox_fallback={strategy_counts['bbox_fallback']}, failed={strategy_counts['failed']}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="SAM 2 pilot: 20-image quality gate")
    parser.add_argument("--project-root", type=str, default=os.environ.get("PROJECT_ROOT", "."))
    parser.add_argument("--images-dir", type=str, default="data/raw/images")
    parser.add_argument("--yolo-weights", type=str, default="artifacts/yolov8n-seg.pt")
    parser.add_argument("--sam2-config", type=str, default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--sam2-checkpoint", type=str, default="artifacts/sam2.1_hiera_tiny.pt")
    parser.add_argument("--n-images", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="artifacts/sam2_pilot")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    if not project_root.exists():
        print(f"Project root does not exist: {project_root}", file=sys.stderr)
        return 1

    _check_cwd_safe(project_root)

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        report = run_pilot(
            project_root=project_root,
            images_dir=str(project_root / args.images_dir),
            yolo_weights=args.yolo_weights,
            sam2_config=args.sam2_config,
            sam2_checkpoint=args.sam2_checkpoint,
            n_images=args.n_images,
            seed=args.seed,
            device=args.device,
            output_dir=project_root / args.output_dir,
        )

    return 0 if report["summary"]["pass_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
