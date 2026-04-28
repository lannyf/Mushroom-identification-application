#!/usr/bin/env python3
"""
Generate SAM 2 masks for evaluation images (holdout + secondary).

Usage:
    cd /tmp && python /path/to/project/scripts/generate_sam2_eval_masks.py \
        --project-root /path/to/project \
        --eval-list data/raw/eval_holdout_30.txt \
        --output-dir data/SegMaskSAM2_eval

Uses the same prompt strategy as generate_sam2_masks.py.
Images are resolved relative to data/raw/ inside the project root.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Re-use the generation logic from generate_sam2_masks.py
sys.path.insert(0, str(Path(__file__).parent))
from generate_sam2_masks import (
    _check_cwd_safe,
    load_yolo_for_prompts,
    load_sam2_predictor,
    attempt_center_point,
    attempt_bbox_fallback,
    is_usable,
    rank_masks,
    _json_default,
)

import json
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image


def run_eval_generation(
    project_root: Path,
    eval_list: str,
    yolo_weights: str,
    sam2_config: str,
    sam2_checkpoint: str,
    device: str,
    output_dir: Path,
) -> Dict[str, Any]:
    _check_cwd_safe(project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"

    # Load image list
    list_path = project_root / eval_list
    with open(list_path, "r", encoding="utf-8") as f:
        raw_paths = [line.strip() for line in f if line.strip()]

    image_paths: List[Path] = []
    for p in raw_paths:
        # Try as-is first, then under data/raw/
        candidate = project_root / p
        if not candidate.exists():
            candidate = project_root / "data" / "raw" / p
        image_paths.append(candidate)

    print(f"Evaluation list: {eval_list} ({len(image_paths)} images)")

    # Resume support
    processed: set = set()
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    processed.add(entry["image"])
                except Exception:
                    pass
        print(f"Resuming: {len(processed)} images already processed.")

    todo = [p for p in image_paths if str(p.relative_to(project_root)) not in processed]
    print(f"Remaining: {len(todo)}")

    print("Loading YOLOv8n-seg for fallback bbox prompts...")
    yolo_model = load_yolo_for_prompts(str(project_root / yolo_weights))

    print("Loading SAM 2 model...")
    predictor = load_sam2_predictor(sam2_config, str(project_root / sam2_checkpoint), device=device)

    strategy_counts = {"center_point": 0, "bbox_fallback": 0, "failed": 0}
    total_time = 0.0

    with open(manifest_path, "a", encoding="utf-8") as manifest_f:
        for idx, img_path in enumerate(todo, 1):
            rel_path = str(img_path.relative_to(project_root))
            print(f"[{idx}/{len(todo)}] {rel_path} ... ", end="", flush=True)
            t0 = time.time()

            img_pil = Image.open(img_path).convert("RGB")
            img_np = np.array(img_pil)

            best = None
            strategy = "center_point"
            attempt1 = attempt_center_point(predictor, img_np)
            if attempt1 is not None:
                masks, iou_preds, ref_box = attempt1
                ranked = rank_masks(masks, iou_preds, ref_box)
                best = ranked[0]
                if not is_usable(best):
                    best = None

            if best is None:
                strategy = "bbox_fallback"
                attempt2 = attempt_bbox_fallback(predictor, img_np, yolo_model, str(img_path))
                if attempt2 is not None:
                    masks, iou_preds, ref_box = attempt2
                    ranked = rank_masks(masks, iou_preds, ref_box)
                    best = ranked[0]
                    if not is_usable(best):
                        best = None

            elapsed = time.time() - t0
            total_time += elapsed

            if best is None:
                print(f"FAILED ({elapsed:.1f}s)")
                strategy_counts["failed"] += 1
                entry = {
                    "image": rel_path,
                    "status": "failed",
                    "elapsed_seconds": round(elapsed, 2),
                }
                manifest_f.write(json.dumps(entry, default=_json_default) + "\n")
                manifest_f.flush()
                continue

            strategy_counts[strategy] += 1
            best_mask = masks[best["mask_index"]]
            mask_uint8 = (best_mask.astype(np.uint8) if best_mask.dtype == bool else (best_mask > 0).astype(np.uint8)) * 255
            mask_name = img_path.stem + "_sam2.png"
            mask_path = output_dir / mask_name
            cv2.imwrite(str(mask_path), mask_uint8)

            entry = {
                "image": rel_path,
                "mask_path": str(mask_path.relative_to(project_root)),
                "status": "success",
                "strategy": strategy,
                "elapsed_seconds": round(elapsed, 2),
                "prompt_overlap": best["prompt_overlap"],
                "compactness": best["compactness"],
                "sam_score": best["sam_score"],
                "border_touch": best["border_touch"],
                "area_ratio": best["area_ratio"],
            }
            manifest_f.write(json.dumps(entry, default=_json_default) + "\n")
            manifest_f.flush()

            label = "USABLE" if is_usable(best) else "REVIEW"
            print(f"{label} ({strategy}, {elapsed:.1f}s)")

    summary = {
        "total_images": len(image_paths),
        "strategy_counts": strategy_counts,
        "mean_time_seconds": round(total_time / len(todo), 2) if todo else 0.0,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    print(f"\nDone. Summary: {summary_path}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate SAM 2 masks for evaluation images")
    parser.add_argument("--project-root", type=str, default=os.environ.get("PROJECT_ROOT", "."))
    parser.add_argument("--eval-list", required=True, help="Text file with one image path per line (relative to project root)")
    parser.add_argument("--yolo-weights", type=str, default="artifacts/yolov8n-seg.pt")
    parser.add_argument("--sam2-config", type=str, default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--sam2-checkpoint", type=str, default="artifacts/sam2.1_hiera_tiny.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="data/SegMaskSAM2_eval")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    if not project_root.exists():
        print(f"Project root does not exist: {project_root}", file=sys.stderr)
        return 1

    _check_cwd_safe(project_root)

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        run_eval_generation(
            project_root=project_root,
            eval_list=args.eval_list,
            yolo_weights=args.yolo_weights,
            sam2_config=args.sam2_config,
            sam2_checkpoint=args.sam2_checkpoint,
            device=args.device,
            output_dir=project_root / args.output_dir,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
