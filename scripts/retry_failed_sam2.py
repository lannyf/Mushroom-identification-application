#!/usr/bin/env python3
"""
Re-run SAM 2 on failed images using Automatic Mask Generation (AMG).

Usage:
    cd /tmp && python /path/to/project/scripts/retry_failed_sam2.py \
        --project-root /path/to/project \
        --manifest data/SegMaskSAM2/manifest.jsonl \
        --output-dir data/SegMaskSAM2

For each image with status="failed" in the manifest:
  1. Run SAM 2 AMG (grid of points, no manual prompts).
  2. Filter generated masks by area_ratio >= 0.01.
  3. Pick the mask whose bbox centroid is closest to image center.
  4. Save mask PNG and update manifest entry to "success".

This is slower than point prompting but more robust for off-center
or hand-occluded mushrooms.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image


def _check_cwd_safe(project_root: Path) -> None:
    cwd = Path.cwd()
    if (cwd / "sam2").is_dir():
        print(
            "ERROR: You are running from a directory that contains a 'sam2' sub-dir.\n"
            "This shadows the sam2 Python package. Please cd to /tmp (or any dir\n"
            "without a 'sam2' sub-dir) and re-run.",
            file=sys.stderr,
        )
        raise SystemExit(1)


def load_sam2_amg(config_file: str, checkpoint: str, device: str = "cpu") -> Any:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    model = build_sam2(config_file, checkpoint, device=device)
    return SAM2AutomaticMaskGenerator(
        model,
        points_per_side=16,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.85,
        min_mask_region_area=100,
    )


def process_failed_image(
    img_path: str,
    mask_generator: Any,
    output_dir: Path,
    min_area_ratio: float = 0.01,
) -> Optional[Dict[str, Any]]:
    """Run AMG on a single image and return the best mask entry, or None."""
    img_np = np.array(Image.open(img_path).convert("RGB"))
    H, W = img_np.shape[:2]

    masks = mask_generator.generate(img_np)
    if not masks:
        return None

    candidates: List[tuple] = []
    for m in masks:
        area_ratio = m["area"] / (W * H)
        if area_ratio < min_area_ratio:
            continue
        bbox = m["bbox"]  # x, y, w, h
        cx = bbox[0] + bbox[2] / 2.0
        cy = bbox[1] + bbox[3] / 2.0
        dist = ((cx - W / 2.0) ** 2 + (cy - H / 2.0) ** 2) ** 0.5
        candidates.append((dist, area_ratio, m))

    if not candidates:
        return None

    # Pick closest to center
    candidates.sort(key=lambda x: x[0])
    _, area_ratio, best = candidates[0]

    mask = best["segmentation"].astype(np.uint8) * 255
    mask_name = Path(img_path).stem + "_sam2.png"
    mask_path = output_dir / mask_name
    cv2.imwrite(str(mask_path), mask)

    return {
        "mask_path": str(mask_path),
        "area_ratio": round(area_ratio, 4),
        "predicted_iou": round(float(best["predicted_iou"]), 4),
        "stability_score": round(float(best["stability_score"]), 4),
        "center_distance_pixels": round(float(candidates[0][0]), 1),
    }


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError


def run_retry(
    project_root: Path,
    manifest_path: Path,
    output_dir: Path,
    sam2_config: str,
    sam2_checkpoint: str,
    device: str,
) -> Dict[str, Any]:
    _check_cwd_safe(project_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    failed_entries = [e for e in entries if e.get("status") == "failed"]
    print(f"Total manifest entries: {len(entries)}")
    print(f"Failed entries to retry: {len(failed_entries)}")

    if not failed_entries:
        print("No failed entries to retry.")
        return {"retried": 0, "recovered": 0}

    print("Loading SAM 2 AMG model...")
    mask_generator = load_sam2_amg(sam2_config, str(project_root / sam2_checkpoint), device=device)

    # Build lookup: image -> index for updating
    entry_index = {e["image"]: i for i, e in enumerate(entries)}
    recovered = 0
    still_failed = 0
    total_time = 0.0

    for idx, entry in enumerate(failed_entries, 1):
        rel_path = entry["image"]
        img_path = project_root / rel_path
        print(f"[{idx}/{len(failed_entries)}] {rel_path} ... ", end="", flush=True)
        t0 = time.time()

        result = process_failed_image(str(img_path), mask_generator, output_dir)
        elapsed = time.time() - t0
        total_time += elapsed

        if result is None:
            print(f"STILL_FAILED ({elapsed:.1f}s)")
            still_failed += 1
            # Update entry to preserve retry info
            entry["retry_status"] = "still_failed"
            entry["retry_elapsed"] = round(elapsed, 2)
        else:
            print(f"RECOVERED ({elapsed:.1f}s)")
            recovered += 1
            entry["status"] = "success"
            entry["strategy"] = "amg_retry"
            entry.update(result)
            entry["retry_elapsed"] = round(elapsed, 2)

        # Update in entries list
        entries[entry_index[rel_path]] = entry

    # Rewrite manifest
    with open(manifest_path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, default=_json_default) + "\n")

    summary = {
        "retried": len(failed_entries),
        "recovered": recovered,
        "still_failed": still_failed,
        "mean_time_seconds": round(total_time / len(failed_entries), 2) if failed_entries else 0.0,
    }
    print(f"\nRetry complete: {summary}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Retry failed SAM 2 masks with AMG")
    parser.add_argument("--project-root", type=str, default=os.environ.get("PROJECT_ROOT", "."))
    parser.add_argument("--manifest", type=str, default="data/SegMaskSAM2/manifest.jsonl")
    parser.add_argument("--output-dir", type=str, default="data/SegMaskSAM2")
    parser.add_argument("--sam2-config", type=str, default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--sam2-checkpoint", type=str, default="artifacts/sam2.1_hiera_tiny.pt")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    if not project_root.exists():
        print(f"Project root does not exist: {project_root}", file=sys.stderr)
        return 1

    _check_cwd_safe(project_root)

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        run_retry(
            project_root=project_root,
            manifest_path=project_root / args.manifest,
            output_dir=project_root / args.output_dir,
            sam2_config=args.sam2_config,
            sam2_checkpoint=args.sam2_checkpoint,
            device=args.device,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
