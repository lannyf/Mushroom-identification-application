#!/usr/bin/env python3
"""
Convert COCO segmentation annotations to YOLO polygon format.

Usage:
    python scripts/convert_coco_to_yolo.py \
        --coco-json "data/Mushroom segmentation.coco-segmentation/train/_annotations.coco.json" \
        --images-dir "data/Mushroom segmentation.coco-segmentation/train" \
        --output-dir data/segmentation/eval_annotations/yolo \
        --rdp-epsilon 2.0

Output:
    One .txt file per image in YOLOv8-seg polygon format:
        <class_id> <x1> <y1> <x2> <y2> ...
    where coordinates are normalized to [0, 1].

Handles both polygon and RLE (run-length encoded) COCO segmentations.
Multi-ring polygons: outer ring kept, inner rings (holes) ignored for YOLO format.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


def decode_rle_to_mask(segmentation: Dict[str, Any], height: int, width: int) -> np.ndarray:
    """Decode COCO RLE segmentation to binary mask."""
    try:
        from pycocotools import mask as maskUtils
    except ImportError as exc:
        raise ImportError(
            "pycocotools is required for RLE decoding. Install with: pip install pycocotools"
        ) from exc

    if isinstance(segmentation["counts"], list):
        # Uncompressed RLE
        rle = maskUtils.frPyObjects(segmentation, height, width)
    else:
        # Compressed RLE
        rle = segmentation
    mask = maskUtils.decode(rle)
    return mask.astype(np.uint8)


def mask_to_polygons(mask: np.ndarray, rdp_epsilon: float = 2.0) -> List[np.ndarray]:
    """Convert binary mask to a list of simplified polygons (outer contours only)."""
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: List[np.ndarray] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1:
            continue
        # Ramer-Douglas-Peucker simplification
        peri = cv2.arcLength(cnt, True)
        eps = rdp_epsilon if rdp_epsilon > 0 else 0.001
        simplified = cv2.approxPolyDP(cnt, eps, True)
        if simplified.shape[0] < 3:
            # Too few points after simplification; keep original
            simplified = cnt
        polygons.append(simplified)
    return polygons


def coco_polygon_to_numpy(segmentation: List[List[float]]) -> np.ndarray:
    """Convert COCO flat polygon list to (N, 1, 2) contour array."""
    flat = segmentation[0] if isinstance(segmentation[0], list) else segmentation
    pts = np.array(flat, dtype=np.float32).reshape(-1, 2)
    return pts.reshape(-1, 1, 2)


def normalize_polygon(poly: np.ndarray, width: int, height: int) -> List[float]:
    """Normalize polygon coordinates to [0, 1]. Returns flat list of x, y pairs."""
    pts = poly.reshape(-1, 2)
    coords: List[float] = []
    for x, y in pts:
        coords.append(max(0.0, min(1.0, x / width)))
        coords.append(max(0.0, min(1.0, y / height)))
    return coords


def convert_annotations(
    coco_json_path: str,
    images_dir: str,
    output_dir: str,
    rdp_epsilon: float = 2.0,
    min_area_pixels: int = 50,
) -> Dict[str, Any]:
    """Convert COCO annotations to YOLO polygon format."""
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    categories = {cat["id"]: cat for cat in coco["categories"]}

    # Group annotations by image
    anns_by_image: Dict[int, List[Dict[str, Any]]] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    stats = {
        "images_processed": 0,
        "annotations_processed": 0,
        "annotations_skipped": 0,
        "rle_converted": 0,
        "polygon_converted": 0,
        "output_files": [],
    }

    for img_id, anns in sorted(anns_by_image.items()):
        img = images.get(img_id)
        if img is None:
            print(f"  Warning: image {img_id} not found in COCO images list", file=sys.stderr)
            continue

        img_name = img["file_name"]
        width = int(img["width"])
        height = int(img["height"])

        # Use original filename from extra metadata if available
        orig_name = img.get("extra", {}).get("name", img_name)
        stem = Path(orig_name).stem
        out_path = output_dir / f"{stem}.txt"

        lines: List[str] = []
        for ann in anns:
            seg = ann["segmentation"]
            polygons: List[np.ndarray] = []

            if isinstance(seg, dict):
                # RLE mask
                mask = decode_rle_to_mask(seg, height, width)
                polygons = mask_to_polygons(mask, rdp_epsilon)
                stats["rle_converted"] += 1
            elif isinstance(seg, list) and len(seg) > 0:
                # Polygon list of lists
                if isinstance(seg[0], list):
                    for ring in seg:
                        cnt = coco_polygon_to_numpy(ring)
                        area = cv2.contourArea(cnt)
                        if area < min_area_pixels:
                            continue
                        peri = cv2.arcLength(cnt, True)
                        eps = rdp_epsilon if rdp_epsilon > 0 else 0.001
                        simplified = cv2.approxPolyDP(cnt, eps, True)
                        if simplified.shape[0] < 3:
                            simplified = cnt
                        polygons.append(simplified)
                else:
                    # Flat list (shouldn't happen in standard COCO but be safe)
                    cnt = coco_polygon_to_numpy(seg)
                    area = cv2.contourArea(cnt)
                    if area >= min_area_pixels:
                        peri = cv2.arcLength(cnt, True)
                        eps = rdp_epsilon if rdp_epsilon > 0 else 0.001
                        simplified = cv2.approxPolyDP(cnt, eps, True)
                        if simplified.shape[0] < 3:
                            simplified = cnt
                        polygons.append(simplified)
                stats["polygon_converted"] += 1
            else:
                print(
                    f"  Warning: unknown segmentation format for ann {ann['id']}",
                    file=sys.stderr,
                )
                stats["annotations_skipped"] += 1
                continue

            for poly in polygons:
                coords = normalize_polygon(poly, width, height)
                if len(coords) < 6:
                    # Need at least 3 points (6 coords)
                    stats["annotations_skipped"] += 1
                    continue
                line = "0 " + " ".join(f"{c:.6f}" for c in coords)
                lines.append(line)
                stats["annotations_processed"] += 1

        if lines:
            out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            stats["output_files"].append(str(out_path))
        else:
            # Write empty file so we know this image was processed
            out_path.write_text("", encoding="utf-8")
            stats["output_files"].append(str(out_path))

        stats["images_processed"] += 1

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert COCO segmentation to YOLO polygon format")
    parser.add_argument("--coco-json", required=True, help="Path to COCO annotations JSON")
    parser.add_argument("--images-dir", required=True, help="Directory containing COCO images")
    parser.add_argument("--output-dir", required=True, help="Output directory for YOLO .txt files")
    parser.add_argument(
        "--rdp-epsilon",
        type=float,
        default=2.0,
        help="Ramer-Douglas-Peucker epsilon for polygon simplification (default: 2.0)",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=50,
        help="Minimum contour area in pixels to keep (default: 50)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.coco_json):
        print(f"Error: COCO JSON not found: {args.coco_json}", file=sys.stderr)
        return 1

    print(f"Converting COCO annotations: {args.coco_json}")
    print(f"Images dir: {args.images_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"RDP epsilon: {args.rdp_epsilon}")
    print()

    stats = convert_annotations(
        coco_json_path=args.coco_json,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        rdp_epsilon=args.rdp_epsilon,
        min_area_pixels=args.min_area,
    )

    print(f"Images processed: {stats['images_processed']}")
    print(f"Annotations processed: {stats['annotations_processed']}")
    print(f"Annotations skipped: {stats['annotations_skipped']}")
    print(f"RLE masks converted: {stats['rle_converted']}")
    print(f"Polygon masks converted: {stats['polygon_converted']}")
    print(f"Output files written: {len(stats['output_files'])}")
    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
