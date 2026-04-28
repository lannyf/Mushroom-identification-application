#!/usr/bin/env python3
"""
Prepare YOLOv8-seg dataset from images + binary masks.

Usage:
    python scripts/prepare_yolo_seg_dataset.py \
        --images-dir data/raw/images \
        --masks-dir data/SegMaskSAM2 \
        --background-dir data/raw/background \
        --output-dir data/segmentation \
        --train-ratio 0.80 \
        --seed 42

Output layout (YOLOv8-seg standard):
    data/segmentation/
      dataset.yaml
      images/
        train/
        val/
      labels/
        train/
        val/

Each label file contains one line per instance:
    <class_id> <x1> <y1> <x2> <y2> ...
where coordinates are normalized to [0, 1].

Background images are copied into train/val without label files.
Stratified split by parent folder name (species) when possible.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


def mask_to_yolo_polygons(
    mask_path: str,
    rdp_epsilon: float = 2.0,
    min_area_pixels: int = 50,
    max_area_ratio: float = 0.95,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Convert a binary mask PNG to YOLO polygon lines.

    Returns (list_of_lines, stats_dict).
    Each line: "0 x1 y1 x2 y2 ..."
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return [], {"error": "cannot_read_mask"}

    H, W = mask.shape[:2]
    binary = (mask > 0).astype(np.uint8)
    total_pixels = H * W
    mask_area = int(np.count_nonzero(binary))

    stats = {
        "mask_path": mask_path,
        "image_size": (W, H),
        "mask_area": mask_area,
        "mask_area_ratio": round(mask_area / total_pixels, 4),
    }

    if mask_area < min_area_pixels:
        stats["skipped"] = "too_small"
        return [], stats

    if mask_area / total_pixels > max_area_ratio:
        stats["skipped"] = "too_large"
        return [], stats

    # Find outer contours only (ignore holes for YOLO format)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines: List[str] = []
    kept_contours = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_pixels:
            continue
        peri = cv2.arcLength(cnt, True)
        eps = rdp_epsilon if rdp_epsilon > 0 else 0.001
        simplified = cv2.approxPolyDP(cnt, eps, True)
        if simplified.shape[0] < 3:
            simplified = cnt
        pts = simplified.reshape(-1, 2).astype(np.float32)
        coords: List[float] = []
        for x, y in pts:
            coords.append(max(0.0, min(1.0, x / W)))
            coords.append(max(0.0, min(1.0, y / H)))
        if len(coords) >= 6:
            line = "0 " + " ".join(f"{c:.6f}" for c in coords)
            lines.append(line)
            kept_contours += 1

    stats["contours_found"] = len(contours)
    stats["contours_kept"] = kept_contours
    stats["lines_written"] = len(lines)
    return lines, stats


def stratified_split(
    items: List[Tuple[str, str]],
    train_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Split items into train/val stratified by group (species folder).

    items: list of (image_path, group_name)
    Returns (train_items, val_items)
    """
    random.seed(seed)
    by_group: Dict[str, List[str]] = defaultdict(list)
    for img_path, group in items:
        by_group[group].append(img_path)

    train: List[Tuple[str, str]] = []
    val: List[Tuple[str, str]] = []
    for group, paths in sorted(by_group.items()):
        random.shuffle(paths)
        n_train = max(1, int(round(len(paths) * train_ratio)))
        # Ensure at least one in val if group has >1 images
        if len(paths) > 1 and n_train == len(paths):
            n_train = len(paths) - 1
        train.extend([(p, group) for p in paths[:n_train]])
        val.extend([(p, group) for p in paths[n_train:]])

    return train, val


def build_dataset(
    images_dir: str,
    masks_dir: str,
    background_dir: Optional[str],
    output_dir: str,
    train_ratio: float,
    seed: int,
    rdp_epsilon: float,
    min_mask_area: int,
    bg_train: int = 30,
    bg_val: int = 10,
) -> Dict[str, Any]:
    """Build the YOLOv8-seg dataset."""
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    output_dir = Path(output_dir)

    out_images_train = output_dir / "images" / "train"
    out_images_val = output_dir / "images" / "val"
    out_labels_train = output_dir / "labels" / "train"
    out_labels_val = output_dir / "labels" / "val"

    for d in (out_images_train, out_images_val, out_labels_train, out_labels_val):
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Pair training images with their SAM 2 masks
    # ------------------------------------------------------------------
    image_mask_pairs: List[Tuple[str, str, str]] = []  # (img_path, mask_path, group)
    mask_map: Dict[str, Path] = {}
    for mask_file in sorted(masks_dir.glob("*.png")):
        # Masks are named like <image_stem>_sam2.png or <image_stem>.png
        stem = mask_file.stem
        if stem.endswith("_sam2"):
            stem = stem[:-5]
        mask_map[stem] = mask_file

    species_dirs = [d for d in images_dir.iterdir() if d.is_dir()]
    for sp_dir in sorted(species_dirs):
        group = sp_dir.name
        for img_file in sorted(sp_dir.glob("*")):
            if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            stem = img_file.stem
            mask_file = mask_map.get(stem)
            if mask_file is None:
                print(f"  Warning: no mask found for {img_file}, skipping", file=sys.stderr)
                continue
            image_mask_pairs.append((str(img_file), str(mask_file), group))

    print(f"Found {len(image_mask_pairs)} image+mask pairs.")

    # ------------------------------------------------------------------
    # 2. Stratified train/val split
    # ------------------------------------------------------------------
    items = [(img, group) for img, _, group in image_mask_pairs]
    train_items, val_items = stratified_split(items, train_ratio, seed)
    train_set = set(p for p, _ in train_items)
    val_set = set(p for p, _ in val_items)

    # ------------------------------------------------------------------
    # 3. Process masks and write labels
    # ------------------------------------------------------------------
    all_stats: List[Dict[str, Any]] = []
    train_images_written = 0
    val_images_written = 0
    train_labels_written = 0
    val_labels_written = 0

    def _write_split(img_path: str, mask_path: str, is_train: bool) -> None:
        nonlocal train_images_written, val_images_written
        nonlocal train_labels_written, val_labels_written

        img_name = Path(img_path).name
        stem = Path(img_path).stem
        out_img_dir = out_images_train if is_train else out_images_val
        out_lbl_dir = out_labels_train if is_train else out_labels_val
        out_img_path = out_img_dir / img_name
        out_lbl_path = out_lbl_dir / f"{stem}.txt"

        # Copy or symlink image
        if out_img_path.exists():
            out_img_path.unlink()
        shutil.copy2(img_path, out_img_path)

        lines, stats = mask_to_yolo_polygons(
            mask_path,
            rdp_epsilon=rdp_epsilon,
            min_area_pixels=min_mask_area,
        )
        all_stats.append(stats)

        if lines:
            out_lbl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            if is_train:
                train_labels_written += len(lines)
            else:
                val_labels_written += len(lines)
        else:
            # Empty label file → background class for YOLO
            out_lbl_path.write_text("", encoding="utf-8")

        if is_train:
            train_images_written += 1
        else:
            val_images_written += 1

    for img_path, mask_path, group in image_mask_pairs:
        is_train = img_path in train_set
        _write_split(img_path, mask_path, is_train)

    # ------------------------------------------------------------------
    # 4. Background images
    # ------------------------------------------------------------------
    bg_stats = {"train": 0, "val": 0}
    if background_dir:
        bg_dir = Path(background_dir)
        bg_images = sorted([
            str(p) for p in bg_dir.glob("*")
            if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")
        ])
        random.seed(seed)
        random.shuffle(bg_images)

        n_bg_train = min(bg_train, len(bg_images))
        n_bg_val = min(bg_val, len(bg_images) - n_bg_train)
        bg_train_imgs = bg_images[:n_bg_train]
        bg_val_imgs = bg_images[n_bg_train:n_bg_train + n_bg_val]

        for img_path in bg_train_imgs:
            img_name = Path(img_path).name
            out_img_path = out_images_train / img_name
            if out_img_path.exists():
                out_img_path.unlink()
            shutil.copy2(img_path, out_img_path)
            # No label file = background
            stem = Path(img_path).stem
            (out_labels_train / f"{stem}.txt").write_text("", encoding="utf-8")
            train_images_written += 1
            bg_stats["train"] += 1

        for img_path in bg_val_imgs:
            img_name = Path(img_path).name
            out_img_path = out_images_val / img_name
            if out_img_path.exists():
                out_img_path.unlink()
            shutil.copy2(img_path, out_img_path)
            stem = Path(img_path).stem
            (out_labels_val / f"{stem}.txt").write_text("", encoding="utf-8")
            val_images_written += 1
            bg_stats["val"] += 1

        print(f"Background images: {bg_stats['train']} train, {bg_stats['val']} val")

    # ------------------------------------------------------------------
    # 5. dataset.yaml
    # ------------------------------------------------------------------
    yaml_path = output_dir / "dataset.yaml"
    yaml_text = """# YOLOv8-seg dataset for mushroom segmentation
# Use relative path so dataset is portable across machines (Colab, local, etc.)
path: .
train: images/train
val: images/val

names:
  0: mushroom
"""
    yaml_path.write_text(yaml_text, encoding="utf-8")

    summary = {
        "train_images": train_images_written,
        "val_images": val_images_written,
        "train_label_instances": train_labels_written,
        "val_label_instances": val_labels_written,
        "background_train": bg_stats["train"],
        "background_val": bg_stats["val"],
        "masks_processed": len(all_stats),
        "masks_skipped": sum(1 for s in all_stats if s.get("skipped")),
        "dataset_yaml": str(yaml_path),
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare YOLOv8-seg dataset")
    parser.add_argument("--images-dir", required=True, help="Directory of training images (species subfolders)")
    parser.add_argument("--masks-dir", required=True, help="Directory of SAM 2 binary mask PNGs")
    parser.add_argument("--background-dir", help="Directory of background-only images")
    parser.add_argument("--output-dir", required=True, help="Output directory for YOLO dataset")
    parser.add_argument("--train-ratio", type=float, default=0.80, help="Train split ratio (default 0.80)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--rdp-epsilon", type=float, default=2.0, help="RDP polygon simplification epsilon")
    parser.add_argument("--min-mask-area", type=int, default=50, help="Minimum mask area in pixels")
    parser.add_argument("--bg-train", type=int, default=30, help="Background images for training")
    parser.add_argument("--bg-val", type=int, default=10, help="Background images for validation")
    args = parser.parse_args()

    if not Path(args.images_dir).exists():
        print(f"Error: images-dir not found: {args.images_dir}", file=sys.stderr)
        return 1
    if not Path(args.masks_dir).exists():
        print(f"Error: masks-dir not found: {args.masks_dir}", file=sys.stderr)
        return 1

    print("Building YOLOv8-seg dataset...")
    print(f"  Images: {args.images_dir}")
    print(f"  Masks: {args.masks_dir}")
    print(f"  Background: {args.background_dir or 'none'}")
    print(f"  Output: {args.output_dir}")
    print()

    summary = build_dataset(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        background_dir=args.background_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
        rdp_epsilon=args.rdp_epsilon,
        min_mask_area=args.min_mask_area,
        bg_train=args.bg_train,
        bg_val=args.bg_val,
    )

    print("Dataset prepared.")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
